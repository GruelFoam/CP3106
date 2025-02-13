import pandas as pd
import json
from sklearn.cluster import KMeans
import numpy as np
import yfinance as yf

def fetch_stock_data(df_full_set, year, mode):
    # Filter data for the given year
    df_processed = df_full_set[df_full_set.Year == year]

    # Get unique tickers
    tickers = df_processed['tic'].unique().tolist()

    # Define the period based on the mode
    if mode == 'in-sample':
        # In-sample mode: start and end within the same year
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
    elif mode == 'forward':
        # Forward mode: next two years
        start_date = f'{year+1}-01-01'
        end_date = f'{year+2}-12-31'
    elif mode == 'backtest':
        # Backtest mode: last 10 years, up until year-1
        start_date = f'{year-10}-01-01'
        end_date = f'{year-1}-12-31'
    else:
        raise ValueError("Mode must be 'in-sample', 'forward', or 'backtest'")

    # Fetch historical stock data for all tickers
    stock_data = yf.download(tickers, threads= False, start=start_date, end=end_date)

    # Get adjusted close prices
    close_prices = stock_data['Adj Close']

    # Drop columns where all values are NaN
    close_prices = close_prices.dropna(axis=1, how='all')

    # Calculate daily returns
    returns = close_prices.pct_change()

    # Melt the returns DataFrame to long format for merging
    returns_long = returns.reset_index().melt(id_vars='Date', var_name='tic', value_name='Return')

# Merge the returns with the original dataframe
    return returns_long



def intra_industry_correlations(df, classification_column):
    intra_matrix = []
    industry_groups = df.groupby(classification_column)
    intra_correlations = {}
    for industry, group in industry_groups:
        pivot_table = group.pivot(index='Date', columns='tic', values='Return')
        corr_matrix = pivot_table.corr()
        # Take the upper triangle of the correlation matrix without the diagonal
        intra_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean()
        intra_correlations[industry] = intra_corr
    return intra_correlations


def cluster(target, n_clusters):
    # Convert to a numpy array
    X = np.array(target)

    # Initialize and fit the k-means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    # Get cluster assignments
    labels = kmeans.labels_
    print("Cluster Labels:", labels)

    # Get cluster centers
    centers = kmeans.cluster_centers_
    print("Cluster Centers:", centers)

    wcss = np.sum((X - centers[labels])**2)
    print("Within-Cluster Sum of Squares (WCSS):", wcss)

    return labels