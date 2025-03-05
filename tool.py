import pandas as pd
import json
from sklearn.cluster import KMeans
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
import torch


# ============================================================================================
# General data processing steps
# ============================================================================================
# Obtain total_embedding (before convert_to_array)
def obtain_total_embedding(data_root):
    item1_embedding = pd.read_csv(data_root+'merged_1197.csv')
    item1_embedding = item1_embedding[['cik', 'tic', 'Year', 'item1_embeddings', 'GICS_Sector']]
    print(f"Length of item1_embedding: {len(item1_embedding)}")

    other_embedding = pd.read_csv(data_root+'output_embeddings_2.csv')
    other_embedding = other_embedding[['cik', 'SP_SHORT_DESC_embeddings', 'SP_LONG_DESC_embeddings', 'ORBIS_PROD_SERV_embeddings', 'ORBIS_OVERVIEW_embeddings']]
    print(f"Length of other_embedding: {len(other_embedding)}")

    total_embedding = pd.merge(item1_embedding, other_embedding, on=['cik'])

    # To reduce mem consumption
    item1_embedding = ''
    other_embedding = ''

    return total_embedding


# helper function for convert_to_array
def parse_nan_string(x, feature_num):
    """
    Convert a string representation of a list containing 'nan' into a NumPy array.
    If the input is NaN, return a NumPy array filled with np.nan.
    """
    if pd.notna(x):
        # Replace 'nan' with 'null' to make it valid JSON
        x_cleaned = x.replace("nan", "null")
        # Parse the JSON string into a Python list
        parsed_list = json.loads(x_cleaned)
        # Convert 'null' (parsed as None) back to np.nan
        return np.array([np.nan if item is None else item for item in parsed_list])
    else:
        # Return an array of np.nan if the input is NaN
        return np.full(feature_num, np.nan)

# Load the string type list to np.array
def convert_to_array(data_df, info_list, target_list, feature_num, ignore_nan):
    '''
    data_df     (dataframe): Entire dataset;
    info_list   (string list): Columns that don't need to be processed like 'cik', 'Year';
    target_list (strign list): The columns that needed to extract;
    feature_num (integer): Length of each datapoint;
    ignore_nan  (boolean): whether ignoring the rows that have missing embeddings
    '''
    
    data_df = data_df[info_list + target_list]
    if ignore_nan:
        data_df = data_df.dropna(how='any')

    for target in target_list:
        data_df[target] = data_df[target].apply(lambda x: parse_nan_string(x, feature_num))
    
    return data_df

# ============================================================================================







# ============================================================================================
# Tools for neural network
# ============================================================================================
'''
convert original embeddings to new latent space with trained_ae and trained_clasf
'''
def safe_inference(model, input_tensor):
    '''
    Passes the input tensor through the network,
    skipping rows containing only NaNs while preserving their original positions in the output.
    '''
    # Create a mask to identify NaN rows
    nan_mask = torch.isnan(input_tensor).all(dim=1)  # True for rows that are fully NaN
    
    # Extract valid (non-NaN) rows
    valid_rows = input_tensor[~nan_mask]  # Select rows where nan_mask is False
    
    with torch.no_grad():
        valid_output = model(valid_rows)

    if isinstance(valid_output, tuple):
        _, valid_output = valid_output
    
    # Create an output tensor filled with NaNs
    output = torch.full((input_tensor.shape[0], valid_output.shape[1]), float('nan'), device=input_tensor.device)
    
    # Insert computed values into the non-NaN positions
    output[~nan_mask] = valid_output
    
    return output
# ============================================================================================




# ============================================================================================
# Obtain stock price return
# ============================================================================================
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
# ============================================================================================



# ============================================================================================
# Correlation evaluation
# ============================================================================================
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
# ============================================================================================




# ============================================================================================
# Cluster evaluation
# ============================================================================================
def cluster(target, n_clusters):
    # Convert to a numpy array
    X = np.array(target)

    # Initialize and fit the k-means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    # Get cluster assignments
    labels = kmeans.labels_
    # print("Cluster Labels:", labels)

    # # Get cluster centers
    # centers = kmeans.cluster_centers_
    # print("Cluster Centers:", centers)

    # wcss = np.sum((X - centers[labels])**2)
    wcss = kmeans.inertia_
    normalized_wcss = wcss / (len(target) * target.shape[1])
    print("Normalized Within-Cluster Sum of Squares (WCSS):", normalized_wcss)

    return labels



def show_cluster_graph(data, label):
    # Dimensionality reduction to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    data_2d = tsne.fit_transform(data)

    # Plotting the results
    plt.figure(figsize=(10, 8))

    # Scatter plot for each cluster
    for cluster in np.unique(label):
        cluster_points = data_2d[label == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")

    # Adding labels and legend
    plt.title("Clustering Results in 2D")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.show()
# ============================================================================================




# ============================================================================================
# For pair evaluation (proportion of positive/negative in clusters)
# ============================================================================================
def check_missing_ciks(pairs_df, main_df):
    """
    Check for missing CIKs in either dataset and print summary
    Returns number of valid pairs for denominator calculation
    """
    # Check missing in pairs dataset
    missing_cik_a = pairs_df['company_a_cik'].isna().sum()
    missing_cik_b = pairs_df['company_b_cik'].isna().sum()
    
    # Check CIKs not found in main dataset
    main_ciks = set(main_df['CIK'])
    not_found_cik_a = sum(~pairs_df['company_a_cik'].isin(main_ciks))
    not_found_cik_b = sum(~pairs_df['company_b_cik'].isin(main_ciks))
    
    print(f"Missing CIKs in pairs dataset: {missing_cik_a} (company A), {missing_cik_b} (company B)")
    print(f"CIKs not found in main dataset: {not_found_cik_a} (company A), {not_found_cik_b} (company B)")
    
    # Total valid pairs are those where both CIKs exist and are found in main dataset
    total_valid = len(pairs_df) - max(missing_cik_a + missing_cik_b, not_found_cik_a + not_found_cik_b)
    print(f"Total valid pairs: {total_valid}")
    
    return total_valid

def get_industry_assignments(cik, main_df, column):
    """
    Helper function to safely get industry assignment for a CIK
    Returns None if CIK not found
    """
    try:
        return main_df[main_df['CIK'] == cik][column].iloc[0]
    except (IndexError, KeyError):
        return None

def calculate_pair_match(pairs_df, main_df, column):
    """
    For a given industry classification column:
    - Get assignments for both CIKs in each pair
    - Check if they match
    Returns tuple of (matching_pairs, total_valid_pairs)
    """
    matching_pairs = 0
    valid_pairs = 0
    
    for _, row in pairs_df.iterrows():
        a_industry = get_industry_assignments(row['company_a_cik'], main_df, column)
        b_industry = get_industry_assignments(row['company_b_cik'], main_df, column)
        
        if a_industry is not None and b_industry is not None:
            valid_pairs += 1
            if a_industry == b_industry:
                matching_pairs += 1
    
    return matching_pairs, valid_pairs

def get_industry_statistics(main_df, column):
    """
    Calculate statistics for a given industry classification
    Returns tuple of (n_industries, avg_firms_per_industry)
    """
    industry_counts = main_df[column].value_counts()
    n_industries = len(industry_counts)
    avg_firms = industry_counts.mean()
    return n_industries, avg_firms

def evaluate_all_classifications(pairs_df, main_df, classification_columns):
    """
    Modified to include industry statistics
    """
    results = []
    total_valid_pairs = check_missing_ciks(pairs_df, main_df)
    
    for column in classification_columns:
        matching_pairs, valid_pairs = calculate_pair_match(pairs_df, main_df, column)
        accuracy = (matching_pairs / valid_pairs * 100) if valid_pairs > 0 else 0
        
        # Get industry statistics
        n_industries, avg_firms = get_industry_statistics(main_df, column)
        
        results.append({
            'Classification_Scheme': column,
            'Matching_Pairs': matching_pairs,
            'Valid_Pairs': valid_pairs,
            'Accuracy_Percentage': accuracy,
            'N_Industries': n_industries,
            'Avg_Firms_per_Industry': avg_firms
        })
    
    results_df = pd.DataFrame(results)
    return results_df
# ============================================================================================







# ============================================================================================
# For pair evaluation (precision and false positive rate when only considering positive pairs)
# ============================================================================================
def convert_to_sorted_tuples(df, col1, col2):
    '''
    Convert the dataframe from "pairs_gpt_competitors_2021.csv" to a set of sorted tuple.
    '''
    result_set = set(tuple(sorted([row[col1], row[col2]])) for _, row in df.iterrows())
    return result_set

def sample_cluster_pairs(df, cluster_col, index_col, sample_size=1):
    '''
    This function samples several firms from every cluster,
    and generates a set of pairs base on the sampled ones.
    '''
    # List to store all pairs
    all_pairs = set()
    grouped = df.groupby(cluster_col)

    for _, group in grouped:
        indices = group[index_col].tolist()
        
        # Make sure sample size is smaller than the number of this cluster
        sample_size_for_one_group = min(sample_size, len(indices))

        sampled_indices = random.sample(indices, sample_size_for_one_group)

        # Create a set of pairs between sampled indices and every other item in the cluster
        for i in range(len(sampled_indices)):
            # Pairs between sampled firms
            for j in range(i + 1, len(sampled_indices)):
                pair = tuple(sorted([sampled_indices[i], sampled_indices[j]]))
                all_pairs.add(pair)
                
            # Pairs between sampled item and every other item in the cluster
            for idx in indices:
                if idx not in sampled_indices:
                    pair = tuple(sorted([sampled_indices[i], idx]))
                    all_pairs.add(pair)
    return all_pairs

def calculate_pair_set(real_pair_set, main_df, column, sample_size=1):
    """
    This function calculates the precision and false positive rate for each column.
    """

    # Drop rows that contain "nan" in the target column
    temp = main_df.dropna(subset=[column])
    sampled_pair_set = sample_cluster_pairs(temp, column, "CIK", sample_size)

    # Remove valid pairs that contains firm don't appear in "temp"
    # This is because some classification scheme cannot do clustering for every firm.
    existing_cik_set = temp['CIK'].unique()
    filtered_real_pair_set = {t for t in real_pair_set if t[0] in existing_cik_set and t[1] in existing_cik_set}
    print(f"========================================\n{column}")
    print(f"Original number of positive pairs: {len(real_pair_set)}")
    print(f"Number of positive pairs after filtering: {len(filtered_real_pair_set)}\n\n")

    # Intersection of the sampled and the total pairs.
    common_set = filtered_real_pair_set & sampled_pair_set

    # Final calculation
    len_filtered_real_pair_set = len(filtered_real_pair_set)
    len_sampled_pair_set = len(sampled_pair_set)
    len_common_set = len(common_set)
    precision = len_common_set / len_filtered_real_pair_set
    false_positive = (len_sampled_pair_set-len_common_set) / len_sampled_pair_set
        
    return precision, false_positive

def precision_and_false_positive(pair_df, main_df, classification_columns, sample_size=1):
    '''
    pair_df: DataFrame loaded from "pairs_gpt_competitors_2021.csv".

    main_df: DataFrame containing classification results. It must have a column named "CIK",
            while the remaining columns represent classification results.

    classification_columns: List of column names in `main_df` that contain classification results.

    sample_size: Integer representing the number of firms sampled from each cluster.  
    '''
    main_df.rename(columns={"cik": "CIK"}, inplace=True)

    results = []
    total_valid_pairs = check_missing_ciks(pair_df, main_df)

    # Obtain every positive pairs
    real_pair_set = convert_to_sorted_tuples(pair_df, "company_a_cik", "company_b_cik")
    
    for column in classification_columns:
        # Get industry statistics
        n_industries, avg_firms = get_industry_statistics(main_df, column)

        precision, false_positive = calculate_pair_set(real_pair_set, main_df, column, sample_size)
        
        results.append({
            'Classification_Scheme': column,
            'Precision': precision,
            'False_Positive_rate': false_positive,
            'N_Industries': n_industries,
            'Avg_Firms_per_Industry': avg_firms
        })
    
    results_df = pd.DataFrame(results)
    return results_df

# ============================================================================================
