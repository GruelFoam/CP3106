import pandas as pd
import numpy as np
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from tool import *
import matplotlib.pyplot as plt
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
Load data
'''
print("=============load data=============")

# Train, test samples
train_data = torch.load("./data/train_data.pth")
train_label = torch.load("./data/train_label.pth")
test_data = torch.load("./data/test_data.pth")
test_label = torch.arange(1197)
print(f"train data shape: {train_data.shape}")
print(f"train label shape: {train_label.shape}")
print(f"test data shape: {test_data.shape}")
print(f"test label shape: {test_label.shape}")

# Firm info
exp_df = pd.read_csv('./data/firm_list.csv') # firm list
unique_cik = exp_df['cik'].unique()
cik_to_index = {cik: idx for idx, cik in enumerate(unique_cik)}
index_to_cik = {idx: cik for idx, cik in enumerate(unique_cik)}
# print("CIK to Index Mapping:", cik_to_index)

# Similarity info
dissimilar_df = pd.read_csv("./data/dissimilar_pairs_2021.csv")
print(f"length of dissimilar_df: {len(dissimilar_df)}")

similar_df = pd.read_csv("./data/pairs_gpt_competitors_2021.csv")
similar_df = similar_df[similar_df['company_a_cik'] != similar_df['company_b_cik']]
print(f"length of similar_df: {len(similar_df)}")
print("\nLoad data finished\n\n\n")



'''
Dataset preperation
'''
print("=============dataset preperation=============")

# Relation (label) matrix
N = len(unique_cik)
relation_matrix = -1 * np.ones((N, N))

for _, row in similar_df.iterrows():
    i, j = cik_to_index[row['company_a_cik']], cik_to_index[row['company_b_cik']]
    relation_matrix[i, j] = 1
    relation_matrix[j, i] = 1  # Ensure symmetry
for _, row in dissimilar_df.iterrows():
    i, j = cik_to_index[row['company_a_cik']], cik_to_index[row['company_b_cik']]
    relation_matrix[i, j] = 0
    relation_matrix[j, i] = 0  # Ensure symmetry
np.fill_diagonal(relation_matrix, -1)

relation_matrix = torch.tensor(relation_matrix, dtype=torch.float32).to(device)
print(f"relation matrix shape: {relation_matrix.shape}")

# Create dataset
class ContrastiveDataset(Dataset):
    def __init__(self, data, index_list):
        self.data = data
        self.index_list = index_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.index_list[idx]

train_dataset = ContrastiveDataset(train_data, train_label)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = ContrastiveDataset(test_data, torch.arange(1197))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
print("\nData preperation finished\n\n\n")


'''
Model training
'''
print("=============model training=============")
# Model
class ProjectionNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(ProjectionNet, self).__init__()
        
        # Define the layers sequentially
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),         # First fully connected layer
            nn.BatchNorm1d(hidden_dim),               # Batch Normalization
            nn.LeakyReLU(negative_slope=0.01),        # LeakyReLU activation
            # nn.Dropout(p=dropout_prob),               # Dropout
            nn.Linear(hidden_dim, output_dim)         # Second fully connected layer
        )
    
    def forward(self, x):
        # Pass the input through the sequential model
        return self.model(x)

# Loss function
class ContrastiveLoss1(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss1, self).__init__()
        self.margin = margin

    def forward(self, z1, labels):
        loss = 0
        for i, j in z1:
            distances = torch.norm(i - j, p=2, dim=0)
            loss += (labels * distances.pow(2)) + ((1 - labels) * torch.relu(self.margin - distances).pow(2))
        return loss.mean()
    
# Helper function for finding pairs
def get_positive_negative_pairs(batch_indices, batch_output):
    batch_size = len(batch_indices)
    
    # Extract the batch submatrix
    batch_rel_matrix = relation_matrix[batch_indices][:, batch_indices]  # Shape (B, B)

    pos_pairs = []
    neg_pairs = []

    for i, j in itertools.combinations(range(batch_size), 2):  
        rel_value = batch_rel_matrix[i, j]
        if rel_value == 1:
            pos_pairs.append((batch_output[i], batch_output[j]))  # Store actual embeddings
        elif rel_value == 0:
            neg_pairs.append((batch_output[i], batch_output[j]))

    return pos_pairs, neg_pairs

# Parameters and model definition
input_dim = 256
output_dim = 256
hidden_dim = 256
margin = 5.0
learning_rate = 0.0001
num_epochs = 50

model = ProjectionNet(input_dim, output_dim, hidden_dim).to(device)
criterion = ContrastiveLoss1(margin=margin)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    for batch_data, batch_idx in train_loader:
        batch_data = batch_data.to(device)
        batch_idx = batch_idx.to(device)

        z1 = model(batch_data)
        pos_pairs, neg_pairs = get_positive_negative_pairs(batch_idx, z1)

        loss = torch.tensor(0.0, device=z1.device, requires_grad=True)
        if len(pos_pairs) > 0:
            loss = loss + criterion(pos_pairs, torch.ones(len(pos_pairs)).to(device))
        if len(neg_pairs) > 0:
            loss = loss + criterion(neg_pairs, torch.zeros(len(neg_pairs)).to(device))

        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], train_avg_Loss: {total_loss/len(train_loader):.4f}")

    total_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_data, batch_idx in test_loader:
            batch_data = batch_data.to(device)
            batch_idx = batch_idx.to(device)

            z1 = model(batch_data)
            pos_pairs, neg_pairs = get_positive_negative_pairs(batch_idx, z1)

            loss = torch.tensor(0.0, device=z1.device, requires_grad=True)
            if len(pos_pairs) > 0:  # Check if there's at least one valid pair
                # Compute loss
                loss = loss + criterion(pos_pairs, torch.ones(len(pos_pairs)).to(device))
            if len(neg_pairs) > 0:
                loss = loss + criterion(neg_pairs, torch.zeros(len(neg_pairs)).to(device))
            
            total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], test_avg_Loss: {total_loss/len(test_loader):.4f}\n")
print("Training finished\n\n\n")



'''
Model evaluation
'''
print("=============model evaluation=============")
# Inference
model.eval()
with torch.no_grad():
    final_representation = model(test_data.to(device))
final_representation = final_representation.cpu()
print(f"Shape of final representation: {final_representation.shape}")

# Clustering
print(f"Columns from exp_df: {exp_df.columns}")
exp_df['cluster_10'] = cluster(final_representation, 10)
exp_df['cluster_100'] = cluster(final_representation, 100)

# Stock price correlation
year = 2021
mode = 'forward'  # Can be 'in-sample', 'forward', or 'backtest'
returns_long = pd.read_csv("./data/returns_long.csv")
exp_returns = pd.merge(exp_df, returns_long, on='tic', how='inner')

exp_intra_corrs_10 = intra_industry_correlations(exp_returns, "cluster_10")
exp_avg_intra_corr_10 = np.nanmean(list(exp_intra_corrs_10.values()))
print(f"\nStock price correlation for 10 clusters: {exp_avg_intra_corr_10}")
exp_intra_corrs_100 = intra_industry_correlations(exp_returns, "cluster_100")
exp_avg_intra_corr_100 = np.nanmean(list(exp_intra_corrs_100.values()))
print(f"\nStock price correlation for 100 clusters: {exp_avg_intra_corr_100}\n")

# Print the result into chart
show_cluster_graph(final_representation, exp_df['cluster_10'], "test_one")

# Pair evaluation
similar_df_results = precision_and_false_positive(similar_df, exp_df.copy(), ['cluster_10', 'cluster_100'], 10000)
print(similar_df_results['Classification_Scheme'])
print(similar_df_results['Precision'])
print(similar_df_results['False_Positive_rate'])
print("\nScript finished")
