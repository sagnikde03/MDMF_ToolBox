import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import parafac
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
import time

# binarize
_ = []

# tensors refer to the rs-dFC tensors
for t in tensors:
    a = np.percentile(t,98)
    tense = np.where(t>a, 1, 0)
    _.append(tense)
  
F = 4
# decompose and three feature pools
Feature_pool_2 = []
Feature_pool_3 = []

for tensor in _:
    factors = parafac(tensor, rank=F)
    factor_matrices = factors.factors

    factor_matrix_2 = factor_matrices[1]
    factor_matrix_3 = factor_matrices[2]
    Feature_pool_2.append(factor_matrix_2[:,0])    
    Feature_pool_2.append(factor_matrix_2[:,1])    
    Feature_pool_2.append(factor_matrix_2[:,2])    
    Feature_pool_3.append(factor_matrix_3[:,0])    
    Feature_pool_3.append(factor_matrix_3[:,1])    
    Feature_pool_3.append(factor_matrix_3[:,2])

matrix = np.corrcoef(Feature_pool_2)
correlation_matrix = np.corrcoef(matrix, rowvar=False)
distance_matrix = (1 - correlation_matrix) / 2
condensed_distance_matrix = squareform(distance_matrix, checks=False)
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
data_transformed = mds.fit_transform(distance_matrix)
# Apply KMeans to the transformed data
k = 4 
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(data_transformed)
# Get the cluster labels
labels = kmeans.labels_
# Reorder the correlation matrix according to the cluster labels
sorted_indices = np.argsort(labels)
sorted_correlation_matrix = correlation_matrix[sorted_indices, :][:, sorted_indices]
s_a = silhouette_score(data_transformed, labels)    
# Plot the reordered correlation matrix with cluster annotations
plt.figure(figsize=(6, 5))
sns.heatmap(sorted_correlation_matrix, cmap='viridis', annot=False, xticklabels=sorted_indices, yticklabels=sorted_indices)
plt.title('Correlation Matrix with K-means Clusters')
plt.axis('off')
plt.show()

