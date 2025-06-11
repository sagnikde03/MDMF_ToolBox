import pandas as pd
import numpy as np
import seaborn as sns
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import AgglomerativeClustering

def generate_sankey_diagram(file_path, percentile_threshold=90, num_clusters=4):
    """
    Generates a Sankey diagram from a structural connectivity (SC) matrix CSV file.

    Parameters:
    - file_path (str): Path to the SC matrix CSV file.
    - percentile_threshold (int): Percentile for thresholding connections (default is 90).
    - num_clusters (int): Number of clusters for hierarchical clustering (default is 4).

    Returns:
    - Plotly Sankey diagram figure.
    """
    # Load the SC matrix
    sc_matrix = pd.read_csv(file_path, index_col=None)
    
    # Thresholding: Keep only connections above a percentile threshold
    threshold = np.percentile(sc_matrix.values, percentile_threshold)
    sc_matrix[sc_matrix < threshold] = 0  # Remove weak connections
    
    # Compute linkage for clustering
    linkage = sch.linkage(sc_matrix, method='ward')
    
    # Plot dendrogram
    plt.figure(figsize=(10, 5))
    sch.dendrogram(linkage, labels=sc_matrix.index, leaf_rotation=90)
    plt.title("Hierarchical Clustering of Structural Connectivity")
    plt.show()
    
    # Extract strong connections after thresholding
    sources, targets, values = [], [], []
    for i, source in enumerate(sc_matrix.index):
        for j, target in enumerate(sc_matrix.columns):
            weight = sc_matrix.iloc[i, j]
            if weight > 0:  # Only keep strong connections
                sources.append(source)
                targets.append(target)
                values.append(weight)
    
    # Apply clustering (assign labels to brain regions)
    cluster_model = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
    region_labels = cluster_model.fit_predict(sc_matrix)
    
    # Assign colors based on clusters
    color_map = ['red', 'blue', 'green', 'purple']  # Adjust colors as needed
    node_colors = [color_map[label] for label in region_labels]
    
    # Create node index mapping
    nodes = list(set(sources + targets))
    node_indices = {node: i for i, node in enumerate(nodes)}
    
    # Convert source/target names to indices
    source_indices = [node_indices[src] for src in sources]
    target_indices = [node_indices[tgt] for tgt in targets]
    
    # Build Sankey Diagram
    link_colors = ['rgba(0, 0, 0, 0.3)' if idx < len(nodes)//2 else 'rgba(144, 238, 144, 0.4)' for idx in source_indices]
    
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=10, thickness=20, label=nodes, color=node_colors
        ),
        link=dict(
            source=source_indices, target=target_indices, value=values, color=link_colors
        )
    ))
    
    fig.update_layout(title_text="Sankey Diagram of SC Matrix", font_size=12)
    fig.show()
    
    return fig
