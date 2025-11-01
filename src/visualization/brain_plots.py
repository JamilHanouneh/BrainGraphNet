"""
Brain network visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx


def plot_brain_network(connectivity, threshold=0.3, save_path=None):
    """
    Plot brain connectivity as 2D network
    
    Args:
        connectivity: (N, N) connectivity matrix
        threshold: Threshold for edge display
        save_path: Path to save figure
    """
    # Create graph
    G = nx.Graph()
    num_nodes = connectivity.shape[0]
    G.add_nodes_from(range(num_nodes))
    
    # Add edges above threshold
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if abs(connectivity[i, j]) > threshold:
                G.add_edge(i, j, weight=connectivity[i, j])
    
    # Layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Draw nodes
    node_sizes = [G.degree(i) * 50 for i in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color='lightblue',
        edgecolors='black',
        linewidths=1
    )
    
    # Draw edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # Color edges by weight
    nx.draw_networkx_edges(
        G, pos,
        width=[abs(w)*3 for w in weights],
        alpha=0.6,
        edge_color=weights,
        edge_cmap=plt.cm.RdBu_r,
        edge_vmin=-1,
        edge_vmax=1
    )
    
    plt.title(f'Brain Connectivity Network (threshold={threshold})', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_brain_connectivity_3d(connectivity, coordinates=None, threshold=0.3, save_path=None):
    """
    Plot brain connectivity in 3D space
    
    Args:
        connectivity: (N, N) connectivity matrix
        coordinates: (N, 3) 3D coordinates of brain regions (optional)
        threshold: Threshold for edge display
        save_path: Path to save figure
    """
    num_nodes = connectivity.shape[0]
    
    # Generate coordinates if not provided (sphere layout)
    if coordinates is None:
        phi = np.linspace(0, 2*np.pi, num_nodes)
        theta = np.linspace(0, np.pi, num_nodes)
        coordinates = np.column_stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot nodes
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        c='lightblue',
        s=100,
        edgecolors='black',
        linewidths=1,
        alpha=0.8
    )
    
    # Plot edges
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if abs(connectivity[i, j]) > threshold:
                x = [coordinates[i, 0], coordinates[j, 0]]
                y = [coordinates[i, 1], coordinates[j, 1]]
                z = [coordinates[i, 2], coordinates[j, 2]]
                
                color = 'red' if connectivity[i, j] > 0 else 'blue'
                alpha = abs(connectivity[i, j])
                
                ax.plot(x, y, z, color=color, alpha=alpha*0.5, linewidth=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Brain Connectivity (threshold={threshold})', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_circular_brain_network(connectivity, region_names=None, threshold=0.3, save_path=None):
    """
    Plot brain network in circular layout
    
    Args:
        connectivity: (N, N) connectivity matrix
        region_names: List of region names (optional)
        threshold: Threshold for edge display
        save_path: Path to save figure
    """
    num_nodes = connectivity.shape[0]
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Circular positions
    angles = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)
    pos = {i: (np.cos(angle), np.sin(angle)) for i, angle in enumerate(angles)}
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if abs(connectivity[i, j]) > threshold:
                G.add_edge(i, j, weight=connectivity[i, j])
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue', 
                          edgecolors='black', linewidths=2, ax=ax)
    
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    nx.draw_networkx_edges(
        G, pos,
        width=[abs(w)*2 for w in weights],
        alpha=0.5,
        edge_color=weights,
        edge_cmap=plt.cm.RdBu_r,
        ax=ax
    )
    
    if region_names:
        labels = {i: region_names[i] if i < len(region_names) else str(i) 
                 for i in range(num_nodes)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    ax.set_title('Circular Brain Network Layout', fontsize=14, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
