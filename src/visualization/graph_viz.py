"""
Graph network visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation


def plot_graph_network(graph, title="Graph Network", save_path=None):
    """
    Plot PyTorch Geometric graph as NetworkX visualization
    
    Args:
        graph: PyTorch Geometric Data object
        title: Plot title
        save_path: Path to save figure
    """
    # Convert to NetworkX
    G = nx.Graph()
    num_nodes = graph.num_nodes
    G.add_nodes_from(range(num_nodes))
    
    edge_index = graph.edge_index.numpy()
    edge_attr = graph.edge_attr.numpy().flatten() if hasattr(graph, 'edge_attr') else None
    
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        weight = edge_attr[i] if edge_attr is not None else 1.0
        G.add_edge(u, v, weight=weight)
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Node features as colors
    node_features = graph.x.numpy()
    node_colors = node_features[:, 0] if node_features.shape[1] > 0 else 'lightblue'
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=300, cmap='viridis',
                          edgecolors='black', linewidths=1)
    
    if edge_attr is not None:
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[abs(w)*2 for w in weights],
                              alpha=0.6, edge_color='gray')
    else:
        nx.draw_networkx_edges(G, pos, alpha=0.6, edge_color='gray')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_temporal_evolution(graph_sequence, save_path=None, interval=500):
    """
    Create animated visualization of temporal graph evolution
    
    Args:
        graph_sequence: List of PyTorch Geometric graphs
        save_path: Path to save animation (requires ffmpeg)
        interval: Milliseconds between frames
    """
    num_timepoints = len(graph_sequence)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Convert first graph to NetworkX for layout
    G = _pyg_to_networkx(graph_sequence[0])
    pos = nx.spring_layout(G, seed=42)
    
    def update(frame):
        ax.clear()
        graph = graph_sequence[frame]
        G = _pyg_to_networkx(graph)
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                              node_size=300, edgecolors='black',
                              linewidths=1, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray', ax=ax)
        
        ax.set_title(f'Timepoint {frame+1}/{num_timepoints}',
                    fontsize=14, fontweight='bold')
        ax.axis('off')
    
    anim = FuncAnimation(fig, update, frames=num_timepoints,
                        interval=interval, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=2)
        plt.close()
    else:
        plt.show()


def plot_graph_statistics(graphs, save_path=None):
    """
    Plot graph statistics over time
    
    Args:
        graphs: List of PyTorch Geometric graphs
        save_path: Path to save figure
    """
    num_timepoints = len(graphs)
    
    # Compute statistics
    num_edges = []
    avg_degree = []
    density = []
    
    for graph in graphs:
        num_edges.append(graph.edge_index.shape[1])
        num_nodes = graph.num_nodes
        avg_degree.append(graph.edge_index.shape[1] / num_nodes)
        max_edges = num_nodes * (num_nodes - 1) / 2
        density.append(graph.edge_index.shape[1] / (2 * max_edges))
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    timepoints = range(1, num_timepoints + 1)
    
    axes[0].plot(timepoints, num_edges, marker='o', linewidth=2)
    axes[0].set_xlabel('Timepoint', fontsize=12)
    axes[0].set_ylabel('Number of Edges', fontsize=12)
    axes[0].set_title('Edge Count Evolution', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(timepoints, avg_degree, marker='s', linewidth=2, color='orange')
    axes[1].set_xlabel('Timepoint', fontsize=12)
    axes[1].set_ylabel('Average Degree', fontsize=12)
    axes[1].set_title('Average Degree Evolution', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(timepoints, density, marker='^', linewidth=2, color='green')
    axes[2].set_xlabel('Timepoint', fontsize=12)
    axes[2].set_ylabel('Graph Density', fontsize=12)
    axes[2].set_title('Density Evolution', fontsize=12, fontweight='bold')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def _pyg_to_networkx(graph):
    """Helper: Convert PyTorch Geometric to NetworkX"""
    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))
    
    edge_index = graph.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        G.add_edge(u, v)
    
    return G
