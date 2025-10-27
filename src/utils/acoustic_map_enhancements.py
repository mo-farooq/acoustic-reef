"""
Enhanced Acoustic Map functionality for diagnostic analysis
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def identify_acoustic_clusters(df: pd.DataFrame, method: str = 'kmeans', n_clusters: int = 3) -> Tuple[np.ndarray, Dict]:
    """
    Identify main acoustic clusters in the UMAP data
    
    Args:
        df: DataFrame with x, y coordinates and labels
        method: 'kmeans' or 'dbscan'
        n_clusters: Number of clusters for KMeans (ignored for DBSCAN)
    
    Returns:
        Tuple of (cluster_labels, cluster_info)
    """
    coords = df[['x', 'y']].values
    
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = clusterer.fit_predict(coords)
        cluster_centers = clusterer.cluster_centers_
        
        # Analyze cluster composition
        cluster_info = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_data = df[cluster_mask]
            
            # Get dominant label in this cluster
            label_counts = cluster_data['label'].value_counts()
            dominant_label = label_counts.index[0] if len(label_counts) > 0 else 'unknown'
            dominant_count = label_counts.iloc[0] if len(label_counts) > 0 else 0
            total_count = len(cluster_data)
            
            cluster_info[i] = {
                'center': cluster_centers[i],
                'size': total_count,
                'dominant_label': dominant_label,
                'dominant_count': dominant_count,
                'dominant_percentage': (dominant_count / total_count) * 100 if total_count > 0 else 0,
                'label_distribution': label_counts.to_dict()
            }
    
    elif method == 'dbscan':
        clusterer = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = clusterer.fit_predict(coords)
        
        # Analyze cluster composition
        cluster_info = {}
        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Noise points
                continue
                
            cluster_mask = cluster_labels == cluster_id
            cluster_data = df[cluster_mask]
            
            # Calculate cluster center
            center = cluster_data[['x', 'y']].mean().values
            
            # Get dominant label in this cluster
            label_counts = cluster_data['label'].value_counts()
            dominant_label = label_counts.index[0] if len(label_counts) > 0 else 'unknown'
            dominant_count = label_counts.iloc[0] if len(label_counts) > 0 else 0
            total_count = len(cluster_data)
            
            cluster_info[cluster_id] = {
                'center': center,
                'size': total_count,
                'dominant_label': dominant_label,
                'dominant_count': dominant_count,
                'dominant_percentage': (dominant_count / total_count) * 100 if total_count > 0 else 0,
                'label_distribution': label_counts.to_dict()
            }
    
    return cluster_labels, cluster_info

def create_enhanced_scatter_plot(df: pd.DataFrame, cluster_labels: np.ndarray, 
                                cluster_info: Dict, uploaded_coord: Optional[np.ndarray] = None,
                                uploaded_filename: Optional[str] = None) -> go.Figure:
    """
    Create enhanced scatter plot with cluster visualization
    """
    # Color mapping for labels
    color_map = {
        'healthy': '#2E8B57',      # Sea Green
        'degraded': '#DC143C',      # Crimson  
        'anthrophony': '#FF8C00'    # Dark Orange
    }
    
    # Create base scatter plot
    fig = go.Figure()
    
    # Add points for each cluster with different marker styles
    unique_clusters = np.unique(cluster_labels)
    cluster_names = []
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Noise points in DBSCAN
            cluster_mask = cluster_labels == cluster_id
            cluster_name = "Noise/Outliers"
            marker_symbol = "x"
        else:
            cluster_mask = cluster_labels == cluster_id
            cluster_name = f"Zone {cluster_id + 1}"
            marker_symbol = ["circle", "square", "diamond", "triangle-up", "star"][cluster_id % 5]
        
        if np.any(cluster_mask):
            cluster_data = df[cluster_mask]
            
            # Add points for this cluster
            fig.add_trace(go.Scatter(
                x=cluster_data['x'],
                y=cluster_data['y'],
                mode='markers',
                marker=dict(
                    color=[color_map.get(label, '#666666') for label in cluster_data['label']],
                    size=8,
                    symbol=marker_symbol,
                    line=dict(width=1, color='white'),
                    opacity=0.7
                ),
                text=cluster_data['filename'],
                hovertemplate="<b>%{text}</b><br>" +
                             "Label: " + cluster_data['label'] + "<br>" +
                             "Position: (%{x:.2f}, %{y:.2f})<br>" +
                             f"Zone: {cluster_name}<br>" +
                             "<extra></extra>",
                name=cluster_name,
                showlegend=True
            ))
            
            cluster_names.append(cluster_name)
    
    # Add cluster center annotations
    for cluster_id, info in cluster_info.items():
        if cluster_id != -1:  # Skip noise points
            fig.add_annotation(
                x=info['center'][0],
                y=info['center'][1],
                text=f"<b>Zone {cluster_id + 1}</b><br>" +
                     f"{info['dominant_label'].title()}<br>" +
                     f"({info['dominant_percentage']:.0f}%)",
                showarrow=True,
                arrowhead=2,
                arrowcolor="black",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10, color="black")
            )
    
    # Add uploaded point if provided
    if uploaded_coord is not None:
        fig.add_trace(go.Scatter(
            x=[uploaded_coord[0, 0]],
            y=[uploaded_coord[0, 1]],
            mode='markers',
            marker=dict(
                symbol="star",
                size=20,
                color="red",
                line=dict(width=3, color="darkred")
            ),
            name="Your Recording",
            hovertemplate=f"<b>Your Recording</b><br>" +
                         f"File: {uploaded_filename}<br>" +
                         "Position: (%{x:.2f}, %{y:.2f})<br>" +
                         "<extra></extra>"
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="🌊 Enhanced Acoustic Landscape with Diagnostic Zones",
            font=dict(size=20, color='#2c3e50'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(
                text="Acoustic Dimension 1",
                font=dict(size=14, color='#2c3e50')
            ),
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True
        ),
        yaxis=dict(
            title=dict(
                text="Acoustic Dimension 2",
                font=dict(size=14, color='#2c3e50')
            ),
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            title=dict(
                text="Acoustic Zones",
                font=dict(size=12, color='#2c3e50')
            ),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        width=800,
        height=600
    )
    
    return fig

def analyze_nearest_neighbors(df: pd.DataFrame, uploaded_coord: np.ndarray, 
                            n_neighbors: int = 5) -> Dict:
    """
    Analyze the nearest neighbors of an uploaded point
    
    Args:
        df: DataFrame with training data
        uploaded_coord: Coordinates of uploaded point
        n_neighbors: Number of neighbors to analyze
    
    Returns:
        Dictionary with neighbor analysis results
    """
    # Calculate distances to all points
    distances = np.sqrt(np.sum((df[['x', 'y']].values - uploaded_coord[0])**2, axis=1))
    
    # Get nearest neighbors
    nearest_indices = np.argsort(distances)[:n_neighbors]
    nearest_distances = distances[nearest_indices]
    nearest_data = df.iloc[nearest_indices]
    
    # Analyze neighbor composition
    label_counts = nearest_data['label'].value_counts()
    
    # Create summary
    summary_parts = []
    for label, count in label_counts.items():
        if count == 1:
            summary_parts.append(f"1 known '{label.title()}' sample")
        else:
            summary_parts.append(f"{count} known '{label.title()}' samples")
    
    summary = f"Your recording is acoustically most similar to {', '.join(summary_parts)}."
    
    return {
        'neighbors': nearest_data,
        'distances': nearest_distances,
        'summary': summary,
        'label_distribution': label_counts.to_dict(),
        'closest_match': {
            'filename': nearest_data.iloc[0]['filename'],
            'label': nearest_data.iloc[0]['label'],
            'distance': nearest_distances[0]
        }
    }

def create_trajectory_plot(base_df: pd.DataFrame, trajectory_data: List[Dict], 
                          cluster_labels: np.ndarray, cluster_info: Dict) -> go.Figure:
    """
    Create trajectory plot for batch uploads showing chronological progression
    
    Args:
        base_df: Base training data
        trajectory_data: List of dicts with 'x', 'y', 'filename', 'date', 'label'
        cluster_labels: Cluster assignments for base data
        cluster_info: Cluster information
    
    Returns:
        Plotly figure with trajectory
    """
    # Create base plot with clusters
    fig = create_enhanced_scatter_plot(base_df, cluster_labels, cluster_info)
    
    if not trajectory_data:
        return fig
    
    # Sort trajectory data by date
    trajectory_df = pd.DataFrame(trajectory_data)
    trajectory_df = trajectory_df.sort_values('date')
    
    # Add trajectory line
    fig.add_trace(go.Scatter(
        x=trajectory_df['x'],
        y=trajectory_df['y'],
        mode='lines+markers',
        line=dict(color='red', width=3, dash='solid'),
        marker=dict(size=10, color='red', symbol='circle'),
        name='Reef Trajectory',
        hovertemplate="<b>%{text}</b><br>" +
                     "Date: " + trajectory_df['date'].astype(str) + "<br>" +
                     "Position: (%{x:.2f}, %{y:.2f})<br>" +
                     "<extra></extra>",
        text=trajectory_df['filename']
    ))
    
    # Add arrows to show direction
    for i in range(len(trajectory_df) - 1):
        x0, y0 = trajectory_df.iloc[i]['x'], trajectory_df.iloc[i]['y']
        x1, y1 = trajectory_df.iloc[i + 1]['x'], trajectory_df.iloc[i + 1]['y']
        
        fig.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="red",
            opacity=0.7
        )
    
    # Update title
    fig.update_layout(
        title=dict(
            text="🌊 Acoustic Trajectory Analysis",
            font=dict(size=20, color='#2c3e50'),
            x=0.5
        )
    )
    
    return fig

def generate_acoustic_insights(uploaded_coord: np.ndarray, cluster_labels: np.ndarray, 
                              cluster_info: Dict, neighbor_analysis: Dict) -> List[str]:
    """
    Generate actionable insights based on acoustic map analysis
    
    Args:
        uploaded_coord: Coordinates of uploaded point
        cluster_labels: Cluster assignments
        cluster_info: Cluster information
        neighbor_analysis: Results from nearest neighbor analysis
    
    Returns:
        List of insight strings
    """
    insights = []
    
    # Find which cluster the uploaded point is closest to
    min_distance = float('inf')
    closest_cluster = None
    
    for cluster_id, info in cluster_info.items():
        if cluster_id != -1:  # Skip noise points
            distance = np.sqrt(np.sum((info['center'] - uploaded_coord[0])**2))
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster_id
    
    # Generate insights based on cluster analysis
    if closest_cluster is not None:
        cluster_data = cluster_info[closest_cluster]
        dominant_label = cluster_data['dominant_label']
        dominant_percentage = cluster_data['dominant_percentage']
        
        if dominant_label == 'anthrophony' and dominant_percentage > 70:
            insights.append("🔊 **High Noise Zone**: Your recording falls in an area dominated by human-made noise. Consider noise reduction strategies.")
        elif dominant_label == 'degraded' and dominant_percentage > 70:
            insights.append("⚠️ **Degraded Zone**: Your recording is in an area with degraded reef health. Focus on pollution reduction and habitat restoration.")
        elif dominant_label == 'healthy' and dominant_percentage > 70:
            insights.append("🌿 **Healthy Zone**: Your recording is in a healthy acoustic environment. Continue current conservation efforts.")
        else:
            insights.append(f"🔍 **Mixed Zone**: Your recording is in a transitional area with {dominant_percentage:.0f}% {dominant_label} characteristics.")
    
    # Generate insights based on nearest neighbors
    neighbor_distribution = neighbor_analysis['label_distribution']
    if 'anthrophony' in neighbor_distribution and neighbor_distribution['anthrophony'] >= 3:
        insights.append("🚨 **Noise Alert**: Most similar recordings show high human noise levels. Immediate noise reduction needed.")
    elif 'degraded' in neighbor_distribution and neighbor_distribution['degraded'] >= 3:
        insights.append("📉 **Health Decline**: Most similar recordings indicate reef degradation. Investigate water quality and stressors.")
    elif 'healthy' in neighbor_distribution and neighbor_distribution['healthy'] >= 3:
        insights.append("✅ **Positive Match**: Your recording closely matches healthy reef patterns. Monitor for any changes.")
    
    return insights
