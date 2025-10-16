import os
import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform, cdist
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from features import extract_features, calculate_feature_risk, NEAREST_NEIGHBORS_COUNT

# Global variables for model components
scaler = None
linkage_matrix = None
# REMOVED: cluster_map = None
cluster_stats = None # NEW: Replaces cluster_map
centroids_matrix = None
centroid_ids = None
adaptive_thresholds = None
iso_forest = None
iso_score_normalizer = None
X_scaled = None
urls = None
labels = None
clusters = None

def load_or_train_model():
    """Load existing model or train new one"""
    # CHANGED: Added cluster_stats to global scope
    global scaler, linkage_matrix, cluster_stats, centroids_matrix, centroid_ids, adaptive_thresholds, iso_forest, iso_score_normalizer, X_scaled, urls, labels, clusters

    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    model_files = {
        'scaler': os.path.join(models_dir, 'scaler.pkl'),
        'linkage': os.path.join(models_dir, 'linkage_matrix.pkl'),
        # CHANGED: 'cluster_map' is now 'cluster_stats'
        'cluster_stats': os.path.join(models_dir, 'cluster_stats.pkl'),
        'centroids': os.path.join(models_dir, 'centroids.pkl'),
        'centroid_ids': os.path.join(models_dir, 'centroid_ids.pkl'),
        'adaptive_thresholds': os.path.join(models_dir, 'adaptive_thresholds.pkl'),
        'iso_forest': os.path.join(models_dir, 'iso_forest.pkl'),
        'iso_normalizer': os.path.join(models_dir, 'iso_normalizer.pkl'),
        'data': os.path.join(models_dir, 'processed_data.pkl')
    }

    # Check if all model files exist
    if all(os.path.exists(f) for f in model_files.values()):
        print("Loading existing model...")
        try:
            scaler = joblib.load(model_files['scaler'])
            linkage_matrix = joblib.load(model_files['linkage'])
            # CHANGED: Load cluster_stats.pkl
            cluster_stats = joblib.load(model_files['cluster_stats'])
            centroids_matrix = joblib.load(model_files['centroids'])
            centroid_ids = joblib.load(model_files['centroid_ids'])
            adaptive_thresholds = joblib.load(model_files['adaptive_thresholds'])
            iso_forest = joblib.load(model_files['iso_forest'])
            iso_score_normalizer = joblib.load(model_files['iso_normalizer'])
            data = joblib.load(model_files['data'])
            X_scaled = data['X_scaled']
            urls = data['urls']
            labels = data['labels']
            clusters = data['clusters']
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new model...")

    # Train new model if loading fails or files are missing
    print("No existing model found or load failed. Training new model...")
    from train import train_model
    return train_model()


def create_dendrogram_figure(truncate_mode='lastp', p=30, color_threshold=None,
                           leaf_rotation=90, figsize=(12, 8), dpi=150):
    """Create dendrogram figure with dark theme"""
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Set dark theme background matching site design
    fig.patch.set_facecolor('#0f0f0f')
    plt.gca().set_facecolor('#0f0f0f')

    # Calculate optimal threshold for visualization that creates multiple colored clusters
    if color_threshold is None:
        distances = linkage_matrix[:, 2]

        # For truncated dendrogram (lastp, p=30), we need to consider the visible merges
        # The last 30 merges represent the highest-level clustering structure
        if truncate_mode == 'lastp' and p == 30:
            # Use a threshold that splits the visible top-level merges into multiple color groups
            # The last 30 distances range from ~65 to ~3000, so we want a threshold in this range
            last_30_distances = distances[-30:]
            # Use a threshold that creates 4-6 color groups in the visible portion
            color_threshold = np.percentile(last_30_distances, 60)  # 60th percentile of visible merges
        else:
            # For full dendrogram, use a threshold that shows hierarchical structure
            color_threshold = np.percentile(distances, 90)

    # Create dendrogram
    dendro = dendrogram(
        linkage_matrix,
        truncate_mode=truncate_mode,
        p=p,
        color_threshold=color_threshold,
        leaf_rotation=leaf_rotation,
        leaf_font_size=12,  # Increased from 8
        show_contracted=True
    )

    plt.title('Hierarchical Clustering Dendrogram', fontsize=18, fontweight='bold', color='white')  # Increased from 16
    plt.xlabel('Sample Index or (cluster size)', fontsize=14, color='white')  # Increased from 12
    plt.ylabel('Distance', fontsize=14, color='white')  # Increased from 12

    # Draw threshold line with calculated optimal threshold
    plt.axhline(y=color_threshold, color="#667eea", linestyle="--",
                label=f'Distance Threshold (t={color_threshold:.1f})')
    plt.grid(True, alpha=0.3, color=(1.0, 1.0, 1.0, 0.3))

    # Set tick colors to white for visibility
    plt.tick_params(colors='white')
    plt.gca().tick_params(colors='white')

    # Add color legend
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
    legend_elements = []
    for i, color in enumerate(colors[:len(set(dendro['color_list']))]):
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=f'Cluster {i+1}'))

    if legend_elements:
        legend = plt.legend(handles=legend_elements, loc='upper right', fontsize=12, facecolor='#0f0f0f', edgecolor=(1.0, 1.0, 1.0, 0.3))  # Increased font size
        plt.setp(legend.get_texts(), color='white')

    return fig

def find_url_position_in_dendrogram(url):
    """Find URL position in the hierarchical clustering tree and return cluster stats."""
    try:
        # Check if model components are available
        if cluster_stats is None:
            return {'url': url, 'prediction': 'unavailable', 'message': 'Cluster statistics are not available.'}
        if centroids_matrix is None or len(centroid_ids) == 0:
            return {'url': url, 'prediction': 'unavailable', 'message': 'No valid centroids found.'}

        # Extract features and find the nearest cluster
        feats = extract_features(url)
        X_new = pd.DataFrame([feats])
        X_new_scaled = scaler.transform(X_new)

        from sklearn.metrics import pairwise_distances_argmin_min
        closest, distances = pairwise_distances_argmin_min(X_new_scaled, centroids_matrix)
        cluster_id = centroid_ids[closest[0]]
        distance_to_centroid = distances[0]

        # --- MODIFIED SECTION: REFINED ENSEMBLE AND CONFIDENCE LOGIC ---

        # Component 1: Clustering score
        adaptive_threshold = adaptive_thresholds.get(cluster_id, 5.0)
        cluster_score = distance_to_centroid / adaptive_threshold if adaptive_threshold > 0 else 1.0

        # Component 2: Isolation Forest score
        iso_score_norm = min(-iso_forest.score_samples(X_new_scaled)[0] / iso_score_normalizer, 2.0) if iso_forest and iso_score_normalizer > 0 else 0

        # Component 3: Feature-based risk score
        feature_risk = calculate_feature_risk(feats)

        # Heuristic weights for the ensemble components. These were determined empirically
        # to balance the influence of structural clustering, anomaly detection, and known risk patterns.
        W_CLUSTER = 0.45 # Weight for how much of an outlier the URL is within its own cluster
        W_ANOMALY = 0.35 # Weight for how anomalous the URL is compared to the entire dataset
        W_HEURISTIC = 0.20 # Weight for known suspicious patterns (e.g., keywords, IP addresses)

        # The combined_risk_score is an unbounded score where higher values indicate higher risk.
        combined_risk_score = (
            W_CLUSTER * cluster_score +
            W_ANOMALY * iso_score_norm +
            W_HEURISTIC * feature_risk
        )

        # We convert the unbounded risk score into a bounded confidence score (0-1).
        # This linear mapping assumes a risk score of ~0.2 is very low risk (100% confidence)
        # and a score of ~1.2 is very high risk (0% confidence).
        LOW_RISK_THRESHOLD = 0.2
        HIGH_RISK_THRESHOLD = 1.2

        # Normalize the risk score to a 0-1 range
        normalized_risk = (combined_risk_score - LOW_RISK_THRESHOLD) / (HIGH_RISK_THRESHOLD - LOW_RISK_THRESHOLD)
        bounded_risk = max(0, min(1, normalized_risk))

        # Confidence is the inverse of the bounded risk
        confidence = 1.0 - bounded_risk

        # --- END OF MODIFIED SECTION ---

        # --- NEAREST NEIGHBORS (No changes here) ---
        distances_to_all = np.linalg.norm(X_scaled - X_new_scaled, axis=1)
        nearest_indices = np.argsort(distances_to_all)[:NEAREST_NEIGHBORS_COUNT]
        max_reasonable_distance = 5.0
        normalized_distances = 1 / (1 + np.exp((distances_to_all[nearest_indices] - max_reasonable_distance/2) / (max_reasonable_distance/4)))

        nearest_neighbors = [{
            'url': urls[i],
            'label': labels[i],
            'distance': float(1 - normalized_distances[idx]), # Convert similarity to distance for consistency
            'cluster': int(clusters[i])
        } for idx, i in enumerate(nearest_indices)]

        neighbor_confidence = np.mean([1 - n['distance'] for n in nearest_neighbors]) if nearest_neighbors else 0

        # --- NEW: RETURN DETAILED CLUSTER STATISTICS ---
        purity_info = cluster_stats.get(cluster_id, {
            'total_count': 0, 'phishing_count': 0, 'legitimate_count': 0,
            'purity': 0, 'majority_class': 'unknown'
        })

        # --- NEW: INCLUDE SUSPICIOUS KEYWORD COUNT IN RESPONSE ---
        suspicious_kw_count = feats.get('suspicious_kw_count', 0)

        return {
            'url': url,
            'cluster_id': int(cluster_id),
            'cluster_purity_info': purity_info,
            'confidence': confidence, # The new, more interpretable confidence
            'risk_score': float(combined_risk_score), # NEW: Also return the raw score for context
            'neighbor_confidence': neighbor_confidence,
            'distance_to_centroid': float(distance_to_centroid),
            'nearest_neighbors': nearest_neighbors,
            'suspicious_kw_count': int(suspicious_kw_count) # NEW
        }

    except Exception as e:
        return {'url': url, 'error': str(e), 'prediction': 'error'}

def create_url_cluster_analysis(url, figsize=(12, 8), dpi=150):
    """Create a comprehensive analysis showing where URL fits in the clustering tree"""
    try:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Set dark theme background matching site design
        fig.patch.set_facecolor('#0f0f0f')
        ax.set_facecolor('#0f0f0f')

        # Get URL analysis
        url_info = find_url_position_in_dendrogram(url)

        # Create a simple but informative visualization
        prediction = url_info.get('prediction', 'unknown')
        confidence = url_info.get('confidence', 0)
        cluster_id = url_info.get('cluster_id', 'N/A')
        distance = url_info.get('distance_to_centroid', 0)

        # Show cluster distribution
        cluster_counts = {}
        cluster_labels = {}

        for i, cluster in enumerate(clusters):
            if cluster not in cluster_counts:
                cluster_counts[cluster] = 0
                cluster_labels[cluster] = []
            cluster_counts[cluster] += 1
            if i < len(labels):
                cluster_labels[cluster].append(labels[i])

        # Determine cluster type based on majority label
        cluster_types = {}
        for cluster, labels_list in cluster_labels.items():
            if labels_list:
                label_counts = pd.Series(labels_list).value_counts()
                majority_label = label_counts.index[0]
                cluster_types[cluster] = majority_label

        # Create bar chart
        clusters_sorted = sorted(cluster_counts.keys())
        counts = [cluster_counts[c] for c in clusters_sorted]
        colors = ['red' if cluster_types.get(c) == 'phishing' else
                 'green' if cluster_types.get(c) == 'legitimate' else 'gray'
                 for c in clusters_sorted]

        bars = ax.bar(range(len(clusters_sorted)), counts, color=colors, alpha=0.7)

        # Highlight the URL's cluster
        if cluster_id != 'N/A' and cluster_id in clusters_sorted:
            cluster_idx = clusters_sorted.index(cluster_id)
            bars[cluster_idx].set_edgecolor('blue')
            bars[cluster_idx].set_linewidth(3)
            bars[cluster_idx].set_alpha(1.0)

        ax.set_title(f'URL Pattern Analysis: {url[:50]}...\nPattern Group: {prediction} | Similarity: {confidence:.2f} | Cluster: {cluster_id}', fontsize=16, fontweight='bold', color='white')  # Increased font size
        ax.set_xlabel('Cluster ID', fontsize=14, color='white')  # Increased font size
        ax.set_ylabel('Number of Samples', fontsize=14, color='white')  # Increased font size
        ax.set_xticks(range(len(clusters_sorted)))

        # Handle crowded x-axis labels by showing every nth label
        n_clusters = len(clusters_sorted)
        if n_clusters > 20:  # If more than 20 clusters, reduce label density
            step = max(1, n_clusters // 10)  # Show about 10 labels maximum
            visible_indices = list(range(0, n_clusters, step))
            visible_labels = [str(clusters_sorted[i]) if i < n_clusters else '' for i in visible_indices]

            # Create full range of positions but only show some labels
            ax.set_xticks(range(n_clusters))
            ax.set_xticklabels([''] * n_clusters)  # Clear all labels first
            for i, label in zip(visible_indices, visible_labels):
                ax.text(i, -0.05, label, ha='center', va='top',
                       fontsize=10, color='white', rotation=45,
                       transform=ax.get_xaxis_transform())
        else:
            ax.set_xticklabels(clusters_sorted, fontsize=12, color='white')
        ax.tick_params(axis='y', labelsize=12, colors='white')  # Increased font size for y-axis tick labels
        ax.grid(True, alpha=0.3, color=(1.0, 1.0, 1.0, 0.3))

        # Add text annotation
        ax.text(0.02, 0.98, f'URL: {url}\nPattern Group: {prediction}\nSimilarity: {confidence:.2f}\nCluster: {cluster_id}\nDistance: {distance:.3f}',
                transform=ax.transAxes, fontsize=14, color='white',  # Increased from 10
                bbox=dict(boxstyle="round,pad=0.3", facecolor=(0.4, 0.494, 0.918, 0.2), edgecolor=(0.4, 0.494, 0.918, 0.4), alpha=0.8),
                verticalalignment='top')

        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='Suspicious Pattern Groups'),
            plt.Rectangle((0,0),1,1, facecolor='green', alpha=0.7, label='Normal Pattern Groups'),
            plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.7, label='Unknown Pattern Groups'),
            plt.Rectangle((0,0),1,1, facecolor='blue', alpha=1.0, label='URL Pattern Group')
        ]
        legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=12, facecolor='#0f0f0f', edgecolor=(1.0, 1.0, 1.0, 0.3))  # Increased font size
        plt.setp(legend.get_texts(), color='white')

        return fig

    except Exception as e:
        print(f"Error creating URL cluster analysis: {e}")
        # Fallback to simple plot with dark theme
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor('#0f0f0f')
        ax.set_facecolor('#0f0f0f')
        ax.text(0.5, 0.5, f'URL Analysis Error\n{str(e)}',
                ha='center', va='center', transform=ax.transAxes, color='white', fontsize=14)
        ax.set_title('URL Cluster Analysis', color='white')
        ax.tick_params(colors='white')
        return fig

def figure_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)  # Increased DPI from 100
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    return image_base64
