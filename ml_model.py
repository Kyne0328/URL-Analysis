import os
import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram
import joblib
from scipy.stats import percentileofscore
import warnings
warnings.filterwarnings('ignore')

from features import extract_features, calculate_feature_risk, NEAREST_NEIGHBORS_COUNT

# Global model components
scaler = None
linkage_matrix = None
cluster_stats = None
centroids_matrix = None
centroid_ids = None
adaptive_thresholds = None
iso_forest = None
iso_score_normalizer = None
X_scaled = None
tsne_mapper = None
feature_names = None
tsne_data = None
urls = None
labels = None
clusters = None

# Load existing model or train new one
def load_or_train_model():
    global scaler, linkage_matrix, cluster_stats, centroids_matrix, centroid_ids, adaptive_thresholds, iso_forest, iso_score_normalizer, X_scaled, urls, labels, clusters, tsne_mapper, tsne_data, feature_names

    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    model_files = {
        'scaler': os.path.join(models_dir, 'scaler.pkl'),
        'linkage': os.path.join(models_dir, 'linkage_matrix.pkl'),
        'cluster_stats': os.path.join(models_dir, 'cluster_stats.pkl'),
        'centroids': os.path.join(models_dir, 'centroids.pkl'),
        'centroid_ids': os.path.join(models_dir, 'centroid_ids.pkl'),
        'adaptive_thresholds': os.path.join(models_dir, 'adaptive_thresholds.pkl'),
        'iso_forest': os.path.join(models_dir, 'iso_forest.pkl'),
        'iso_normalizer': os.path.join(models_dir, 'iso_normalizer.pkl'),
        'feature_names': os.path.join(models_dir, 'feature_names.pkl'),
        'data': os.path.join(models_dir, 'processed_data.pkl'),
        'tsne_mapper': os.path.join(models_dir, 'tsne_mapper.pkl'),
        'tsne_data': os.path.join(models_dir, 'tsne_data.pkl'),
    }

    # Check if all model files exist
    if all(os.path.exists(f) for f in model_files.values()):
        print("Loading existing model...")
        try:
            scaler = joblib.load(model_files['scaler'])
            linkage_matrix = joblib.load(model_files['linkage'])
            cluster_stats = joblib.load(model_files['cluster_stats'])
            centroids_matrix = joblib.load(model_files['centroids'])
            centroid_ids = joblib.load(model_files['centroid_ids'])
            adaptive_thresholds = joblib.load(model_files['adaptive_thresholds'])
            iso_forest = joblib.load(model_files['iso_forest'])
            iso_score_normalizer = joblib.load(model_files['iso_normalizer'])
            feature_names = joblib.load(model_files['feature_names'])
            tsne_mapper = joblib.load(model_files['tsne_mapper'])
            tsne_data = joblib.load(model_files['tsne_data'])
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


# Create dendrogram figure with dark theme
def create_dendrogram_figure(truncate_mode='lastp', p=30, color_threshold=None,
                           leaf_rotation=90, figsize=(12, 8), dpi=150):
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Use dark theme to match site design
    fig.patch.set_facecolor('#0f0f0f')
    plt.gca().set_facecolor('#0f0f0f')

    # Calculate threshold for clear cluster visualization
    if color_threshold is None:
        distances = linkage_matrix[:, 2]

        if truncate_mode == 'lastp' and p == 30:
            # Focus on visible merges to create distinct color groups
            last_30_distances = distances[-30:]
            color_threshold = np.percentile(last_30_distances, 60)
        else:
            # Show overall hierarchical structure
            color_threshold = np.percentile(distances, 90)

    # color palette for dendrogram
    cm = plt.get_cmap('tab20')
    custom_palette = [matplotlib.colors.rgb2hex(cm(i)) for i in range(cm.N)]
    sch.set_link_color_palette(custom_palette)

    # Create dendrogram
    dendro = dendrogram(
        linkage_matrix,
        truncate_mode=truncate_mode,
        p=p,
        color_threshold=color_threshold,
        leaf_rotation=leaf_rotation,
        leaf_font_size=12,
        show_contracted=True,
        above_threshold_color='#666666'
    )

    # Reset palette to default to avoid side effects elsewhere
    sch.set_link_color_palette(None)

    plt.title('Hierarchical Clustering Dendrogram', fontsize=18, fontweight='bold', color='white')
    plt.xlabel('Sample Index or (cluster size)', fontsize=14, color='white')
    plt.ylabel('Distance', fontsize=14, color='white')

    # Show threshold line
    plt.axhline(y=color_threshold, color="#667eea", linestyle="--",
                label=f'Distance Threshold (t={color_threshold:.1f})')
    plt.grid(True, alpha=0.3, color=(1.0, 1.0, 1.0, 0.3))

    # Use white text for dark background
    plt.tick_params(colors='white')
    plt.gca().tick_params(colors='white')

    # Dynamically create the legend based on the actual colors used in the plot.
    # First, get the default color used for lines above the threshold.
    default_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

    # Get all unique colors from the dendrogram's line collection.
    unique_colors = set(dendro['color_list'])
    # Remove the default color so it doesn't show up in the legend.
    unique_colors.discard(default_color)

    # Create a legend entry for each unique cluster color.
    legend_elements = []
    # Sorting ensures the legend order is consistent.
    for i, color in enumerate(sorted(list(unique_colors))):
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=f'Cluster {i+1}'))

    if legend_elements:
        legend = plt.legend(handles=legend_elements, loc='upper right', fontsize=12, facecolor='#0f0f0f', edgecolor=(1.0, 1.0, 1.0, 0.3))
        plt.setp(legend.get_texts(), color='white')

    return fig

# Prepare data for feature comparison radar chart
def get_feature_comparison_data(url_scaled_vector, cluster_id):
    if centroids_matrix is None or X_scaled is None or feature_names is None:
        return None

    # Focus on key features for clear visualization
    RADAR_FEATURES = [
        "url_length", "domain_entropy", "suspicious_kw_count",
        "path_length", "num_slashes", "special_char_ratio"
    ]

    try:
        # Get cluster centroid for comparison
        cluster_idx = centroid_ids.index(cluster_id)
        centroid_vector = centroids_matrix[cluster_idx]

        url_percentiles = []
        centroid_percentiles = []

        for feature in RADAR_FEATURES:
            feature_idx = feature_names.index(feature)
            # Convert to percentiles for standardized comparison
            url_percentiles.append(percentileofscore(X_scaled[:, feature_idx], url_scaled_vector[feature_idx]))
            centroid_percentiles.append(percentileofscore(X_scaled[:, feature_idx], centroid_vector[feature_idx]))

        return {"labels": RADAR_FEATURES, "url_values": url_percentiles, "centroid_values": centroid_percentiles}
    except (ValueError, IndexError) as e:
        print(f"Error getting feature comparison data: {e}")
        return None

# Prepare data for t-SNE neighborhood visualization
def get_tsne_visualization_data(url_scaled_features, neighbor_urls):
    if tsne_mapper is None or tsne_data is None:
        return None

    try:
        # Map URL to 2D space for visualization
        url_2d = tsne_mapper.predict([url_scaled_features])[0].tolist()

        # Map neighbor URLs to 2D space
        neighbor_indices = [urls.index(n['url']) for n in neighbor_urls if n['url'] in urls]
        neighbor_vectors = X_scaled[neighbor_indices]
        neighbors_2d = tsne_mapper.predict(neighbor_vectors).tolist()

        return {
            'background_data': tsne_data,
            'analyzed_url_coord': {'x': url_2d[0], 'y': url_2d[1]},
            'neighbor_coords': [{'x': coord[0], 'y': coord[1]} for coord in neighbors_2d]
        }
    except Exception as e:
        print(f"Error getting t-SNE visualization data: {e}")
        return None

# Format cluster statistics for purity plot
def get_purity_plot_data():
    if cluster_stats is None:
        return []

    plot_data = []
    for cluster_id, stats in cluster_stats.items():
        plot_data.append({
            'x': stats['total_count'],
            'y': stats['purity'] * 100,
            'label': f"Cluster {cluster_id}",
            'majority_class': stats['majority_class'],
            'phishing_count': stats['phishing_count'],
            'legitimate_count': stats['legitimate_count']
        })
    return plot_data

# Format cluster distribution data for bar chart
def get_cluster_distribution_data():
    if cluster_stats is None:
        return []

    distribution_data = []
    for cluster_id, stats in cluster_stats.items():
        distribution_data.append({
            'cluster_id': int(cluster_id),
            'total_count': stats['total_count'],
            'phishing_count': stats['phishing_count'],
            'legitimate_count': stats['legitimate_count'],
            'purity': stats['purity'] * 100,
            'majority_class': stats['majority_class']
        })

    # Ensure consistent ordering
    distribution_data.sort(key=lambda x: x['cluster_id'])
    return distribution_data

# Find URL position in hierarchical clustering tree
def find_url_position_in_dendrogram(url):
    try:
        # Verify model components are loaded
        if cluster_stats is None:
            return {'url': url, 'prediction': 'unavailable', 'message': 'Cluster statistics are not available.'}
        if centroids_matrix is None or len(centroid_ids) == 0:
            return {'url': url, 'prediction': 'unavailable', 'message': 'No valid centroids found.'}

        # Extract features and find nearest cluster
        feats = extract_features(url)
        X_new = pd.DataFrame([feats], columns=feature_names)
        X_new_scaled = scaler.transform(X_new)

        from sklearn.metrics import pairwise_distances_argmin_min
        closest, distances = pairwise_distances_argmin_min(X_new_scaled, centroids_matrix)
        cluster_id = centroid_ids[closest[0]]
        distance_to_centroid = distances[0]

        # Calculate ensemble risk score from multiple components
        adaptive_threshold = adaptive_thresholds.get(cluster_id, 5.0)
        cluster_score = distance_to_centroid / adaptive_threshold if adaptive_threshold > 0 else 1.0

        iso_score_norm = min(-iso_forest.score_samples(X_new_scaled)[0] / iso_score_normalizer, 2.0) if iso_forest and iso_score_normalizer > 0 else 0

        feature_risk = calculate_feature_risk(feats)

        # Combine components with empirically determined weights
        W_CLUSTER = 0.45
        W_ANOMALY = 0.35
        W_HEURISTIC = 0.20

        combined_risk_score = (
            W_CLUSTER * cluster_score +
            W_ANOMALY * iso_score_norm +
            W_HEURISTIC * feature_risk
        )

        # Convert to bounded confidence score
        LOW_RISK_THRESHOLD = 0.2
        HIGH_RISK_THRESHOLD = 1.2

        normalized_risk = (combined_risk_score - LOW_RISK_THRESHOLD) / (HIGH_RISK_THRESHOLD - LOW_RISK_THRESHOLD)
        bounded_risk = max(0, min(1, normalized_risk))

        confidence = 1.0 - bounded_risk

        # Find nearest neighbors in feature space
        distances_to_all = np.linalg.norm(X_scaled - X_new_scaled, axis=1)
        nearest_indices = np.argsort(distances_to_all)[:NEAREST_NEIGHBORS_COUNT]
        max_reasonable_distance = 5.0
        normalized_distances = 1 / (1 + np.exp((distances_to_all[nearest_indices] - max_reasonable_distance/2) / (max_reasonable_distance/4)))

        nearest_neighbors = [{
            'url': urls[i],
            'label': labels[i],
            'distance': float(1 - normalized_distances[idx]),
            'cluster': int(clusters[i])
        } for idx, i in enumerate(nearest_indices)]

        neighbor_confidence = np.mean([1 - n['distance'] for n in nearest_neighbors]) if nearest_neighbors else 0

        # Get cluster statistics for presentation
        purity_info = cluster_stats.get(cluster_id, {
            'total_count': 0, 'phishing_count': 0, 'legitimate_count': 0,
            'purity': 0, 'majority_class': 'unknown'
        })

        # Determine pattern group classification
        pattern_group = 'Unknown Pattern'
        pattern_style = 'mixed'
        pattern_icon = 'fas fa-question-circle'

        if purity_info and purity_info['total_count'] > 0:
            purity = purity_info.get('purity', 0)
            majority_class = purity_info.get('majority_class')

            if purity >= 0.70:
                if majority_class == 'phishing':
                    pattern_group = 'High-Risk Pattern Group'
                    pattern_style = 'suspicious'
                    pattern_icon = 'fas fa-exclamation-triangle'
                else:
                    pattern_group = 'Low-Risk Pattern Group'
                    pattern_style = 'safe'
                    pattern_icon = 'fas fa-check-circle'
            else:
                pattern_group = 'Mixed-Signal Pattern Group'
                pattern_style = 'mixed'
                pattern_icon = 'fas fa-exclamation-circle'

        suspicious_kw_count = feats.get('suspicious_kw_count', 0)

        return {
            'url': url,
            'cluster_id': int(cluster_id),
            'cluster_purity_info': purity_info,
            'confidence': confidence,
            'risk_score': float(combined_risk_score),
            'neighbor_confidence': neighbor_confidence,
            'distance_to_centroid': float(distance_to_centroid),
            'nearest_neighbors': nearest_neighbors,
            'suspicious_kw_count': int(suspicious_kw_count),
            'pattern_group': pattern_group,
            'pattern_style': pattern_style,
            'pattern_icon': pattern_icon,
            'raw_features': feats,
            'scaled_features': X_new_scaled.tolist()
        }

    except Exception as e:
        return {'url': url, 'error': str(e), 'prediction': 'error'}


# Convert matplotlib figure to base64 string
def figure_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    return image_base64
