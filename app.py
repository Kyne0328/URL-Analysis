import os
import base64
import io
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from flask import Flask, render_template, request, jsonify, send_file
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform, cdist
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration constants
NEAREST_NEIGHBORS_COUNT = 10  # Number of nearest neighbors to analyze
SUSPICIOUS_KEYWORDS = ("login", "secure", "update", "verify", "account", "signin", 
                       "banking", "paypal", "confirm", "suspend", "wallet", "password")

app = Flask(__name__)

# Global variables for model components
scaler = None
linkage_matrix = None
cluster_map = None
centroids_matrix = None
centroid_ids = None
adaptive_thresholds = None
iso_forest = None
iso_score_normalizer = None
X_scaled = None
urls = None
labels = None
clusters = None

def calculate_entropy(s):
    """Calculate Shannon entropy of a string"""
    if not s or len(s) == 0:
        return 0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * np.log2(p + 1e-10) for p in probs)

def extract_features(url):
    """Extract features from URL for clustering - enhanced with entropy and risk indicators"""
    # Parse URL components
    if "://" in url:
        protocol, rest = url.split("://", 1)
    else:
        protocol = "http"  # Default to http if no protocol
        rest = url

    # Split domain and path
    if "/" in rest:
        domain = rest.split("/")[0]
        path = "/" + "/".join(rest.split("/")[1:])
    else:
        domain = rest
        path = ""

    # Extract domain parts
    domain_parts = domain.split(".")
    
    # Normalize domain for consistent clustering (remove www. prefix)
    normalized_domain = domain.replace("www.", "", 1) if domain.startswith("www.") else domain
    normalized_parts = normalized_domain.split(".")
    
    # Calculate subdomain (excluding www for consistent clustering)
    # www.example.com should have same subdomain features as example.com
    subdomain_parts = normalized_parts[:-2] if len(normalized_parts) > 2 else []
    subdomain = ".".join(subdomain_parts)
    main_domain = ".".join(normalized_parts[-2:]) if len(normalized_parts) >= 2 else normalized_domain
    
    return {
        # Basic URL features (use normalized domain for consistency)
        "url_length": len(url),
        "domain_length": len(normalized_domain),  # Use normalized to avoid www/non-www clustering split
        "path_length": len(path),
        
        # Domain structure features (use normalized domain)
        "num_dots": normalized_domain.count("."),
        "num_dashes": normalized_domain.count("-"),
        "num_underscores": normalized_domain.count("_"),
        "num_digits": sum(c.isdigit() for c in normalized_domain),
        
        # Subdomain features
        "has_subdomain": 1 if subdomain else 0,
        "subdomain_length": len(subdomain),
        "subdomain_digits": sum(c.isdigit() for c in subdomain),
        
        # Path features
        "num_slashes": path.count("/"),
        "has_query": 1 if "?" in url else 0,
        "has_fragment": 1 if "#" in url else 0,
        "query_length": len(url.split("?")[-1]) if "?" in url else 0,
        
        # Protocol/subdomain features (informational only, reduced weight)
        # Note: These should NOT heavily influence clustering as legitimate sites
        # often have both HTTP/HTTPS and www/non-www variants
        "is_https": 1 if protocol == "https" else 0,
        "has_www": 1 if domain.startswith("www.") else 0,
        
        # Suspicious patterns
        "has_at": 1 if "@" in url else 0,
        "has_ip": 1 if any(part.isdigit() and len(part) <= 3 for part in normalized_parts) else 0,
        "suspicious_chars": sum(1 for c in normalized_domain if c in "!@#$%^&*()+=[]{}|;:,.<>?"),

        # TLD features (use normalized parts)
        "tld_length": len(normalized_parts[-1]) if normalized_parts else 0,
        "is_common_tld": 1 if normalized_parts[-1] in ["com", "org", "net", "edu", "gov"] else 0,
        
        # Enhanced features for ensemble (use normalized domain for consistency)
        "url_entropy": calculate_entropy(url),
        "domain_entropy": calculate_entropy(normalized_domain),
        "suspicious_kw_count": sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url.lower()),
        "digit_ratio": sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0,
        "special_char_ratio": (normalized_domain.count("-") + normalized_domain.count("_") + url.count("@") + url.count("%")) / len(url) if len(url) > 0 else 0,
        "has_port": 1 if ":" in normalized_domain and not normalized_domain.startswith("[") else 0,
        "num_ampersand": url.count("&"),
        "num_equals": url.count("="),
        "num_percent": url.count("%"),
    }

def calculate_feature_risk(features_dict):
    """Calculate rule-based risk score from features"""
    risk_score = (
        features_dict.get('has_ip', 0) * 0.3 +
        features_dict.get('suspicious_kw_count', 0) * 0.15 +
        features_dict.get('has_at', 0) * 0.2 +
        (1 - features_dict.get('is_common_tld', 1)) * 0.15 +
        (1 if features_dict.get('digit_ratio', 0) > 0.3 else 0) * 0.1 +
        features_dict.get('has_port', 0) * 0.1
    )
    return min(risk_score, 1.0)  # Cap at 1.0

def load_or_train_model():
    """Load existing model or train new one"""
    global scaler, linkage_matrix, cluster_map, centroids_matrix, centroid_ids, adaptive_thresholds, iso_forest, iso_score_normalizer, X_scaled, urls, labels, clusters
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    model_files = {
        'scaler': os.path.join(models_dir, 'scaler.pkl'),
        'linkage': os.path.join(models_dir, 'linkage_matrix.pkl'),
        'cluster_map': os.path.join(models_dir, 'cluster_map.pkl'),
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
            cluster_map = joblib.load(model_files['cluster_map'])
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
    
    # Train new model
    print("Training new hierarchical clustering model...")
    return train_hierarchical_model()

def train_hierarchical_model():
    """Train hierarchical clustering model with ensemble approach"""
    global scaler, linkage_matrix, cluster_map, centroids_matrix, centroid_ids, adaptive_thresholds, iso_forest, iso_score_normalizer, X_scaled, urls, labels, clusters
    
    try:
        # Load dataset
        df = pd.read_csv("URL dataset.csv")
        
        # Standardize column names to expected names
        df = df.rename(columns={"URL": "url", "Url": "url", "type": "label", "Type": "label"})
        
        # If no label column exists, create placeholder
        if "label" not in df.columns:
            df["label"] = "unknown"
        
        # Clean data
        df = df.drop_duplicates(subset="url").dropna(subset=["url"])
        print("Dataset size:", df.shape)
        
        # Stratified sampling to handle class imbalance
        print("\nBalancing dataset for training...")
        
        # Separate by actual class
        legitimate_rows = df[df["label"] == "legitimate"]
        phishing_rows = df[df["label"] == "phishing"]
        
        print(f"Available - Legitimate: {len(legitimate_rows)}, Phishing: {len(phishing_rows)}")
        
        # Use balanced ratio: 3:1 (legitimate:phishing) to maintain real-world distribution
        # but prevent extreme imbalance and memory issues
        n_phishing_sample = min(15000, len(phishing_rows))  # Use up to 15k phishing (reduced for memory)
        n_legitimate_sample = min(n_phishing_sample * 3, len(legitimate_rows), 45000)  # 3:1 ratio, max 45k
        
        if len(phishing_rows) > 0 and len(legitimate_rows) > 0:
            df_sample = pd.concat([
                legitimate_rows.sample(n=n_legitimate_sample, random_state=42),
                phishing_rows.sample(n=n_phishing_sample, random_state=42)
            ]).reset_index(drop=True)
        else:
            # Fallback to simple sampling if classes not available
            df_sample = df.sample(n=min(25000, len(df)), random_state=42).reset_index(drop=True)
        
        print(f"Training sample: {len(df_sample)} URLs")
        print(f"  - Legitimate: {(df_sample['label'] == 'legitimate').sum()}")
        print(f"  - Phishing: {(df_sample['label'] == 'phishing').sum()}")
        
        # Extract and scale features
        features_sample = df_sample["url"].apply(extract_features)
        X_sample = pd.DataFrame(features_sample.tolist())
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_sample)
        
        # Hierarchical clustering with Ward linkage
        # Use smaller subset for clustering to avoid memory issues
        print("Performing hierarchical clustering...")
        clustering_sample_size = min(10000, len(X_scaled))  # Max 10k for memory efficiency
        print(f"  Using {clustering_sample_size} samples for clustering (from {len(X_scaled)} total)")
        
        # Random sample for clustering
        if len(X_scaled) > clustering_sample_size:
            clustering_indices = np.random.choice(len(X_scaled), size=clustering_sample_size, replace=False)
            X_clustering = X_scaled[clustering_indices]
        else:
            clustering_indices = np.arange(len(X_scaled))
            X_clustering = X_scaled
        
        # Build hierarchical clustering on subset
        linkage_matrix = sch.linkage(X_clustering, method="ward")
        
        # Get clusters for the clustering subset
        clusters_subset = fcluster(linkage_matrix, t=10, criterion="distance")
        
        # Calculate centroids for each cluster
        unique_clusters = np.unique(clusters_subset)
        cluster_centroids = {}
        for c in unique_clusters:
            mask = clusters_subset == c
            cluster_centroids[c] = X_clustering[mask].mean(axis=0)
        
        # Assign all samples to nearest cluster centroid
        print(f"  Assigning all {len(X_scaled)} samples to clusters...")
        centroid_matrix = np.array([cluster_centroids[c] for c in unique_clusters])
        distances = cdist(X_scaled, centroid_matrix, metric='euclidean')
        closest_clusters = np.argmin(distances, axis=1)
        clusters = unique_clusters[closest_clusters]
        
        df_sample["cluster"] = clusters
        
        # Map clusters to pattern groups with purity thresholds
        cluster_map = {}
        cluster_purity = {}  # Track how "pure" each cluster is

        for c in df_sample["cluster"].unique():
            labels_in_cluster = df_sample.loc[df_sample["cluster"] == c, "label"]
            # Use only recognized labels for mapping
            valid_labels = labels_in_cluster[labels_in_cluster.isin(["phishing", "legitimate"])]

            if len(valid_labels) > 0:
                # Calculate purity (percentage of majority class)
                label_counts = valid_labels.value_counts()
                total_valid = len(valid_labels)
                majority_count = label_counts.iloc[0]
                purity = majority_count / total_valid
                
                cluster_purity[c] = purity
                
                # Only assign labels to clusters with high purity (80%+)
                if purity >= 0.8:  # 80% purity threshold
                    majority_label = label_counts.index[0]
                    cluster_map[c] = "suspicious_pattern" if majority_label == "phishing" else "normal_pattern"
                else:
                    # Low purity clusters remain unmapped (will show as "uncertain")
                    cluster_map[c] = "mixed_pattern"
            else:
                cluster_map[c] = "unknown_pattern"
        
        if cluster_map:
            df_sample["prediction"] = df_sample["cluster"].map(cluster_map)
            labeled = df_sample[df_sample["label"].isin(["phishing", "legitimate"])]
            if not labeled.empty:
                from sklearn.metrics import classification_report
                print("Classification Report:")
                print(classification_report(labeled["label"], df_sample["prediction"]))
                
            # Print cluster purity information
            print("\nCluster Purity Analysis:")
            for cluster_id, purity in cluster_purity.items():
                pattern_type = cluster_map.get(cluster_id, "unknown")
                print(f"Cluster {cluster_id}: {pattern_type} (purity: {purity:.2f})")
        else:
            print("Warning: No labeled clusters were mapped. URL checker will report unavailable classification.")
        
        # Calculate adaptive thresholds per cluster
        print("\nCalculating adaptive thresholds...")
        adaptive_thresholds = {}
        cluster_stats = {}
        
        for c in df_sample["cluster"].unique():
            cluster_data = df_sample[df_sample["cluster"] == c]
            cluster_indices = cluster_data.index.values
            cluster_distances = []
            
            # Calculate distances to cluster centroid
            cluster_points = X_scaled[cluster_indices]
            if len(cluster_points) > 0:
                centroid = cluster_points.mean(axis=0)
                for point in cluster_points:
                    dist = np.linalg.norm(point - centroid)
                    cluster_distances.append(dist)
                
                # Calculate phishing rate for this cluster
                cluster_labels = df_sample.loc[df_sample["cluster"] == c, "label"]
                valid_labels = cluster_labels[cluster_labels.isin(["phishing", "legitimate"])]
                phishing_rate = 0
                if len(valid_labels) > 0:
                    phishing_rate = (valid_labels == "phishing").sum() / len(valid_labels)
                
                # Store cluster statistics
                cluster_stats[c] = {
                    'mean_distance': np.mean(cluster_distances),
                    'std_distance': np.std(cluster_distances),
                    'phishing_rate': phishing_rate
                }
                
                # Adaptive thresholding: stricter for high-risk clusters
                if phishing_rate > 0.3:  # High-risk cluster (>30% phishing)
                    threshold = np.percentile(cluster_distances, 85)  # 85th percentile - more sensitive
                else:
                    threshold = np.percentile(cluster_distances, 95)  # 95th percentile - less sensitive
                
                adaptive_thresholds[c] = threshold
                
                print(f"Cluster {c}: phishing_rate={phishing_rate:.2f}, threshold={threshold:.3f}")
        
        # Train Isolation Forest for anomaly detection
        print("\nTraining Isolation Forest...")
        phishing_rate = (df_sample["label"] == "phishing").sum() / len(df_sample) if len(df_sample) > 0 else 0.23
        iso_forest = IsolationForest(
            n_estimators=200,  # More trees for stability
            contamination=min(phishing_rate, 0.25),  # Match actual phishing rate, cap at 25%
            max_samples=min(10000, len(df_sample)),
            random_state=42,
            n_jobs=-1
        )
        iso_forest.fit(X_scaled)
        
        # Calculate normalization factor for scores
        iso_scores_train = -iso_forest.score_samples(X_scaled)
        iso_score_normalizer = np.percentile(iso_scores_train, 99)  # 99th percentile
        print(f"Isolation Forest trained - contamination: {phishing_rate:.2%}, normalizer: {iso_score_normalizer:.3f}")
        
        # Precompute centroids per mapped cluster (following preprocess.py)
        centroid_ids = []
        centroids_matrix = []
        
        for c, mapped_label in cluster_map.items():
            mask = (df_sample["cluster"].values == c)
            pts = X_scaled[mask]
            if pts.shape[0] > 0:
                centroids_matrix.append(pts.mean(axis=0))
                centroid_ids.append(c)
        
        if len(centroids_matrix) > 0:
            centroids_matrix = np.vstack(centroids_matrix)
        else:
            centroids_matrix = None  # signal no usable centroids
        
        # Store data
        urls = df_sample['url'].tolist()
        labels = df_sample['label'].tolist()
        
        # Save model
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)

        joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
        joblib.dump(linkage_matrix, os.path.join(models_dir, 'linkage_matrix.pkl'))
        joblib.dump(cluster_map, os.path.join(models_dir, 'cluster_map.pkl'))
        joblib.dump(centroids_matrix, os.path.join(models_dir, 'centroids.pkl'))
        joblib.dump(centroid_ids, os.path.join(models_dir, 'centroid_ids.pkl'))
        joblib.dump(adaptive_thresholds, os.path.join(models_dir, 'adaptive_thresholds.pkl'))
        joblib.dump(iso_forest, os.path.join(models_dir, 'iso_forest.pkl'))
        joblib.dump(iso_score_normalizer, os.path.join(models_dir, 'iso_normalizer.pkl'))
        joblib.dump({
            'X_scaled': X_scaled,
            'urls': urls,
            'labels': labels,
            'clusters': clusters
        }, os.path.join(models_dir, 'processed_data.pkl'))
        
        print("Model trained and saved successfully!")
        return True
        
    except Exception as e:
        print(f"Error training model: {e}")
        return False

def create_dendrogram_figure(truncate_mode='lastp', p=30, color_threshold=None, 
                           leaf_rotation=90, figsize=(12, 8), dpi=150):
    """Create dendrogram figure"""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
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
    
    plt.title('Hierarchical Clustering Dendrogram', fontsize=18, fontweight='bold')  # Increased from 16
    plt.xlabel('Sample Index or (cluster size)', fontsize=14)  # Increased from 12
    plt.ylabel('Distance', fontsize=14)  # Increased from 12
    plt.axhline(y=10, color="r", linestyle="--", label='Distance Threshold (t=10)')
    plt.grid(True, alpha=0.3)
    
    # Add color legend
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
    legend_elements = []
    for i, color in enumerate(colors[:len(set(dendro['color_list']))]):
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=f'Cluster {i+1}'))
    
    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper right', fontsize=12)  # Increased font size
    
    return fig

def find_url_position_in_dendrogram(url):
    """Find URL position in the hierarchical clustering tree - following preprocess.py approach"""
    try:
        # Check if model components are available
        if not cluster_map:
            return {
                'url': url,
                'prediction': 'unavailable',
                'confidence': 0,
                'message': 'No labeled clusters available — cannot classify this URL.'
            }
        if centroids_matrix is None or len(centroid_ids) == 0:
            return {
                'url': url,
                'prediction': 'unavailable',
                'confidence': 0,
                'message': 'No valid centroids found — cannot classify this URL.'
            }
        
        # Extract features for the input URL
        feats = extract_features(url)
        X_new = pd.DataFrame([feats])
        X_new_scaled = scaler.transform(X_new)
        
        # Find nearest centroid using pairwise_distances_argmin_min (like preprocess.py)
        from sklearn.metrics import pairwise_distances_argmin_min
        closest, distances = pairwise_distances_argmin_min(X_new_scaled, centroids_matrix)
        cluster_id = centroid_ids[closest[0]]
        distance_to_centroid = distances[0]
        
        # Get pattern group from cluster map
        if cluster_id in cluster_map:
            base_prediction = cluster_map[cluster_id]
        else:
            base_prediction = 'unknown'
        
        # ENSEMBLE APPROACH: Combine three detection methods
        
        # Component 1: Clustering score (adaptive threshold)
        if adaptive_thresholds and cluster_id in adaptive_thresholds:
            adaptive_threshold = adaptive_thresholds[cluster_id]
            cluster_score = distance_to_centroid / adaptive_threshold
        else:
            # Fallback to fixed threshold if adaptive not available
            adaptive_threshold = 5.0
            cluster_score = distance_to_centroid / adaptive_threshold
        
        # Component 2: Isolation Forest anomaly score
        if iso_forest and iso_score_normalizer:
            iso_raw_score = -iso_forest.score_samples(X_new_scaled)[0]
            iso_score_norm = min(iso_raw_score / iso_score_normalizer, 2.0)
        else:
            iso_score_norm = 0
        
        # Component 3: Feature-based risk score
        feature_risk = calculate_feature_risk(feats)
        
        # Weighted Ensemble (weights optimized for precision-recall balance)
        combined_score = (
            0.45 * cluster_score +      # 45% clustering
            0.35 * iso_score_norm +     # 35% anomaly detection
            0.20 * feature_risk         # 20% rule-based
        )
        
        # Calculate final confidence (inverse of combined score)
        # Lower combined_score = higher confidence
        confidence = max(0, min(1, 1 - (combined_score - 0.5)))
        
        # Determine cluster assignment with purity-based logic
        if confidence < 0.1:  # Very low confidence
            prediction = 'uncertain'
            confidence = 1 - confidence  # Show uncertainty level
        elif base_prediction == 'suspicious_pattern':
            prediction = 'suspicious_pattern'
        elif base_prediction == 'normal_pattern':
            prediction = 'normal_pattern'
        elif base_prediction == 'mixed_pattern':
            prediction = 'mixed_pattern'  # New category for impure clusters
        else:
            prediction = 'uncertain'
            confidence = 0.5
        
        # Find nearest neighbors for additional context
        distances_to_all = np.linalg.norm(X_scaled - X_new_scaled, axis=1)
        nearest_indices = np.argsort(distances_to_all)[:NEAREST_NEIGHBORS_COUNT]  # Use configurable count
        
        nearest_neighbors = []
        for i in nearest_indices:
            neighbor_cluster = clusters[i]
            neighbor_label = labels[i] if i < len(labels) else 'unknown'
            neighbor_url = urls[i] if i < len(urls) else 'unknown'
            
            nearest_neighbors.append({
                'url': neighbor_url,
                'label': neighbor_label,
                'distance': float(distances_to_all[i]),
                'cluster': int(neighbor_cluster)
            })
        
        # Calculate cluster votes for additional context
        cluster_votes = {}
        neighbor_confidence = 0

        for neighbor in nearest_neighbors:
            cluster_id = neighbor['cluster']
            if cluster_id in cluster_map:
                pattern_group = cluster_map[cluster_id]
                cluster_votes[pattern_group] = cluster_votes.get(pattern_group, 0) + 1
        
        # Calculate neighbor-based confidence (how consistent are the neighbors?)
        if len(nearest_neighbors) > 0:
            neighbor_distances = [n['distance'] for n in nearest_neighbors]
            avg_neighbor_distance = np.mean(neighbor_distances)
            # Lower average distance = higher confidence
            neighbor_confidence = max(0, 1 - (avg_neighbor_distance / 3.0))
        
        return {
            'url': url,
            'prediction': prediction,
            'confidence': confidence,
            'combined_score': float(combined_score),
            'cluster_score': float(cluster_score),
            'iso_score': float(iso_score_norm) if iso_forest else None,
            'feature_risk': float(feature_risk),
            'neighbor_confidence': neighbor_confidence,
            'distance_to_centroid': float(distance_to_centroid),
            'adaptive_threshold': float(adaptive_thresholds.get(cluster_id, 0)) if adaptive_thresholds else None,
            'nearest_neighbors': nearest_neighbors,
            'cluster_votes': cluster_votes,
            'cluster_id': int(cluster_id)
        }
        
    except Exception as e:
        return {
            'url': url,
            'error': str(e),
            'prediction': 'error',
            'confidence': 0
        }

def create_url_cluster_analysis(url, figsize=(12, 8), dpi=150):
    """Create a comprehensive analysis showing where URL fits in the clustering tree"""
    try:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
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
        
        ax.set_title(f'URL Pattern Analysis: {url[:50]}...\nPattern Group: {prediction} | Similarity: {confidence:.2f} | Cluster: {cluster_id}', fontsize=16, fontweight='bold')  # Increased font size
        ax.set_xlabel('Cluster ID', fontsize=14)  # Increased font size
        ax.set_ylabel('Number of Samples', fontsize=14)  # Increased font size
        ax.set_xticks(range(len(clusters_sorted)))
        ax.set_xticklabels(clusters_sorted, fontsize=12)  # Increased font size for tick labels
        ax.tick_params(axis='y', labelsize=12)  # Increased font size for y-axis tick labels
        ax.grid(True, alpha=0.3)
        
        # Add text annotation
        ax.text(0.02, 0.98, f'URL: {url}\nPattern Group: {prediction}\nSimilarity: {confidence:.2f}\nCluster: {cluster_id}\nDistance: {distance:.3f}', 
                transform=ax.transAxes, fontsize=14,  # Increased from 10
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                verticalalignment='top')
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='Suspicious Pattern Groups'),
            plt.Rectangle((0,0),1,1, facecolor='green', alpha=0.7, label='Normal Pattern Groups'),
            plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.7, label='Unknown Pattern Groups'),
            plt.Rectangle((0,0),1,1, facecolor='blue', alpha=1.0, label='URL Pattern Group')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)  # Increased font size
        
        return fig
        
    except Exception as e:
        print(f"Error creating URL cluster analysis: {e}")
        # Fallback to simple plot
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, f'URL Analysis Error\n{str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('URL Cluster Analysis')
        return fig

def figure_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)  # Increased DPI from 100
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    return image_base64

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_url():
    """Analyze URL and return results"""
    global clusters, labels
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Analyze URL
        url_info = find_url_position_in_dendrogram(url)
        
        # Generate matplotlib charts and convert to base64
        main_dendro_fig = create_dendrogram_figure()
        url_analysis_fig = create_url_cluster_analysis(url)
        
        # Convert to base64 images
        main_dendro_b64 = figure_to_base64(main_dendro_fig) if main_dendro_fig else None
        url_analysis_b64 = figure_to_base64(url_analysis_fig) if url_analysis_fig else None
        
        # Close figures to free memory
        if main_dendro_fig:
            plt.close(main_dendro_fig)
        if url_analysis_fig:
            plt.close(url_analysis_fig)
        
        return jsonify({
            'success': True,
            'url_info': url_info,
            'main_dendrogram': f'data:image/png;base64,{main_dendro_b64}' if main_dendro_b64 else None,
            'url_analysis': f'data:image/png;base64,{url_analysis_b64}' if url_analysis_b64 else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize model on startup
    if load_or_train_model():
        print("Flask app starting...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize model. Please check your dataset.")
