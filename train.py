import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist
import joblib
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from features import extract_features

# Train hierarchical clustering model and save artifacts
def train_model():
    try:
        # Load and clean dataset
        df = pd.read_csv("URL dataset.csv")
        df = df.rename(columns={"URL": "url", "Url": "url", "type": "label", "Type": "label"})
        if "label" not in df.columns: df["label"] = "unknown"
        df = df.drop_duplicates(subset="url").dropna(subset=["url"])
        print(f"Dataset size: {df.shape}")

        # Create balanced training sample
        legit_rows = df[df["label"] == "legitimate"]
        phish_rows = df[df["label"] == "phishing"]
        n_phish = min(15000, len(phish_rows))
        n_legit = min(n_phish * 3, len(legit_rows))
        df_sample = pd.concat([
            legit_rows.sample(n=n_legit, random_state=42),
            phish_rows.sample(n=n_phish, random_state=42)
        ]).reset_index(drop=True)
        print(f"Training with {len(df_sample)} samples ({n_legit} legit, {n_phish} phish)")

        # Extract features and scale for clustering
        features_list = df_sample["url"].apply(extract_features).tolist()
        X_sample = pd.DataFrame(features_list)
        feature_names = X_sample.columns.tolist()
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_sample)

        # Use subsample for efficient clustering
        clustering_sample_size = min(10000, len(X_scaled))
        print(f"Building dendrogram with {clustering_sample_size} stratified samples...")

        # Create stratified sample for clustering
        clustering_indices, _ = train_test_split(
            np.arange(len(X_scaled)),
            train_size=clustering_sample_size,
            stratify=df_sample['label'],
            random_state=42
        )
        X_clustering = X_scaled[clustering_indices]

        linkage_matrix = sch.linkage(X_clustering, method="ward")
        clusters_subset = fcluster(linkage_matrix, t=10, criterion="distance")

        # Assign all samples to nearest centroids
        unique_clusters, counts = np.unique(clusters_subset, return_counts=True)
        print(f"Found {len(unique_clusters)} initial clusters.")
        cluster_centroids = {c: X_clustering[clusters_subset == c].mean(axis=0) for c in unique_clusters}

        centroid_matrix = np.array([cluster_centroids[c] for c in unique_clusters])
        distances = cdist(X_scaled, centroid_matrix, metric='euclidean')
        clusters = unique_clusters[np.argmin(distances, axis=1)]
        df_sample["cluster"] = clusters

        # Calculate cluster purity statistics
        print("\nCalculating cluster purity statistics...")
        cluster_stats = {}
        for c in np.unique(clusters):
            cluster_data = df_sample[df_sample["cluster"] == c]
            labels_in_cluster = cluster_data["label"]

            total = len(labels_in_cluster)
            phish_count = (labels_in_cluster == 'phishing').sum()
            legit_count = (labels_in_cluster == 'legitimate').sum()

            if total > 0:
                purity = max(phish_count, legit_count) / total
                majority_class = 'phishing' if phish_count >= legit_count else 'legitimate'
            else:
                purity = 0
                majority_class = 'unknown'

            cluster_stats[c] = {
                'total_count': int(total),
                'phishing_count': int(phish_count),
                'legitimate_count': int(legit_count),
                'purity': float(purity),
                'majority_class': majority_class
            }
        print("Cluster statistics calculated.")

        # Calculate adaptive thresholds per cluster
        adaptive_thresholds = {}
        for c, stats in cluster_stats.items():
            cluster_indices = df_sample[df_sample["cluster"] == c].index
            cluster_points = X_scaled[cluster_indices]
            if len(cluster_points) > 0:
                centroid = cluster_points.mean(axis=0)
                distances_to_centroid = [np.linalg.norm(p - centroid) for p in cluster_points]
                phishing_rate = stats['phishing_count'] / stats['total_count'] if stats['total_count'] > 0 else 0
                percentile = 85 if phishing_rate > 0.3 else 95
                adaptive_thresholds[c] = np.percentile(distances_to_centroid, percentile)

        phishing_rate_overall = (df_sample["label"] == "phishing").sum() / len(df_sample)
        iso_forest = IsolationForest(contamination=min(phishing_rate_overall, 0.25), random_state=42, n_jobs=-1)
        iso_forest.fit(X_scaled)
        iso_scores_train = -iso_forest.score_samples(X_scaled)
        iso_score_normalizer = np.percentile(iso_scores_train, 99)

        # Generate t-SNE model for 2D visualization
        print("\nGenerating t-SNE model for 2D visualization...")
        tsne_sample_size = min(5000, len(X_scaled))
        tsne_indices, _ = train_test_split(
            np.arange(len(X_scaled)),
            train_size=tsne_sample_size,
            stratify=df_sample['label'],
            random_state=42
        )
        X_tsne_sample = X_scaled[tsne_indices]

        # Create 2D embedding
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        X_tsne_2d = tsne.fit_transform(X_tsne_sample)

        # Train mapper to project new URLs into 2D space
        print("Training t-SNE mapping model (KNeighborsRegressor)...")
        tsne_mapper = KNeighborsRegressor(n_neighbors=10)
        tsne_mapper.fit(X_tsne_sample, X_tsne_2d)

        # Prepare data for background plotting
        tsne_data_for_plot = {
            'coords': X_tsne_2d.tolist(),
            'labels': df_sample.iloc[tsne_indices]['label'].tolist()
        }

        # Save all model artifacts
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
        joblib.dump(linkage_matrix, os.path.join(models_dir, 'linkage_matrix.pkl'))
        joblib.dump(cluster_stats, os.path.join(models_dir, 'cluster_stats.pkl'))
        joblib.dump(adaptive_thresholds, os.path.join(models_dir, 'adaptive_thresholds.pkl'))
        joblib.dump(iso_forest, os.path.join(models_dir, 'iso_forest.pkl'))
        joblib.dump(iso_score_normalizer, os.path.join(models_dir, 'iso_normalizer.pkl'))
        joblib.dump(feature_names, os.path.join(models_dir, 'feature_names.pkl'))
        joblib.dump(tsne_mapper, os.path.join(models_dir, 'tsne_mapper.pkl'))
        joblib.dump(tsne_data_for_plot, os.path.join(models_dir, 'tsne_data.pkl'))

        # Save centroids and processed data
        centroid_ids = list(cluster_centroids.keys())
        centroids_matrix = np.array(list(cluster_centroids.values()))
        joblib.dump(centroids_matrix, os.path.join(models_dir, 'centroids.pkl'))
        joblib.dump(centroid_ids, os.path.join(models_dir, 'centroid_ids.pkl'))
        joblib.dump({
            'X_scaled': X_scaled,
            'urls': df_sample['url'].tolist(),
            'labels': df_sample['label'].tolist(),
            'clusters': clusters
        }, os.path.join(models_dir, 'processed_data.pkl'))

        print("\nModel training complete and artifacts saved.")
        return True

    except Exception as e:
        print(f"Error during model training: {e}")
        return False

if __name__ == "__main__":
    train_model()
