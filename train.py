#!/usr/bin/env python3
"""
Training script for the URL clustering model.
Run this script to train the hierarchical clustering model on the URL dataset.
"""

import os
import pandas as pd
from ml_model import train_hierarchical_model, load_or_train_model

def main():
    """Main training function"""
    print("URL Clustering Model Training")
    print("=" * 50)

    # Check if dataset exists
    dataset_path = "URL dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file '{dataset_path}' not found!")
        print("Please ensure the URL dataset is in the project root directory.")
        return False

    # Check dataset size
    try:
        df = pd.read_csv(dataset_path)
        print(f"Dataset loaded: {len(df)} URLs")
        print(f"Columns: {list(df.columns)}")

        # Check for required columns
        required_cols = ['url']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            # Try alternative column names
            alt_names = {'URL': 'url', 'Url': 'url'}
            for alt, standard in alt_names.items():
                if alt in df.columns and standard not in df.columns:
                    df = df.rename(columns={alt: standard})
                    print(f"Renamed column '{alt}' to '{standard}'")

        if 'url' not in df.columns:
            print("Error: No 'url' column found in dataset!")
            print("Expected column names: 'url', 'URL', or 'Url'")
            return False

        # Clean data
        df = df.drop_duplicates(subset="url").dropna(subset=["url"])
        print(f"After cleaning: {len(df)} URLs")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

    # Train the model
    print("\nStarting model training...")
    success = train_hierarchical_model()

    if success:
        print("\nTraining completed successfully!")
        print("Model files saved to 'models/' directory:")
        print("- scaler.pkl: Feature scaler")
        print("- linkage_matrix.pkl: Hierarchical clustering structure")
        print("- cluster_map.pkl: Cluster pattern mappings")
        print("- centroids.pkl: Cluster centroids")
        print("- adaptive_thresholds.pkl: Adaptive distance thresholds")
        print("- iso_forest.pkl: Isolation Forest model")
        print("- processed_data.pkl: Training data and labels")
        return True
    else:
        print("\nTraining failed!")
        return False

if __name__ == "__main__":
    main()
