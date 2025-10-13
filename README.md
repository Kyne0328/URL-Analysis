# URL Analysis - Hierarchical Clustering Explorer

A machine learning-based web application for exploring URL patterns using advanced hierarchical clustering and unsupervised learning techniques.

## Features

- **URL Pattern Analysis**: Analyze URL structural patterns and similarities
- **Hierarchical Clustering**: Advanced clustering algorithms including:
  - Ward linkage hierarchical clustering for pattern recognition
  - Isolation Forest for anomaly detection
  - Adaptive thresholds for dynamic cluster analysis
- **Interactive Web Interface**: User-friendly Flask-based web application
- **Visual Analytics**: Interactive dendrograms and cluster analysis charts

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Kyne0328/URL-Analysis.git
cd URL-Analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Enter a URL to analyze and explore its clustering patterns and similarities

## How Hierarchical Clustering Works

This project implements **Ward Linkage Hierarchical Clustering**, a bottom-up approach that:

1. **Starts with individual URLs** as separate clusters
2. **Iteratively merges** the most similar clusters based on Ward's criterion
3. **Builds a dendrogram** showing the complete clustering hierarchy
4. **Uses distance thresholds** to determine final cluster assignments

### Key Features:
- **29 URL Features**: Analyzes structural patterns like domain length, entropy, special characters
- **Ward Linkage**: Minimizes within-cluster variance for compact, well-separated clusters
- **Adaptive Thresholds**: Dynamic distance thresholds based on cluster purity
- **Ensemble Approach**: Combines clustering with anomaly detection for robust analysis

### Clustering Process:
1. **Feature Extraction**: Converts URLs into 29-dimensional feature vectors
2. **Normalization**: RobustScaler handles outliers and different scales
3. **Hierarchical Clustering**: Ward linkage builds the dendrogram
4. **Cluster Assignment**: Distance-based assignment to final clusters
5. **Pattern Analysis**: Identifies normal, suspicious, and mixed patterns

### Technical Details:
- **Linkage Method**: Ward (minimizes within-cluster variance)
- **Distance Metric**: Euclidean distance in 29-dimensional feature space
- **Clustering Threshold**: t=10 (distance threshold for cluster formation)
- **Sample Size**: 10,000 URLs for clustering (memory-efficient)
- **Feature Scaling**: RobustScaler (handles outliers)
- **Cluster Purity**: 80% threshold for pattern classification

## Project Structure

```
URL-Analysis/
│
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── URL dataset.csv        # Training dataset
│
├── models/                # Pre-trained ML models
│   ├── adaptive_thresholds.pkl
│   ├── centroid_ids.pkl
│   ├── centroids.pkl
│   ├── cluster_map.pkl
│   ├── iso_forest.pkl
│   ├── iso_normalizer.pkl
│   ├── linkage_matrix.pkl
│   ├── processed_data.pkl
│   └── scaler.pkl
│
├── templates/             # HTML templates
│   └── index.html
│
└── static/               # Static assets
    ├── css/
    │   └── templatemo-3d-coverflow.css
    └── js/
        └── templatemo-3d-coverflow-scripts.js
```

## Technologies Used

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn, pandas, numpy, scipy
- **Clustering**: Hierarchical clustering with Ward linkage
- **Frontend**: HTML, CSS, JavaScript
- **Visualization**: Plotly, Matplotlib
- **Data Processing**: Feature engineering with 29 URL attributes

## Educational Value

This project serves as an excellent learning resource for:

- **Understanding Hierarchical Clustering**: See how Ward linkage builds clusters step-by-step
- **Feature Engineering**: Learn how to extract meaningful features from text data
- **Dendrogram Interpretation**: Visualize cluster hierarchies and understand distance metrics
- **Ensemble Methods**: Combine multiple ML approaches for robust analysis
- **Real-world Application**: Apply clustering to cybersecurity and web analysis

### Educational Features:
- **Educational Section**: Comprehensive explanations of clustering concepts
- **Tooltips**: Hover over technical terms for detailed explanations
- **Visual Learning**: Interactive dendrograms and cluster analysis charts
- **Clustering Comparison**: Side-by-side comparison of different clustering methods
- **Technical Details**: In-depth explanations of Ward linkage and feature engineering

## Use Cases

- **Cybersecurity Research**: Analyze URL patterns for threat intelligence
- **Web Analytics**: Understand website structure patterns
- **Machine Learning Education**: Learn clustering algorithms with interactive examples
- **Data Science Projects**: Template for hierarchical clustering implementations
- **Pattern Recognition**: Discover similarities in structured text data

## Important Note

⚠️ **This is a clustering analysis tool, not a security detection system.** It groups URLs based on structural similarity patterns for educational and research purposes. Do not rely on it for security decisions.

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or support, please open an issue on GitHub.

