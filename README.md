# URL Analysis - Phishing Detection System

A machine learning-based web application for detecting phishing URLs using advanced clustering and anomaly detection techniques.

## Features

- **Real-time URL Analysis**: Instantly analyze URLs for phishing indicators
- **Machine Learning Models**: Uses multiple ML algorithms including:
  - Isolation Forest for anomaly detection
  - Hierarchical clustering for pattern recognition
  - Adaptive thresholds for dynamic detection
- **Interactive Web Interface**: User-friendly Flask-based web application
- **Visual Analytics**: Beautiful 3D coverflow design for results presentation

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

3. Enter a URL to analyze and get instant phishing detection results

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
- **Machine Learning**: scikit-learn, pandas, numpy
- **Frontend**: HTML, CSS, JavaScript
- **Visualization**: Plotly, Matplotlib

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or support, please open an issue on GitHub.

