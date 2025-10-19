from flask import Flask, render_template, request, jsonify, send_file
from ml_model import load_or_train_model, find_url_position_in_dendrogram, create_dendrogram_figure, get_purity_plot_data, get_feature_comparison_data, get_cluster_distribution_data, get_tsne_visualization_data, figure_to_base64
import matplotlib.pyplot as plt

app = Flask(__name__)

# Main page route
@app.route('/')
def index():
    return render_template('index.html')

# Analyze URL and return results
@app.route('/analyze', methods=['POST'])
def analyze_url():
    try:
        data = request.get_json()
        url = data.get('url', '').strip()

        if not url:
            return jsonify({'error': 'URL is required'}), 400

        # Ensure URL has protocol
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # Perform URL analysis
        url_info = find_url_position_in_dendrogram(url)

        # Generate visualization data
        main_dendro_fig = create_dendrogram_figure()
        purity_plot_data = get_purity_plot_data()
        cluster_distribution_data = get_cluster_distribution_data()

        # Prepare radar chart data
        radar_chart_data = None
        if url_info and 'scaled_features' in url_info:
            radar_chart_data = get_feature_comparison_data(url_info['scaled_features'][0], url_info['cluster_id'])

        # Prepare t-SNE visualization data
        tsne_plot_data = get_tsne_visualization_data(url_info['scaled_features'][0], url_info['nearest_neighbors'])

        # Convert plot to base64
        main_dendro_b64 = figure_to_base64(main_dendro_fig) if main_dendro_fig else None

        # Clean up memory
        if main_dendro_fig:
            plt.close(main_dendro_fig)

        return jsonify({
            'success': True,
            'url_info': url_info,
            'main_dendrogram': f'data:image/png;base64,{main_dendro_b64}' if main_dendro_b64 else None,
            'purity_plot_data': purity_plot_data,
            'cluster_distribution_data': cluster_distribution_data,
            'radar_chart_data': radar_chart_data,
            'tsne_plot_data': tsne_plot_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
