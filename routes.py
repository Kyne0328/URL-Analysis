from flask import Flask, render_template, request, jsonify, send_file
from ml_model import load_or_train_model, find_url_position_in_dendrogram, create_dendrogram_figure, create_url_cluster_analysis, figure_to_base64
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_url():
    """Analyze URL and return results"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()

        if not url:
            return jsonify({'error': 'URL is required'}), 400

        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # --- MODIFIED SECTION: Analyze ONCE and pass results ---
        # 1. Run the analysis and store the complete result object
        url_info = find_url_position_in_dendrogram(url)

        # 2. Generate the plots
        main_dendro_fig = create_dendrogram_figure()
        # 3. Pass the *result object* to the chart function, not the raw URL
        url_analysis_fig = create_url_cluster_analysis(url_info)
        # --- END OF MODIFIED SECTION ---

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
