from routes import app
from ml_model import load_or_train_model

if __name__ == '__main__':
    # Initialize model on startup
    if load_or_train_model():
        print("Flask app starting...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize model. Please check your dataset.")
