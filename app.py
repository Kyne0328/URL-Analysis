from routes import app
from ml_model import load_or_train_model

print("Initializing and loading ML model for production...")
model_loaded = load_or_train_model()
if not model_loaded:
    print("FATAL: ML model failed to load. The application may not work correctly.")

if __name__ == '__main__':
    print("Flask app starting in debug mode...")
    app.run(debug=True, host='0.0.0.0', port=5000)
