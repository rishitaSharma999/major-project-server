from flask import Flask, request, jsonify, render_template
import os
import uuid
import json
import numpy as np
import gdown
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define local paths and environment variables for Google Drive URLs
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, 'final_model.keras')
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, 'class_indices.json')

# Use direct download URLs with file IDs instead of shareable links
MODEL_FILE_ID = os.environ.get("MODEL_FILE_ID", "1GoneNJyyl-Hy1O_QWl4-viZ_fKHoKfws")
CLASS_INDICES_FILE_ID = os.environ.get("CLASS_INDICES_FILE_ID", "1X48AVSq7dHfj5xzMNa7LGvncYRqxN8f2")

# Function to download model with proper error handling
def download_model():
    if not os.path.exists(MODEL_PATH):
        try:
            print(f"Downloading model with file ID: {MODEL_FILE_ID}...")
            url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
            print(f"Model downloaded successfully to {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            # If model fails to download, create a placeholder file to indicate failure
            with open(f"{MODEL_PATH}.failed", "w") as f:
                f.write(f"Download failed: {str(e)}")
            return False
    else:
        print("Model already exists locally.")
        return True

# Function to download class indices with proper error handling  
def download_class_indices():
    if not os.path.exists(CLASS_INDICES_PATH):
        try:
            print(f"Downloading class indices with file ID: {CLASS_INDICES_FILE_ID}...")
            url = f'https://drive.google.com/uc?id={CLASS_INDICES_FILE_ID}'
            gdown.download(url, CLASS_INDICES_PATH, quiet=False)
            print(f"Class indices downloaded successfully to {CLASS_INDICES_PATH}")
            return True
        except Exception as e:
            print(f"Error downloading class indices: {str(e)}")
            # If class indices fail to download, create a placeholder file to indicate failure
            with open(f"{CLASS_INDICES_PATH}.failed", "w") as f:
                f.write(f"Download failed: {str(e)}")
            return False
    else:
        print("Class indices already exist locally.")
        return True

# Try downloading files on startup
model_download_success = download_model()
class_indices_download_success = download_class_indices()

# Define image size for preprocessing
IMAGE_SIZE = (224, 224)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_species(image_path, model_path=None, class_indices_path=None):
    """
    Function for making predictions.
    
    Parameters:
        image_path (str): Path to the image file.
        model_path (str): Path to the saved model (optional).
        class_indices_path (str): Path to saved class indices JSON (optional).
    
    Returns:
        dict: Prediction results including species and confidence.
    """
    if model_path is None:
        model_path = MODEL_PATH
    if class_indices_path is None:
        class_indices_path = CLASS_INDICES_PATH
    
    # Check if model and class indices are available
    if not os.path.exists(model_path):
        return {"error": "Model file not available. Please check server logs."}
    
    if not os.path.exists(class_indices_path):
        return {"error": "Class indices file not available. Please check server logs."}
    
    # Load the model
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return {"error": f"Could not load model: {str(e)}"}
    
    # Load class indices
    try:
        with open(class_indices_path, 'r') as f:
            class_indices_str = json.load(f)
            class_indices = {k: v for k, v in class_indices_str.items()}
    except Exception as e:
        print(f"Error loading class indices: {e}")
        return {"error": f"Could not load class indices: {str(e)}"}
    
    # Load and preprocess image
    try:
        img = image.load_img(image_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
    except Exception as e:
        print(f"Error processing image: {e}")
        return {"error": f"Could not process image: {str(e)}"}
    
    # Predict
    try:
        prediction = model.predict(img_array)
        predicted_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_idx])
        
        # Convert index to class name
        class_names = {int(v): k for k, v in class_indices.items()}
        species = class_names[predicted_idx]
        
        result = {
            'species': species,
            'confidence': confidence,
            'top_predictions': [
                {
                    'species': class_names[int(i)],
                    'confidence': float(prediction[0][i])
                }
                for i in np.argsort(-prediction[0])[:5]  # Top 5 predictions
            ]
        }
        return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": f"Prediction failed: {str(e)}"}

@app.route('/')
def home():
    # Add status check for model and class indices
    model_status = "Available" if os.path.exists(MODEL_PATH) else "Not Available"
    class_indices_status = "Available" if os.path.exists(CLASS_INDICES_PATH) else "Not Available"
    
    return render_template('index.html', 
                          model_status=model_status,
                          class_indices_status=class_indices_status)

@app.route('/status')
def status():
    """Endpoint to check the status of model and class indices files"""
    model_available = os.path.exists(MODEL_PATH)
    class_indices_available = os.path.exists(CLASS_INDICES_PATH)
    
    # Check for failed downloads
    model_failed = os.path.exists(f"{MODEL_PATH}.failed")
    class_indices_failed = os.path.exists(f"{CLASS_INDICES_PATH}.failed")
    
    # Read error messages if available
    model_error = None
    class_indices_error = None
    
    if model_failed:
        with open(f"{MODEL_PATH}.failed", "r") as f:
            model_error = f.read()
            
    if class_indices_failed:
        with open(f"{CLASS_INDICES_PATH}.failed", "r") as f:
            class_indices_error = f.read()
    
    return jsonify({
        "model_available": model_available,
        "class_indices_available": class_indices_available,
        "model_failed": model_failed,
        "class_indices_failed": class_indices_failed,
        "model_error": model_error,
        "class_indices_error": class_indices_error
    })

@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict():
    # Check if model and class indices are available
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not available. Please check server status at /status endpoint."})
    
    if not os.path.exists(CLASS_INDICES_PATH):
        return jsonify({"error": "Class indices not available. Please check server status at /status endpoint."})
    
    # Allow GET for testing /predict separately
    if request.method == 'GET':
        return jsonify({"message": "Predict endpoint is accessible."})
    
    # POST method for file upload and prediction
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        prediction_result = predict_species(
            image_path=file_path,
            model_path=MODEL_PATH,
            class_indices_path=CLASS_INDICES_PATH
        )
        
        return jsonify(prediction_result)
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/trigger-download', methods=['POST'])
def trigger_download():
    """Endpoint to manually trigger model and class indices download"""
    try:
        model_result = download_model()
        class_indices_result = download_class_indices()
        
        return jsonify({
            "model_download_success": model_result,
            "class_indices_download_success": class_indices_result
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)