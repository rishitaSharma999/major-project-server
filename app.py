from flask import Flask, request, jsonify, render_template
import os
import uuid
import json
import numpy as np
import gdown  # used to download files from Google Drive
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)  # Enable CORS globally if desired

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

MODEL_DRIVE_URL = os.environ.get("MODEL_DRIVE_URL", "https://drive.google.com/uc?id=1GoneNJyyl-Hy1O_QWl4-viZ_fKHoKfws")
CLASS_INDICES_DRIVE_URL = os.environ.get("CLASS_INDICES_DRIVE_URL", "https://drive.google.com/uc?id=1X48AVSq7dHfj5xzMNa7LGvncYRqxN8f2")

# Download model and class indices if not present locally
if not os.path.exists(MODEL_PATH):
    print(f"Downloading model from {MODEL_DRIVE_URL} ...")
    gdown.download(id="1GoneNJyyl-Hy1O_QWl4-viZ_fKHoKfws", output=MODEL_PATH, quiet=False, fuzzy=True)
else:
    print("Model already exists locally.")

if not os.path.exists(CLASS_INDICES_PATH):
    print(f"Downloading class indices from {CLASS_INDICES_DRIVE_URL} ...")
    gdown.download(CLASS_INDICES_DRIVE_URL, CLASS_INDICES_PATH, quiet=False, fuzzy=True)
else:
    print("Class indices already exist locally.")

# Define image size for preprocessing (required for predict_species function)
IMAGE_SIZE = (224, 224)  # Adjust this based on your model requirements

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
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()  # Enable CORS explicitly for this endpoint
def predict():
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use PORT from env; default to 10000
    app.run(host="0.0.0.0", port=port)


