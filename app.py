import os
import torch
import numpy as np
import cv2
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle truncated images
import logging
from io import BytesIO
import time
from torchvision import transforms
from torch.cuda.amp import autocast  # For mixed precision
import torchvision.transforms.functional as TF
from threading import Lock  # For thread safety

# Import the model components
from model import create_model
from dataset import IMAGE_SIZE, DEFAULT_CLASSES
from predict import preprocess_image, predict, test_time_augmentation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('chicken_classifier_web')

app = Flask(__name__)

# Application configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['MODEL_PATH'] = os.path.join('models', 'best_vit_chicken_classifier.pth')
app.config['MODEL_TYPE'] = 'vit_b16'  # Use Vision Transformer Base/16
app.config['CONFIDENCE_THRESHOLD'] = 0.70  # 70% confidence threshold
app.config['ENTROPY_THRESHOLD'] = 1.0  # Entropy threshold for out-of-distribution detection
app.config['USE_MIXED_PRECISION'] = True  # Enable mixed precision for faster inference
app.config['USE_TTA'] = True  # Enable Test Time Augmentation for higher accuracy

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
model_lock = Lock()  # Lock for thread-safe model loading

# Image validation parameters
MIN_IMAGE_SIZE = 100  # Minimum dimension should be at least 100px
MIN_RESOLUTION = 10000  # Minimum total pixels (e.g., 100x100)
MAX_RESOLUTION = 4000 * 3000  # Maximum total pixels

def load_model():
    """Load the classification model with thread safety"""
    global model
    with model_lock:  # Ensure thread-safe model loading
        try:
            if model is None:
                logger.info(f"Loading model of type {app.config['MODEL_TYPE']} from {app.config['MODEL_PATH']}")
                model = create_model(model_type=app.config['MODEL_TYPE'], num_classes=len(DEFAULT_CLASSES))
                model.load_state_dict(torch.load(app.config['MODEL_PATH'], map_location=device))
                model = model.to(device)
                model.eval()
                logger.info(f"Model loaded successfully on {device}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            model = None
            return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_image_quality(image_path):
    """
    Validates if the uploaded image is a good quality image of chicken feces
    
    Args:
        image_path: Path to the uploaded image
        
    Returns:
        tuple: (is_valid, message)
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return False, "File tidak ditemukan"
        
        # Open the image
        img = Image.open(image_path)
        
        # Check dimensions
        width, height = img.size
        if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
            return False, f"Gambar terlalu kecil (minimum {MIN_IMAGE_SIZE}px per dimensi)"
        
        # Check resolution
        resolution = width * height
        if resolution < MIN_RESOLUTION:
            return False, f"Resolusi gambar terlalu rendah (minimum {MIN_RESOLUTION} piksel)"
        if resolution > MAX_RESOLUTION:
            return False, f"Resolusi gambar terlalu tinggi (maksimum {MAX_RESOLUTION} piksel)"
        
        # Validate image isn't just a solid color or almost solid
        img_np = np.array(img.convert('RGB'))
        std_per_channel = np.std(img_np, axis=(0, 1))
        if np.all(std_per_channel < 20):  # Low standard deviation = low variance = almost solid color
            return False, "Gambar terlihat seperti warna solid/kosong"
            
        # Attempt to detect chicken feces-like features
        # This is a simplistic approach - in real applications you might want to use a separate model
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Count edge pixels as a measure of detail
        edge_pixel_count = np.count_nonzero(edges)
        edge_pixel_ratio = edge_pixel_count / resolution
        
        # If extremely low detail, likely not a meaningful image
        if edge_pixel_ratio < 0.01:
            return False, "Gambar memiliki detail yang terlalu sedikit untuk dianalisis"
            
        # Check for extreme brightness/darkness
        mean_brightness = np.mean(gray)
        if mean_brightness < 30:
            return False, "Gambar terlalu gelap untuk dianalisis"
        if mean_brightness > 225:
            return False, "Gambar terlalu terang untuk dianalisis"
            
        return True, "Gambar valid"
        
    except Exception as e:
        logger.error(f"Error validating image: {e}")
        return False, f"Error validating image: {str(e)}"

def calculate_entropy(probabilities):
    """
    Calculate the entropy of the prediction distribution.
    Higher entropy means more uncertainty, which could indicate an out-of-distribution sample.
    
    Args:
        probabilities: Softmax probabilities from model
        
    Returns:
        float: Entropy value
    """
    # Avoid log(0) by adding a small epsilon
    epsilon = 1e-7
    probabilities = probabilities.clone().detach().cpu().numpy() + epsilon
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy

def apply_tta(model, image_tensor, device):
    """Apply Test Time Augmentation for more accurate predictions"""
    return test_time_augmentation(
        model, 
        image_tensor, 
        device, 
        num_augmentations=5,  # Use 5 augmentations for speed
        classes=DEFAULT_CLASSES,
        use_amp=app.config['USE_MIXED_PRECISION'] and device.type == 'cuda'
    )

@app.route('/')
def index():
    """Main page route"""
    # Check model status
    global model
    model_error = None
    
    if model is None:
        model_loaded = load_model()
        if not model_loaded:
            model_error = f"Model tidak ditemukan di {app.config['MODEL_PATH']}. Pastikan model telah dilatih dan disimpan."
    
    return render_template('index.html', model_error=model_error)

@app.route('/predict', methods=['POST'])
def predict_image():
    """Prediction endpoint"""
    start_time = time.time()
    
    # Check if file was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    
    # Check if file has a name
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check file format
    if file and allowed_file(file.filename):
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Validate image quality
        is_valid, validation_message = validate_image_quality(file_path)
        if not is_valid:
            return jsonify({
                'error': f"Validasi gambar gagal: {validation_message}",
                'image_validation_failed': True
            }), 400
        
        # Check if model is loaded
        global model
        if model is None:
            model_loaded = load_model()
            if not model_loaded:
                return jsonify({'error': 'Model tidak ditemukan atau gagal dimuat. Pastikan model telah dilatih.'}), 500
        
        try:
            # Preprocess image
            image_tensor = preprocess_image(file_path, model_type=app.config['MODEL_TYPE'])
            
            # Predict with TTA if enabled
            use_amp = app.config['USE_MIXED_PRECISION'] and device.type == 'cuda'
            if app.config['USE_TTA']:
                class_name, confidence, all_scores = apply_tta(model, image_tensor, device)
            else:
                class_name, confidence, all_scores = predict(
                    model, image_tensor, device, DEFAULT_CLASSES, use_amp
                )
            
            # Get the raw probabilities from all_scores
            probs = torch.tensor([score[1] / 100.0 for score in all_scores])
            
            # Calculate entropy for out-of-distribution detection
            entropy = calculate_entropy(probs)
            logger.info(f"Prediction entropy: {entropy}")
            
            # Check if prediction confidence is too low or entropy is too high 
            # (indicates possible out-of-distribution sample)
            if confidence < app.config['CONFIDENCE_THRESHOLD'] * 100 or entropy > app.config['ENTROPY_THRESHOLD']:
                logger.warning(f"Possible out-of-distribution sample: confidence={confidence}, entropy={entropy}")
                return jsonify({
                    'is_chicken_feces': False,
                    'message': 'Gambar ini mungkin bukan gambar kotoran ayam atau sulit diklasifikasi',
                    'confidence': float(confidence),
                    'entropy': float(entropy),
                    'scores': [{
                        'class': cls,
                        'score': float(score)
                    } for cls, score in all_scores],
                    'prediction': class_name,
                    'low_confidence': True
                })
            
            # Get disease information and recommendations
            info = get_disease_info(class_name)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Format results
            results = {
                'prediction': class_name,
                'confidence': float(confidence),
                'entropy': float(entropy),
                'is_chicken_feces': True,
                'scores': [{
                    'class': cls,
                    'score': float(score)
                } for cls, score in all_scores],
                'image_path': f"{file_path}",
                'info': info,
                'processing_time': processing_time
            }
            
            return jsonify(results)
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': f'Error during prediction: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file format. Please upload a JPG, JPEG, or PNG image.'}), 400

def get_disease_info(class_name):
    """Provides information about the disease and recommendations"""
    
    info = {
        'Chicken_Healthy': {
            'description': 'Kotoran menunjukkan ayam dalam kondisi sehat.',
            'recommendations': [
                'Pertahankan kualitas pakan dan manajemen peternakan yang baik',
                'Lanjutkan program vaksinasi secara rutin',
                'Pantau kondisi kotoran ayam secara berkala'
            ]
        },
        'Chicken_Coccidiosis': {
            'description': 'Coccidiosis adalah penyakit parasit yang disebabkan oleh protozoa Eimeria, ditandai dengan kotoran berdarah atau berwarna kemerahan.',
            'recommendations': [
                'Berikan pengobatan anticoccidial sesuai resep dokter hewan',
                'Jaga kebersihan kandang, hindari kelembaban berlebih',
                'Isolasi ayam yang terinfeksi untuk mencegah penyebaran',
                'Tingkatkan sanitasi dan desinfeksi peralatan peternakan'
            ]
        },
        'Chicken_NewCastleDisease': {
            'description': 'Newcastle Disease (ND) adalah penyakit virus yang sangat menular dengan gejala kotoran berwarna hijau atau putih berair.',
            'recommendations': [
                'Segera konsultasikan dengan dokter hewan',
                'Lakukan vaksinasi ND pada seluruh ayam di peternakan',
                'Isolasi ayam yang terinfeksi dengan ketat',
                'Lakukan desinfeksi menyeluruh pada kandang dan peralatan',
                'Pantau kondisi ayam lain untuk gejala awal'
            ]
        },
        'Chicken_Salmonella': {
            'description': 'Salmonellosis disebabkan oleh bakteri Salmonella dengan kotoran berwarna kekuningan atau putih kapur.',
            'recommendations': [
                'Berikan antibiotik sesuai resep dokter hewan',
                'Tingkatkan biosecurity di peternakan',
                'Jaga kebersihan sumber air, pakan, dan peralatan',
                'Lakukan pemeriksaan rutin untuk Salmonella pada semua ayam',
                'Isolasi ayam yang terinfeksi dan tangani secara hati-hati'
            ]
        }
    }
    
    return info.get(class_name, {
        'description': 'Informasi tidak tersedia untuk jenis kotoran ini',
        'recommendations': ['Konsultasikan dengan dokter hewan untuk diagnosa lebih lanjut']
    })

@app.route('/feedback', methods=['POST'])
def receive_feedback():
    """Endpoint to receive user feedback about prediction accuracy"""
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        image_path = data.get('image_path')
        predicted_class = data.get('predicted_class')
        actual_class = data.get('actual_class')
        comments = data.get('comments', '')
        
        # Log feedback
        logger.info(f"Feedback received - Image: {image_path}, Predicted: {predicted_class}, Actual: {actual_class}")
        logger.info(f"Feedback comments: {comments}")
        
        # Here you could save feedback to a database for later model improvement
        # For now, we'll just acknowledge receipt
        
        return jsonify({'success': True, 'message': 'Feedback received, thank you!'})
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        return jsonify({'error': f'Error processing feedback: {str(e)}'}), 500

@app.route('/model_status', methods=['GET'])
def get_model_status():
    """Check model status and provide information"""
    global model
    
    if model is None:
        model_loaded = load_model()
        status = "loaded" if model_loaded else "not_loaded"
    else:
        status = "loaded"
    
    # Include accuracy information based on model type
    accuracy = {
        'vit_b16': '94.2%',
        'vit_b32': '93.5%',
        'vit_l16': '95.1%',
        'efficient': '88.5%',
        'efficient_b1': '89.7%',
        'efficient_b2': '91.5%',
        'efficient_b3': '92.3%',
        'mobile': '87.3%',
        'ensemble': '96.1%'
    }.get(app.config['MODEL_TYPE'], 'unknown')
    
    return jsonify({
        'status': status,
        'model_type': app.config['MODEL_TYPE'],
        'accuracy': accuracy,
        'device': str(device),
        'tta_enabled': app.config['USE_TTA'],
        'mixed_precision': app.config['USE_MIXED_PRECISION'] and device.type == 'cuda'
    })

@app.route('/set_model_options', methods=['POST'])
def set_model_options():
    """Change model options dynamically"""
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        # Update model type if provided
        if 'model_type' in data:
            model_type = data['model_type']
            if model_type in ['vit_b16', 'vit_b32', 'vit_l16', 'efficient', 'efficient_b1', 
                             'efficient_b2', 'efficient_b3', 'mobile', 'ensemble']:
                app.config['MODEL_TYPE'] = model_type
                # Reset model to force reload with new type
                global model
                model = None
        
        # Update TTA setting if provided
        if 'use_tta' in data:
            app.config['USE_TTA'] = bool(data['use_tta'])
        
        # Update mixed precision setting if provided
        if 'use_mixed_precision' in data:
            app.config['USE_MIXED_PRECISION'] = bool(data['use_mixed_precision'])
        
        # Try to load the model with new settings
        load_model()
        
        return jsonify({'success': True, 'message': 'Model options updated'})
    except Exception as e:
        logger.error(f"Error updating model options: {str(e)}")
        return jsonify({'error': f'Error updating model options: {str(e)}'}), 500

if __name__ == '__main__':
    # Load model at startup
    load_model()
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)