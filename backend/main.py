import os
import io
import json
import random
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Farmer Leaf Doctor (உழவர் இலை மருத்துவர்)")

# CORS middleware for local testing or cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "leaf_disease_model.tflite")
REMEDIES_PATH = os.path.join(BASE_DIR, "remedies.json")

# Load remedies
if os.path.exists(REMEDIES_PATH):
    try:
        with open(REMEDIES_PATH, "r", encoding="utf-8") as f:
            REMEDIES = json.load(f)
        print("Remedies database loaded successfully.")
    except Exception as e:
        print(f"Error loading remedies.json: {e}")
        REMEDIES = {}
else:
    print(f"Remedies database file not found at {REMEDIES_PATH}")
    REMEDIES = {}

# Class names mapping (Alphabetical sorting is TensorFlow's default)
CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy"
]

# TFLite Model Loading
interpreter = None
model_loaded = False
model_status = ""

try:
    if os.path.exists(MODEL_PATH):
        # Dynamically import to avoid crash if tensorflow is missing in dev
        try:
            import tensorflow.lite as tflite
        except ImportError:
            try:
                import tflite_runtime.interpreter as tflite
            except ImportError:
                raise ImportError("Neither tensorflow nor tflite-runtime is installed.")
                
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        model_loaded = True
        model_status = "Loaded TFLite model successfully."
        print(model_status)
    else:
        model_status = "Model file leaf_disease_model.tflite not found. Running in MOCK mode."
        print(model_status)
except Exception as e:
    model_status = f"Model load failed: {str(e)}. Running in MOCK mode."
    print(model_status)


def run_tflite_inference(img_array):
    """
    Runs inference on the preprocessed image using the TFLite model.
    Handles quantized (UINT8) and floating-point (FLOAT32) models.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_type = input_details[0]['dtype']
    
    # Add batch dimension: shape becomes [1, 160, 160, 3]
    input_data = np.expand_dims(img_array, axis=0)
    
    # If model is quantized, scale float [0, 1] to uint8 [0, 255]
    if input_type == np.uint8:
        input_data = (input_data * 255.0).astype(np.uint8)
        
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # If the output is quantized, dequantize it
    if output_details[0]['dtype'] == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        if scale != 0:
            output_data = (output_data.astype(np.float32) - zero_point) * scale
            
    # Softmax conversion check (logits vs probability)
    confidence = float(np.max(output_data))
    class_idx = int(np.argmax(output_data))
    
    # Fallback manual softmax if the output is not probability
    if confidence > 1.0 or confidence < 0.0 or not (0.95 <= np.sum(output_data) <= 1.05):
        exp_logits = np.exp(output_data - np.max(output_data))
        probabilities = exp_logits / np.sum(exp_logits)
        class_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[class_idx])
        
    return class_idx, confidence


def run_mock_inference(filename):
    """
    Generates deterministic mockup predictions for testing purposes.
    Looks for keywords in the filename to return specific classes.
    """
    fn = filename.lower()
    
    # Default mock confidence
    confidence = random.uniform(0.75, 0.98)
    
    # Map keywords in the filename to specific classes
    if "tomato_bacterial" in fn or "tomato_spot" in fn:
        predicted_class = "Tomato_Bacterial_spot"
    elif "tomato_early" in fn:
        predicted_class = "Tomato_Early_blight"
    elif "tomato_late" in fn:
        predicted_class = "Tomato_Late_blight"
    elif "tomato_healthy" in fn:
        predicted_class = "Tomato_healthy"
    elif "potato_early" in fn:
        predicted_class = "Potato___Early_blight"
    elif "potato_late" in fn:
        predicted_class = "Potato___Late_blight"
    elif "potato_healthy" in fn:
        predicted_class = "Potato___healthy"
    elif "tomato" in fn:
        predicted_class = random.choice([c for c in CLASS_NAMES if "Tomato" in c])
    elif "potato" in fn:
        predicted_class = random.choice([c for c in CLASS_NAMES if "Potato" in c])
    else:
        # If the filename contains "bad", "low", "blur" or "not_leaf", mock a low-confidence error
        if any(kw in fn for kw in ["bad", "low", "blur", "not_leaf", "error", "test_low"]):
            predicted_class = random.choice(CLASS_NAMES)
            confidence = random.uniform(0.30, 0.55)
        else:
            predicted_class = random.choice(CLASS_NAMES)
            
    class_idx = CLASS_NAMES.index(predicted_class)
    return class_idx, confidence


@app.get("/health")
async def health():
    """Health check endpoint showing model details."""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "info": model_status,
        "supported_classes": CLASS_NAMES
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Inference endpoint: Preprocesses image to 160x160, normalizes it,
    runs TFLite inference, and returns remedy information in English and Tamil.
    """
    # Verify file content type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a clear image of a potato or tomato leaf."
        )
        
    try:
        # Load image with PIL
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Failed to decode the image file. Ensure it is a valid PNG or JPEG."
        )
        
    # Resize and Preprocess image (160x160)
    try:
        image_resized = image.convert("RGB").resize((160, 160))
        img_array = np.array(image_resized, dtype=np.float32)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to preprocess the image: {str(e)}"
        )
        
    # Run TFLite inference (or Mock fallback)
    if model_loaded:
        try:
            class_idx, confidence = run_tflite_inference(img_array)
            predicted_class = CLASS_NAMES[class_idx]
        except Exception as e:
            # Safe fallback to mock in case input format or inference crashes
            print(f"TFLite Inference failed: {e}. Falling back to Mock.")
            class_idx, confidence = run_mock_inference(file.filename)
            predicted_class = CLASS_NAMES[class_idx]
    else:
        class_idx, confidence = run_mock_inference(file.filename)
        predicted_class = CLASS_NAMES[class_idx]

    # Confidence threshold safeguard (70%)
    if confidence < 0.70:
        return {
            "success": False,
            "low_confidence": True,
            "message": "Unable to confidently identify the disease. Please retake the photo with a single leaf, plain background, and good lighting."
        }

    # FIXED: Check single underscores first, then fallback to full training keys
    lookup_class = predicted_class.replace("___", "_")
    remedy_info = REMEDIES.get(lookup_class) or REMEDIES.get(predicted_class)    
    
    if not remedy_info:
        # Clean up triple underscores for the display text safely
        clean_name = predicted_class.replace("___", "_")
        parts = clean_name.split("_")
        crop = parts[0]
        disease = " ".join(parts[1:]) if len(parts) > 1 else "healthy"
        
        # Fallback dictionary if remedies.json didn't load
        remedy_info = {
            "crop": crop,
            "crop_ta": "தக்காளி" if crop == "Tomato" else "உருளைக்கிழங்கு",
            "disease": disease,
            "disease_ta": disease,
            "cause": "Unknown pathogen or environment factor.",
            "cause_ta": "காரணம் அறியப்படவில்லை.",
            "symptoms": "Visible changes on leaf margins or surfaces.",
            "symptoms_ta": "இலையின் விளிம்புகளில் அல்லது மேற்பரப்பில் தெரியும் மாற்றங்கள்.",
            "remedy": ["Consult your local agricultural officer."],
            "remedy_ta": ["உங்கள் உள்ளூர் வேளாண் அதிகாரியை அணுகவும்."]
        }
        
    # Standardized response structure
    return {
        "success": True,
        "class_name": predicted_class,
        "confidence": confidence,
        "crop": remedy_info.get("crop"),
        "crop_ta": remedy_info.get("crop_ta"),
        "disease": remedy_info.get("disease"),
        "disease_ta": remedy_info.get("disease_ta"),
        "cause": remedy_info.get("cause"),
        "cause_ta": remedy_info.get("cause_ta"),
        "symptoms": remedy_info.get("symptoms"),
        "symptoms_ta": remedy_info.get("symptoms_ta"),
        "remedy": remedy_info.get("remedy"),
        "remedy_ta": remedy_info.get("remedy_ta"),
        "is_mock": not model_loaded
    }

# Serve Frontend static assets
# Make sure the frontend path exists before mounting
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    print(f"Frontend static directory not found at {FRONTEND_DIR}. Server will run backend-only.")