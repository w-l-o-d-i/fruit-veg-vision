import os
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tensorflow as tf
from PIL import Image
import pandas as pd
from typing import Dict, List, Optional
import io
from pathlib import Path

app = FastAPI(title="Fruit and Vegetable Classifier API")

# Setup templates directory
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = "public/models/model.keras"
LABEL_PATH = "streamlit/labels.csv"
WEIGHTS_PATH = "streamlit/fruits_vegetables_weights_full (1).csv"
IMAGE_SIZE = (224, 224)

# Global variables to store the loaded model and data
model = None
labels = []
weights_data = {}

# Load the model and data
async def load_resources():
    global model, labels, weights_data
    
    try:
        # Load the Keras model
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        # Load labels
        df_labels = pd.read_csv(LABEL_PATH)
        labels = df_labels['label'].tolist()
        
        # Load weights data
        df_weights = pd.read_csv(WEIGHTS_PATH, sep=';')
        df_weights['min'] = df_weights['min'].astype(str)
        df_weights['max'] = df_weights['max'].astype(str)
        df_weights['avg'] = df_weights['avg'].astype(str)
        weights_data = df_weights.set_index('name').to_dict('index')
        
        print("Model and data loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading resources: {str(e)}")
        return False

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess the image for model inference."""
    img = image.convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)
    img_batch = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_batch

@app.on_event("startup")
async def startup_event():
    """Load the model and data when the application starts."""
    success = await load_resources()
    if not success:
        raise RuntimeError("Failed to load model or data")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict the class of an uploaded image.
    
    Args:
        file: The image file to classify
        
    Returns:
        JSON response with prediction results
    """
    if model is None or not labels or not weights_data:
        raise HTTPException(status_code=503, detail="Model or data not loaded")
    
    # Check if the file is an image
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File is not an image")
    
    try:
        # Read and preprocess the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        img_batch = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(img_batch)
        predicted_class_index = int(np.argmax(prediction))
        
        if 0 <= predicted_class_index < len(labels):
            predicted_label = labels[predicted_class_index]
            confidence = float(np.max(prediction[0]) * 100)
            
            # Get weight information
            item_weights = weights_data.get(predicted_label, {})
            
            return {
                "success": True,
                "prediction": {
                    "label": predicted_label,
                    "confidence": confidence,
                    "weights": {
                        "min": item_weights.get("min", "N/A"),
                        "avg": item_weights.get("avg", "N/A"),
                        "max": item_weights.get("max", "N/A")
                    }
                }
            }
        else:
            return {
                "success": False,
                "error": f"Invalid prediction index: {predicted_class_index}"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None and labels and weights_data else "unhealthy",
        "model_loaded": model is not None,
        "labels_loaded": len(labels) > 0,
        "weights_loaded": len(weights_data) > 0
    }

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

if __name__ == "__main__":
    # Create static directory if it doesn't exist
    (BASE_DIR / "static").mkdir(exist_ok=True)
    
    # Run the application
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
