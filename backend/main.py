from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import tensorflow as tf
import cv2
import tempfile
from model.utils import preprocess_image, calculate_blur_score

# ================================================
# ⚙️ CONFIG
# ================================================
MODEL_PATH = "model/forgery_model.keras"

# Load model once at startup
print("🔄 Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ================================================
# 🚀 FastAPI App Initialization
# ================================================
app = FastAPI(title="Document Forgery Detection API")

# Enable CORS (so React frontend can call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================
# 🧠 Prediction Function
# ================================================
def predict_forgery(image_path: str):
    img = preprocess_image(image_path, img_size=224)
    pred = model.predict(img)[0][0]

    forged_prob = float(pred) * 100
    original_prob = 100 - forged_prob
    label = "Forged" if pred > 0.5 else "Original"
    blur_score = calculate_blur_score(image_path)

    return {
        "prediction": label,
        "forged_prob": round(forged_prob, 2),
        "original_prob": round(original_prob, 2),
        "blur_score": round(blur_score, 2)
    }

# ================================================
# 📸 API Route: Upload & Predict
# ================================================
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Run prediction
        result = predict_forgery(tmp_path)

        return {
            "filename": file.filename,
            "prediction": result["prediction"],
            "forged_prob": result["forged_prob"],
            "original_prob": result["original_prob"],
            "blur_score": result["blur_score"]
        }

    except Exception as e:
        return {"error": str(e)}

# ================================================
# 🏠 Root Route
# ================================================
@app.get("/")
def home():
    return {"message": "🧠 Document Forgery Detection API is running!"}

# ================================================
# ▶️ Run Server
# ================================================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5050)
