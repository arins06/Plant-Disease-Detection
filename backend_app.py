"""
Plant Disease Detection - Cloud Backend
=========================================
FastAPI service that:
  1. Receives a raw JPEG image (POST /predict) from the ESP32-CAM
  2. Runs it through a fine-tuned MobileNetV2 model trained on PlantVillage
  3. Returns {"status": "healthy" | "diseased", "confidence": float, "disease_class": str}
  4. Logs every prediction (timestamp, class, confidence) to a local SQLite DB
     so the dashboard can show history without needing a separate database service.

Run locally:
    pip install -r requirements.txt
    uvicorn backend_app:app --host 0.0.0.0 --port 8000

Deploy: works as-is on Render / Railway / any host that runs a Docker/Python app.
Free-tier friendly - no external DB required (SQLite file), no paid APIs.
"""

import io
import sqlite3
import datetime
from contextlib import contextmanager

import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf

# ---------------- CONFIG ----------------
MODEL_PATH = "plant_disease_model.h5"   # produced by train_model.py
DB_PATH = "predictions.db"
IMG_SIZE = (224, 224)                    # MobileNetV2 default input size

# Class labels must match the order used during training (see train_model.py's
# train_generator.class_indices after training - copy that dict here).
# These 10 match the recommended PlantVillage Tomato demo subset (see DATASET_GUIDE.md).
# ImageDataGenerator sorts folder names alphabetically, so this is the order it will
# actually produce - double check against your own printed class_indices after training.
CLASS_LABELS = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]
# -----------------------------------------

app = FastAPI(title="Plant Disease Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your dashboard's domain before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None  # lazy-loaded on first request to keep cold-start fast on free tiers


def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    return model


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL
            )
        """)
        conn.commit()


def preprocess_image(raw_bytes: bytes) -> tuple:
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    resized = img.resize(IMG_SIZE)
    arr = np.array(resized, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0), img  # model input, plus original PIL image for the leaf-color check


def looks_like_leaf(img: Image.Image, min_green_ratio: float = 0.12) -> bool:
    """
    Heuristic pre-filter: every training image is a leaf (green/brown) on some
    background. This won't catch everything, but it catches the clear cases -
    like a blown-out ceiling/wire photo with almost no green content - where
    the CNN can still be confidently wrong since it was never trained to say
    "this isn't a leaf at all."
    """
    hsv = img.resize((128, 128)).convert("HSV")
    arr = np.array(hsv)
    h, s, v = arr[..., 0], arr[..., 1], arr[..., 2]
    # PIL's HSV hue is 0-255 for 0-360 degrees; green/yellow-green/brown-ish
    # foliage roughly falls in ~35-140 on this scale, with some real saturation/brightness
    green_mask = (h >= 25) & (h <= 140) & (s >= 35) & (v >= 25)
    green_ratio = float(np.mean(green_mask))
    return green_ratio >= min_green_ratio


@app.on_event("startup")
def startup():
    init_db()


@app.post("/predict")
async def predict(request: Request):
    raw_bytes = await request.body()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="No image data received")

    try:
        input_tensor, original_img = preprocess_image(raw_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    if not looks_like_leaf(original_img):
        return {
            "status": "not_a_leaf",
            "disease_class": None,
            "confidence": None,
            "message": "This doesn't look like a plant leaf - too little green/foliage color detected. Point the camera at an actual leaf.",
        }

    predictions = get_model().predict(input_tensor)[0]
    class_index = int(np.argmax(predictions))
    confidence = float(predictions[class_index])
    predicted_class = CLASS_LABELS[class_index]
    status = "healthy" if predicted_class.endswith("healthy") else "diseased"

    with get_db() as conn:
        conn.execute(
            "INSERT INTO predictions (timestamp, predicted_class, status, confidence) VALUES (?, ?, ?, ?)",
            (datetime.datetime.utcnow().isoformat(), predicted_class, status, confidence),
        )
        conn.commit()

    return {
        "status": status,
        "disease_class": predicted_class,
        "confidence": round(confidence, 4),
    }


@app.get("/history")
async def history(limit: int = 20):
    """Returns recent predictions for the dashboard."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT timestamp, predicted_class, status, confidence FROM predictions "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {"timestamp": r[0], "disease_class": r[1], "status": r[2], "confidence": r[3]}
        for r in rows
    ]


@app.get("/health")
async def health_check():
    return {"ok": True}
