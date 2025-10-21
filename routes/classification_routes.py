import os, io, base64, sys, traceback, requests
from pathlib import Path
from flask import Blueprint, request, jsonify, make_response
from flask_cors import cross_origin
import tensorflow as tf
import numpy as np
from PIL import Image
from database.connection import get_db_connection

# =============================================================
# ✅ Blueprint Initialization
# =============================================================
classification_bp = Blueprint("classification_bp", __name__)

# ✅ Handle OPTIONS preflight requests globally (before POST)
@classification_bp.route("/<path:path>", methods=["OPTIONS"])
@cross_origin(
    origins=[
        "https://proctor-vision-client.vercel.app",
        "https://proctorvision-client.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    supports_credentials=True,
)
def handle_preflight(path):
    """Handles all OPTIONS preflight requests for API endpoints."""
    response = make_response(jsonify({"status": "ok"}), 200)
    response.headers["Access-Control-Allow-Origin"] = request.headers.get("Origin", "*")
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


# =============================================================
# ✅ TensorFlow / Model Setup
# =============================================================
try:
    from tensorflow.keras.applications import mobilenet_v2 as _mv2
except Exception:
    from keras.applications import mobilenet_v2 as _mv2

preprocess_input = _mv2.preprocess_input

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Hugging Face hosted model and threshold
MODEL_URLS = {
    "model": "https://huggingface.co/Gwen01/ProctorVision-Models/resolve/main/cheating_mobilenetv2_final.keras",
    "threshold": "https://huggingface.co/Gwen01/ProctorVision-Models/resolve/main/best_threshold.npy",
}

MODEL_PATH = MODEL_DIR / "cheating_mobilenetv2_final.keras"
THRESHOLD_PATH = MODEL_DIR / "best_threshold.npy"

def download_if_missing(url: str, dest: Path):
    """Download file from Hugging Face if not already cached locally."""
    try:
        if not dest.exists() or dest.stat().st_size < 10000:  # assume too small = invalid
            print(f"📥 Downloading {dest.name} from {url}...")
            r = requests.get(url)
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.write(r.content)
            print(f"✅ Downloaded {dest.name} ({dest.stat().st_size / 1_000_000:.2f} MB)")
        else:
            print(f"✅ {dest.name} already exists ({dest.stat().st_size / 1_000_000:.2f} MB)")
    except Exception as e:
        print(f"🚨 Failed to download {dest.name}: {e}")

# Download model + threshold from Hugging Face
download_if_missing(MODEL_URLS["model"], MODEL_PATH)
download_if_missing(MODEL_URLS["threshold"], THRESHOLD_PATH)

# Load TensorFlow model
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"🚨 Failed to load model: {e}")

# Load threshold
try:
    THRESHOLD = float(np.load(THRESHOLD_PATH)[0])
except Exception:
    THRESHOLD = 0.555
print(f"📊 Using decision threshold: {THRESHOLD:.3f}")

if model is not None:
    H, W = model.input_shape[1:3]
else:
    H, W = 224, 224

LABELS = ["Cheating", "Not Cheating"]


# =============================================================
# ✅ Utility Functions
# =============================================================
def preprocess_pil(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image -> model-ready tensor (1, H, W, 3)."""
    img = pil_img.convert("RGB")
    if img.size != (W, H):
        img = img.resize((W, H), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32)
    x = preprocess_input(x)
    return np.expand_dims(x, 0)


def predict_batch(batch_np: np.ndarray) -> np.ndarray:
    """Predict batch and return probability of 'Not Cheating'."""
    probs = model.predict(batch_np, verbose=0).ravel()
    if probs.ndim == 0:
        probs = np.array([probs])
    if len(probs) != batch_np.shape[0]:
        raw = model.predict(batch_np, verbose=0)
        if raw.ndim == 2 and raw.shape[1] == 2:
            probs = raw[:, 1]
        else:
            probs = raw.ravel()
    return probs


def label_from_prob(prob_non_cheating: float) -> str:
    """Return label based on probability threshold."""
    return LABELS[int(prob_non_cheating >= THRESHOLD)]


# =============================================================
# ✅ Route 1: classify uploaded multiple files
# =============================================================
@classification_bp.route("/classify_multiple", methods=["POST"])
@cross_origin(
    origins=[
        "https://proctor-vision-client.vercel.app",
        "https://proctorvision-client.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    supports_credentials=True,
)
def classify_multiple():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    files = request.files.getlist("files") if "files" in request.files else []
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    batch = []
    for f in files:
        try:
            pil = Image.open(io.BytesIO(f.read()))
            batch.append(preprocess_pil(pil)[0])
        except Exception as e:
            return jsonify({"error": f"Error reading an image: {str(e)}"}), 400

    batch_np = np.stack(batch, axis=0)
    probs = predict_batch(batch_np)
    labels = [label_from_prob(p) for p in probs]

    return jsonify({
        "threshold": THRESHOLD,
        "results": [
            {"label": lbl, "prob_non_cheating": float(p)}
            for lbl, p in zip(labels, probs)
        ],
    })


# =============================================================
# ✅ Route 2: classify suspicious behavior logs (DB)
# =============================================================
@classification_bp.route("/classify_behavior_logs", methods=["POST"])
@cross_origin(
    origins=[
        "https://proctor-vision-client.vercel.app",
        "https://proctorvision-client.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    supports_credentials=True,
)
def classify_behavior_logs():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    exam_id = data.get("exam_id")
    if not user_id or not exam_id:
        return jsonify({"error": "Missing user_id or exam_id"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT id, image_base64 FROM suspicious_behavior_logs
            WHERE user_id = %s AND exam_id = %s AND image_base64 IS NOT NULL
        """, (user_id, exam_id))
        logs = cursor.fetchall()

        CHUNK = 64
        for i in range(0, len(logs), CHUNK):
            chunk = logs[i:i+CHUNK]
            batch, ids = [], []
            for log in chunk:
                try:
                    img_data = base64.b64decode(log["image_base64"])
                    pil = Image.open(io.BytesIO(img_data))
                    batch.append(preprocess_pil(pil)[0])
                    ids.append(log["id"])
                except Exception as e:
                    print(f"⚠️ Failed to read image ID {log['id']}: {e}")

            if not batch:
                continue

            batch_np = np.stack(batch, axis=0)
            probs = predict_batch(batch_np)
            labels = [label_from_prob(p) for p in probs]

            cur2 = conn.cursor()
            for _id, lbl in zip(ids, labels):
                cur2.execute(
                    "UPDATE suspicious_behavior_logs SET classification_label=%s WHERE id=%s",
                    (lbl, _id)
                )
            conn.commit()

        conn.close()
        return jsonify({"message": "Classification complete.", "threshold": THRESHOLD}), 200

    except Exception as e:
        print("🚨 Exception inside /classify_behavior_logs route:", file=sys.stderr)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
