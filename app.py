import os
import traceback
from flask import Flask, jsonify
from flask_cors import CORS

# =============================================================
# ✅ Environment Configuration
# =============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MPLCONFIGDIR = os.path.join(BASE_DIR, "mpl_tmp")

# Environment setup
os.environ["MODEL_DIR"] = MODEL_DIR
os.environ["MPLCONFIGDIR"] = MPLCONFIGDIR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Silence TensorFlow logs
os.environ["GLOG_minloglevel"] = "2"      # Silence MediaPipe logs

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MPLCONFIGDIR, exist_ok=True)

# =============================================================
# ✅ Flask App Initialization
# =============================================================
app = Flask(__name__)

# ✅ CORS Configuration (only once — no duplicates)
CORS(
    app,
    resources={r"/api/*": {"origins": [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://proctorvision-client.vercel.app",
        "https://proctor-vision-client.vercel.app",
        r"https://.*\.vercel\.app"
    ]}},
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"]
)

# =============================================================
# ✅ Register Blueprints (Classification)
# =============================================================
try:
    print("🔍 Importing classification routes...")
    from routes.classification_routes import classification_bp
    app.register_blueprint(classification_bp, url_prefix="/api")
    print("✅ classification_bp registered successfully.")
except Exception as e:
    print("⚠️ Failed to import or register classification routes.")
    print("🚨 Exception type:", type(e).__name__)
    print("📝 Exception message:", str(e))
    traceback.print_exc()

# =============================================================
# 🔎 Debug: Show Registered Routes
# =============================================================
print("\n🔍 Registered Routes:")
for rule in app.url_map.iter_rules():
    print(f"➡ {rule.endpoint} → {rule}")

# =============================================================
# 🌐 Root Health Check Route
# =============================================================
@app.route("/")
def home():
    """Health check route to verify server status."""
    routes = [str(rule) for rule in app.url_map.iter_rules()]
    return jsonify({
        "status": "ok",
        "message": "✅ ProctorVision AI Backend is Running on Railway",
        "available_routes": routes
    }), 200

# =============================================================
# 🚀 Main Entrypoint (Railway)
# =============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Railway auto-assigns this
    print(f"\n🚀 Starting Flask on Railway (port={port}) ...")
    app.run(host="0.0.0.0", port=port, debug=False)
