import os
import traceback
from flask import Flask, jsonify
from flask_cors import CORS

# =============================================================
# ✅ Environment Configuration
# =============================================================
os.environ["MODEL_DIR"] = "/tmp/model"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"

os.makedirs(os.environ["MODEL_DIR"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# =============================================================
# ✅ Flask Initialization
# =============================================================
app = Flask(__name__)

CORS(
    app,
    resources={r"/api/*": {
        "origins": [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "https://proctorvision-client.vercel.app",
            "https://proctorvision-server-production.up.railway.app",
        ]
    }},
    supports_credentials=True,
)

# =============================================================
# 🔍 Import Blueprint (Classification only)
# =============================================================
try:
    print("🔍 Attempting to import classification routes...")

    from routes.classification_routes import classification_bp
    app.register_blueprint(classification_bp, url_prefix="/api")

    print("✅ classification_bp registered successfully.")

except Exception as e:
    print("⚠️ Failed to import or register blueprints.")
    print("🚨 Exception type:", type(e).__name__)
    print("📝 Exception message:", str(e))
    print("📄 Full traceback:")
    traceback.print_exc()

# =============================================================
# 🔎 Debug: List Registered Routes
# =============================================================
print("\n🔍 Final Registered Routes:")
for rule in app.url_map.iter_rules():
    print(f"➡ {rule.endpoint} → {rule}")

# =============================================================
# 🌐 Root Route
# =============================================================
@app.route("/")
def home():
    routes = [str(rule) for rule in app.url_map.iter_rules()]
    return jsonify({
        "status": "ok",
        "message": "✅ ProctorVision AI Classification Backend Running",
        "available_routes": routes
    })

# =============================================================
# 🚀 Main Entrypoint
# =============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    print(f"\n🚀 Starting Flask server on port {port} (debug={debug})...")
    app.run(host="0.0.0.0", port=port, debug=debug)
