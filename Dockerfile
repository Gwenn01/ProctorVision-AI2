# ================================================
# 🐍 Base Image
# ================================================
FROM python:3.10-slim-bullseye

# ================================================
# ⚙️ Install required system dependencies
# ================================================
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libatlas-base-dev \
    ffmpeg \
    pkg-config \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# ================================================
# 📦 Set working directory
# ================================================
WORKDIR /app

# ================================================
# 📄 Copy and install dependencies
# ================================================
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ================================================
# 📁 Copy all source files
# ================================================
COPY . .

# ================================================
# 🌐 Expose Railway’s dynamic port
# ================================================
EXPOSE 8080

# ================================================
# 🚀 Start app using Gunicorn (Production WSGI)
# ================================================
# Railway automatically sets the PORT environment variable.
# Use 0.0.0.0 to allow external access.
CMD ["gunicorn", "app:app", "--workers", "2", "--threads", "4", "--timeout", "300", "--bind", "0.0.0.0:8080"]
