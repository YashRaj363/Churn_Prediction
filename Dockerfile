# Churn Prediction API — Cloud Run compatible image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source + trained artifacts. The model is baked into the image so the
# container is self-contained; retrain + rebuild to ship a new model.
COPY config.py .
COPY src/ ./src/
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY models/ ./models/
COPY data/reference_profile.json ./data/reference_profile.json

# Cloud Run injects $PORT (default 8080). Bind to it.
ENV PORT=8080
EXPOSE 8080

# Shell form so $PORT expands at runtime.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
