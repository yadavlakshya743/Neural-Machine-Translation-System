# Hugging Face Space image: React UI + FastAPI API in one container on port 7860.
# Assembled and uploaded by scripts/setup_space.py (also run by the GitHub deploy workflow).
# Trained checkpoints are NOT baked in: scripts/fetch_models.py downloads the latest ones
# from the Hugging Face model repo named in the HF_MODEL_REPO Space variable at startup.

FROM node:20-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    DEVICE=cpu \
    FRONTEND_DIST=/app/frontend_dist \
    DATABASE_URL=sqlite:////app/data/translation_history.db \
    DATA_REPORT_PATH=/app/models/data_report.json \
    HOME=/home/user

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt "huggingface_hub>=0.24"

COPY ml/ ml/
COPY backend/ backend/
COPY scripts/fetch_models.py scripts/fetch_models.py
COPY --from=web /web/dist /app/frontend_dist
RUN mkdir -p /app/models /app/data && chown -R user /app
USER user

EXPOSE 7860
CMD python scripts/fetch_models.py && uvicorn backend.app.main:app --host 0.0.0.0 --port 7860
