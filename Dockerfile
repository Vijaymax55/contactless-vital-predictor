FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System dependencies for OpenCV + scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1-mesa-glx \
        && rm -rf /var/lib/apt/lists/*

# ──────────────────────────────────────────────
# Build stage — install Python dependencies
# ──────────────────────────────────────────────
FROM base AS builder

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-deps -r requirements.txt

# ──────────────────────────────────────────────
# Runtime stage
# ──────────────────────────────────────────────
FROM base AS runtime

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source
COPY blusim/ /app/blusim/
COPY config/ /app/config/

# Model weights volume (mounted at runtime)
VOLUME ["/app/models"]

# Data volume (optional)
VOLUME ["/app/data"]

EXPOSE 8000

ENV BLUSIM_DATA_ROOT=/app/data \
    BLUSIM_MODELS_DIR=/app/models \
    BLUSIM_LOGS_DIR=/app/logs

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "blusim.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
