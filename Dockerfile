# Multi-mode Docker setup for RedRob Ranker
# Build modes: RANK (CPU, minimal), PRECOMPUTE (GPU, full ML stack), SERVE (demo UI)
#
# Build examples:
#   docker build -t redrob:rank --build-arg ENV_MODE=RANK .
#   docker build -t redrob:precompute --build-arg ENV_MODE=PRECOMPUTE .
#   docker build -t redrob:serve --build-arg ENV_MODE=SERVE .

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# App code is promoted to the image root (not versions/vX/), so rank.py / precompute.py
# must resolve the repo root to /app instead of dirname×3(__file__).
ENV RANKER_ROOT=/app

WORKDIR /app

# LightGBM needs the OpenMP runtime (libgomp), absent from python:slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Build-time argument for environment mode
ARG ENV_MODE=RANK

# Copy all requirements files
COPY requirements*.txt ./

# Install dependencies based on environment mode
RUN set -e; \
    if [ "$ENV_MODE" = "PRECOMPUTE" ]; then \
        echo "Installing PRECOMPUTE dependencies (ML + GPU support)..."; \
        pip install --no-cache-dir -r requirements-precompute.txt; \
    elif [ "$ENV_MODE" = "SERVE" ]; then \
        echo "Installing SERVE dependencies (demo UI)..."; \
        pip install --no-cache-dir -r requirements-rank.txt; \
        pip install --no-cache-dir gradio>=4.0; \
    else \
        echo "Installing RANK dependencies (CPU only, minimal)..."; \
        pip install --no-cache-dir -r requirements-rank.txt; \
    fi && \
    pip list

# Copy application code
COPY . .

# Set environment mode for runtime
ENV ENV_MODE=${ENV_MODE}

# Expose ports (7860 for Gradio, 8000 for API)
EXPOSE 7860 8000

# Make entrypoint executable
RUN chmod +x entrypoint.sh || true

# Run the appropriate entrypoint
ENTRYPOINT ["./entrypoint.sh"]
CMD ["${ENV_MODE}"]
