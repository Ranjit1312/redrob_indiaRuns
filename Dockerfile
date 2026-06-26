# Multi-mode Docker for the RedRob v7 "JD-seam" ranker.
#
#   ENV_MODE=RANK        CPU only, minimal deps — the JUDGED submission step
#                        (rank.py -> submission.csv -> validate). <=5 min, <=16 GB,
#                        no network, no GPU (torch is never installed in this image).
#   ENV_MODE=PRECOMPUTE  Full ML/GPU stack — the offline artifact build
#                        (embed_candidates -> jd_compile -> features -> train).
#   ENV_MODE=SERVE       CPU + Gradio — the HuggingFace Space sandbox (app.py on :7860).
#
# Build:
#   docker build -t redrob:rank       --build-arg ENV_MODE=RANK .
#   docker build -t redrob:precompute --build-arg ENV_MODE=PRECOMPUTE .
#   docker build -t redrob:serve      --build-arg ENV_MODE=SERVE .

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Code is promoted to the image root (not versions/vN/), so the BASE resolver in
# rank.py / embed_candidates.py / jd_compile.py / train.py must point at /app
# instead of dirname x3(__file__). artifacts_v7/ then lives at /app/artifacts_v7.
ENV RANKER_ROOT=/app

WORKDIR /app

# LightGBM needs the OpenMP runtime (libgomp), absent from python:slim.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ARG ENV_MODE=RANK

# Requirements first, so the dependency layer caches across code edits.
COPY requirements*.txt ./

RUN set -e; \
    if [ "$ENV_MODE" = "PRECOMPUTE" ]; then \
        echo "Installing PRECOMPUTE deps (ML + GPU: torch / sentence-transformers / rank_bm25)..."; \
        pip install --no-cache-dir -r requirements-precompute.txt; \
    elif [ "$ENV_MODE" = "SERVE" ]; then \
        echo "Installing SERVE deps (CPU rank stack + gradio)..."; \
        pip install --no-cache-dir -r requirements-rank.txt; \
        pip install --no-cache-dir "gradio>=4.0"; \
    else \
        echo "Installing RANK deps (CPU only, minimal — no torch)..."; \
        pip install --no-cache-dir -r requirements-rank.txt; \
    fi && \
    pip list

# Application code + (for RANK/SERVE) the precomputed artifacts_v7/.
# PRECOMPUTE generates artifacts_v7/ at run time into a mounted volume.
COPY . .

ENV ENV_MODE=${ENV_MODE}

# Gradio sandbox (SERVE). HuggingFace Spaces inject PORT=7860.
EXPOSE 7860

RUN chmod +x entrypoint.sh || true

ENTRYPOINT ["./entrypoint.sh"]
