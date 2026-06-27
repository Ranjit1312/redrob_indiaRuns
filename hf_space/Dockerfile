# HuggingFace Space (Docker SDK) — RedRob v7 sandbox.
#
# HF builds the Space's root Dockerfile WITHOUT --build-arg, so the multi-mode
# root Dockerfile would default to ENV_MODE=RANK and never install Gradio. This
# SERVE-only Dockerfile hardcodes the sandbox path: CPU rank stack + Gradio, then
# `python app.py` (which subprocesses rank.py over the bundled demo pool).
#
# Assemble the Space with this as the repo-root `Dockerfile` (see DOCKER_RUNBOOK
# §3 / the HF runbook). It needs, alongside it: app.py, rank.py,
# validate_submission.py, redrob_ranker/, jd/, requirements-rank.txt, pool.jsonl,
# and a demo-pool artifacts_v7/.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RANKER_ROOT=/app \
    ENV_MODE=SERVE

WORKDIR /app

# LightGBM needs the OpenMP runtime; python:slim doesn't ship it.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer first so it caches across code edits.
COPY requirements-rank.txt ./
RUN pip install --no-cache-dir -r requirements-rank.txt "gradio>=4.0"

COPY . .

# HF Spaces route the public URL to this port.
EXPOSE 7860

CMD ["python", "app.py"]
