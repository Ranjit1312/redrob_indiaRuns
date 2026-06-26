#!/usr/bin/env sh
# RedRob v7 container entrypoint. Mode is selected by $ENV_MODE (baked at build
# time, overridable at run time with `-e ENV_MODE=...`).
set -e

: "${ENV_MODE:=RANK}"
: "${CANDIDATES:=candidates.jsonl}"   # bind-mounted at run time
: "${OUT:=submission.csv}"
export RANKER_ROOT="${RANKER_ROOT:-/app}"

echo "[entrypoint] ENV_MODE=$ENV_MODE RANKER_ROOT=$RANKER_ROOT CANDIDATES=$CANDIDATES"

case "$ENV_MODE" in

  PRECOMPUTE)
    # Offline GPU build of artifacts_v7/. Each line prints its own wall/device
    # timing; redirect stdout to a log to capture telemetry for the README.
    #   step 1 re-runs only when the candidate POOL changes
    #   steps 2-4 re-run when the JD changes (edit jd/jd_profile.yaml first)
    echo "[precompute] 1/4 embed_candidates.py  (GPU, JD-independent embeddings)"
    python embed_candidates.py --candidates "$CANDIDATES"
    echo "[precompute] 2/4 jd_compile.py        (JD seam -> jd_vectors + bm25_facets)"
    python jd_compile.py
    echo "[precompute] 3/4 rank.py --features-only  (live feature pass for training)"
    python rank.py --candidates "$CANDIDATES" --features-only
    echo "[precompute] 4/4 train.py             (cross-encoder teacher -> LightGBM student)"
    python train.py
    echo "[precompute] DONE -> $RANKER_ROOT/artifacts_v7/ (embeddings + model.txt + feature_cols.json)"
    ;;

  SERVE)
    # HuggingFace Space sandbox: Gradio UI over the constrained rank step.
    exec python app.py
    ;;

  *)  # RANK — the judged path: CPU only, <5 min, <=16 GB, offline.
    python rank.py --candidates "$CANDIDATES" --out "$OUT"
    # Pre-flight the output against the Stage-1 auto-validator checks. Drop
    # --candidates here: the id-exists scan is a redundant full pass over the
    # big mounted file (ids already come from the precomputed pool).
    exec python validate_submission.py --submission "$OUT"
    ;;
esac
