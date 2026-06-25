#!/usr/bin/env sh
set -e
: "${ENV_MODE:=RANK}"
: "${CANDIDATES:=candidates.jsonl}"
: "${OUT:=submission.csv}"
case "$ENV_MODE" in
  PRECOMPUTE) exec python precompute.py ;;                 # v5 orchestrator (or v6 chain)
  SERVE)      exec python app.py ;;                        # Gradio demo on :7860
  *)          python rank.py --candidates "$CANDIDATES" --out "$OUT"
              exec python validate_submission.py --submission "$OUT" --candidates "$CANDIDATES" ;;
esac