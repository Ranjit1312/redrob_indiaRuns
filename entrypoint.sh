#!/usr/bin/env sh
set -e
: "${ENV_MODE:=RANK}"
: "${CANDIDATES:=candidates.jsonl}"
: "${OUT:=submission.csv}"
: "${COVERAGE:=skip}"     # RANK mode default: skip the candidate_id coverage scan.
                          # The mounted candidates.jsonl IS the precomputed pool, so the
                          # scan is a no-op — and a full pass over a bind-mounted ~487MB
                          # file costs minutes. Set COVERAGE=check to force it (new dataset).
case "$ENV_MODE" in
  PRECOMPUTE) exec python precompute.py --candidates "$CANDIDATES" ;;   # coverage-aware / incremental
  SERVE)      exec python app.py ;;                                     # Gradio demo on :7860
  *)          COV=""; [ "$COVERAGE" = "skip" ] && COV="--no-coverage-check"
              python rank.py --candidates "$CANDIDATES" --out "$OUT" $COV
              # validate format only (drop --candidates: the id-exists scan is a 3rd full
              # pass over the big file and is redundant — ids come from the precomputed pool)
              exec python validate_submission.py --submission "$OUT" ;;
esac