# RedRob Candidate Ranker — Intelligent Candidate Discovery & Ranking

Ranks the candidate pool for the Redrob "Senior AI Engineer — Founding Team" JD by
**meaning, not keyword matching**, and down-weights behaviourally unavailable or
trap candidates. Two phases: an **unconstrained offline precompute** (embeddings +
teacher→student distillation, GPU optional) and a **constrained CPU-only ranking
step** that produces the submission CSV in seconds.

## The one reproduce command (Stage-3 ranking step)

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py --submission ./submission.csv --candidates ./candidates.jsonl
```

CPU only, no network, ~3.6 s wall, ~0.34 GB peak RAM on the 100K pool — well inside
the 5 min / 16 GB / 5 GB budget (spec §3). Telemetry is written to `rank_telemetry.json`
on every run.

## Repository layout

| Path | Role |
|---|---|
| `rank.py` | **the constrained ranking step** (CPU-only by construction) |
| `precompute.py` | offline orchestrator (5 stages); coverage-aware / incremental |
| `run_step35.py`, `features.py` | stage 1 — bi-encoder facet embeddings + BM25 + structured features |
| `build_rule_features.py` | stage 2 — JD-rule evidence / integrity / assessment features |
| `precompute_teacher.py` | stage 3 — cross-encoder teacher (GPU) + signal features |
| `train_student.py` | stage 4 — tuned LightGBM LambdaMART student |
| `build_final_features.py` | stage 5 — full-profile-audit flags + blend |
| `validate_submission.py` | offline pre-flight validator (mirrors Stage-1 checks) |
| `app.py` | Gradio sandbox (HuggingFace Space / Docker SERVE mode) |
| `tests/` | challenge-oriented pytest suite (contract + traps + determinism) |
| `artifacts_full/` | the **~17 MB** the rank step loads (parquets / model / json) |
| `Dockerfile`, `entrypoint.sh` | multi-mode image: `RANK` / `PRECOMPUTE` / `SERVE` |
| `submission_metadata.yaml` | portal-metadata mirror (spec §10.2) — fill before submitting |
| `docs/` | `RESULTS.md` (telemetry) + `audit_report.md` (the 5-round audit loop) |

## Why `candidates.jsonl` and `*.npy` embeddings are **not** in this repository — and how it still runs on a new dataset

This is deliberate, and the code is built to handle a brand-new candidate set.

1. **`candidates.jsonl` (~487 MB) is organiser-provided input, not our code.** It is
   not ours to redistribute and exceeds GitHub's file limit. Mount/point to it at run
   time (`--candidates`, or `-v` for Docker).
2. **The bi-encoder embeddings (`artifacts_full/*.npy`, ~620 MB) are *derived*, not
   source.** They are fully regenerable from `candidates.jsonl` by `run_step35.py`, so
   we ship **the script, not the blob** (spec §10.3 explicitly allows "a script that
   produces them"). Only `run_step35.py` ever reads the `.npy`; nothing downstream does.
3. **What we *do* commit is the 17 MB the constrained rank step actually needs:**
   `features.parquet`, `features_refined_v3.parquet`, `features_v4.parquet`,
   `signals_features.parquet`, `model_v3.txt`, `feature_cols_v2.json`. That keeps the
   graded ranking step fully reproducible with no large download.

### Running on a NEW candidates dataset (the `candidate_id`-vs-parquet check)

`rank.py` checks every `candidate_id` in the supplied `candidates.jsonl` against the
ids already covered by `artifacts_full/features.parquet`:

- **Ids already covered** → ranked directly. On the official pool this is *all* of them,
  so the step is byte-identical to development.
- **Ids missing** (organisers supply new candidates) → `rank.py` prints a clear
  instruction and ranks the covered subset (or aborts under `--strict-coverage`). To
  fill the gap, run the **unconstrained** precompute, which embeds **only the missing
  candidates** and merges them in:

  ```bash
  python precompute.py --candidates ./candidates.jsonl   # embeds ONLY new ids, merges features.parquet
  python rank.py       --candidates ./candidates.jsonl --out ./submission.csv
  ```

  `precompute.py` diffs the ids first: if everything is already covered it is a **no-op**;
  if some are new it runs the single GPU-bound stage (bi-encoder embeddings) for just
  those ids and then re-derives the cheap CPU stages over the full pool so every table
  stays consistent. The trained LightGBM student generalises to the new candidates — no
  retraining required for the embeddings to be usable. This is why "precompute only the
  new candidates" is exactly what happens, and the constrained `rank.py` never embeds.

## GPU / CPU split — how we comply

- **GPU touches only precompute** (`run_step35.py` bi-encoder, `precompute_teacher.py`
  cross-encoder). Both print the device they load on; `precompute.py` prints
  `torch.cuda.is_available()` up front. The chain also runs CPU-only (slower).
- **The ranking step is CPU-only by construction.** `rank.py` imports only
  numpy/pandas/pyarrow/orjson/psutil/lightgbm — `torch`/`sentence-transformers` are
  never imported, are absent from `requirements-rank.txt`, and are not installed in the
  rank venv. There is no code path that could reach a GPU or the network.
  (`tests/test_rank_smoke.py` enforces this statically.)

See `docs/RESULTS.md` for measured precompute timings (≈20 min on an RTX 4050,
≈5–6 h CPU-only; one-time, skippable with the shipped artifacts).

## Tests

```bash
pip install pytest
set RANK_VERSION=5            # Windows (RANK_VERSION=5 on bash); gates version-specific traps
pytest -q                    # full suite (runs rank.py end-to-end — needs artifacts_full/ + candidates.jsonl)
pytest -q -m "not slow"      # fast: contract + trap + CPU-only-import guards
```

The suite encodes the challenge rules: the Stage-1 submission contract, a CPU-only/offline
import guard, run determinism ("100/100 identical"), and trap guards (no YoE-inflation
honeypot, CV-primary disqualifier, dormant+unresponsive, or certification anachronism in
the top-100). Trap tests are gated on `RANK_VERSION` so they document which audit round
closed each hole.

## Docker

```bash
# Constrained ranking step (CPU, minimal) — the Stage-3 reproduction
docker build -t redrob:rank --build-arg ENV_MODE=RANK .
docker run --rm -v "$PWD/candidates.jsonl:/app/candidates.jsonl" \
                -v "$PWD/out:/app/out" -e OUT=/app/out/submission.csv redrob:rank

# Gradio sandbox
docker build -t redrob:serve --build-arg ENV_MODE=SERVE .
docker run --rm -p 7860:7860 redrob:serve        # http://localhost:7860
```

## Sandbox

`app.py` is a CPU-only Gradio demo: upload ≤100 candidate profiles (JSON array or JSONL
matching `candidate_schema.json`) and download the ranked CSV; the bundled
`sample_candidates.json` is a one-click demo. Deployed as a HuggingFace Space (link in
`submission_metadata.yaml`).

## How we got here — iteration log (v1 → v5)

The ranking logic was built by a disciplined **rank → audit → base-rate-verify → encode →
re-rank** loop (full record in `docs/audit_report.md`); each step below is a real commit:

- **v1 — baseline composite.** Six-facet bi-encoder + BM25 hybrid with structured damps
  and a binary honeypot gate. A top-30 audit found 13/30 weak or red-flagged (blurry
  cosine, a CV-primary disqualifier at rank 10, a plain-language strong-fit buried at 91).
- **v2 — career-evidence layer + continuous integrity ladder.** Ownership×context×recency
  evidence from career descriptions (summaries excluded); the binary gate becomes a
  continuous integrity score; CV-primary damp, hopper damp, YoE-vs-history honeypot
  threshold. The stated-YoE honeypot falls to ~99,952.
- **v3 — evidence-gated skill assessments.** Proctored assessment scores as a gated
  "validated potential" signal (full credit only at evidence coverage ≥ 0.25), stopping
  single-test profiles from leaping ~200 ranks.
- **v4 — teacher → student distillation + full-profile audit.** A cross-encoder teacher
  (bge-reranker-v2-m3) labels a shortlist; labels are integrity/anti-stuffer gated and
  distilled into a tuned LightGBM LambdaMART student, blended at α=0.2 (rules dominate).
  Base-rate-verified audit rules added: dormant×0.5, anachronism×0.30, soft penalties for
  pervasive generator noise. **Rare patterns → strong rules; pervasive → soft penalties.**
- **v5-final — submission package.** The v4 method packaged per spec §10.3: one precompute
  orchestrator, one ranking command, the validator, `submission_metadata.yaml`, and the
  coverage-aware new-dataset handling above. Final state: zero disqualifiers / zero
  hallucinations / zero dormant profiles in the top-100; reproduction == development.

### v6 (experimental — intentionally not in this submission)

A later **JD-decoupled re-architecture** explored splitting the pipeline into a
JD-independent candidate side (`embed_candidates.py` → vectors + intrinsic features) and a
separately compiled JD profile (`jd_profile.yaml` → `jd_compile.py`), computing all
JD-dependent features live in `rank.py` and dropping BM25. It serves a *different* purpose
(fast re-targeting to a new JD without re-embedding candidates) and its artifacts are
incomplete, so it is **kept out of this submission line**; v5-final is the validated,
shipped ranker. The exploration is noted here for transparency about the iteration.
