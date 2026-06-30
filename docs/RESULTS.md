# Version run results — telemetry & validation (2026-06-13, constant script names)

Rows **v1–v5 are the iteration line** — the audit-and-refine archive, each `rank.py`
in its own directory run in its **own slim venv** (Python 3.12.10;
numpy/pandas/pyarrow/orjson/psutil; +lightgbm v4/v5), telemetry per version
(`telemetry.json` / `rank_telemetry.json`). The **shipped submission is v7** (the
JD-seam architecture documented in the root `README.md`); its row is added at the
bottom and is the authoritative entry.

| Version | Wall | Budget | Peak RAM | Validation |
|---|---|---|---|---|
| v1 baseline composite | 2.1 s | 0.7% | 0.22 GB | PASS |
| v2 evidence + integrity* | 112.2 s | 37.4% | ~1.6 GB | PASS |
| v3 + assessments* | 114.5 s | 38.2% | ~1.6 GB | PASS |
| v4 + CE-distilled blend | 2.3 s | 0.8% | 0.34 GB | PASS |
| v5-final (iteration line) | 3.6 s | 1.2% | 0.34 GB | PASS |
| **v7 (shipped submission)†** | **38.8 s** | **12.9%** | **0.31 GB** | **PASS (11/11)** |

\* v2/v3 `rank.py` re-derive rule features by streaming the 465 MB JSONL each run (how
iteration was done; embeddings always reused). v1/v4/v5 use the artifact-consuming pattern
the gate grades.

† **Why v7 is slower than v5 at rank time and still the shipped version.** v4/v5 loaded a
precomputed `features.parquet` (hence ~2–3.6 s). v7 trades that for the JD-seam
architecture: `rank.py` **rebuilds the JD-dependent features live** from pooled,
JD-independent artifacts (38.3 s of the 38.8 s, parallel across workers), so retargeting
to a new JD is a config edit with no candidate re-embed. Still **12.9%** of the 5-minute
budget. v7's Stage-3 command:
`python rank.py --candidates ./candidates.jsonl --out ./submission.csv`.

## GPU/CPU split (v5-final)
- **GPU only in precompute** (`precompute.py` orchestrator): bi-encoder embeddings +
  cross-encoder teacher print their device at runtime; with shipped `artifacts_full\`
  artifacts, precompute is skippable (`--skip-embeddings`) — no GPU ever needed again
  unless new candidates are added.
- **Rank step is CPU-only by construction**: `rank.py` imports only
  numpy/pandas/pyarrow/orjson/psutil/lightgbm; torch is not in the [rank step]
  requirements and **not installed in the rank venv at all** — GPU use is impossible,
  not merely disabled. Measured: 3.6 s, 0.34 GB, 16.5 MB artifacts.

## Pre-computation time: with vs without GPU (measured on this machine)

| Stage | CPU only (measured/est.) | With RTX 4050 GPU | Notes |
|---|---|---|---|
| Bi-encoder embeddings (400K chunks) | **117 min (measured)** | ~4 min (est., 25-30x typical) | bge-small-en-v1.5; auto-uses CUDA if torch sees it |
| Cross-encoder teacher (12K pairs) | ~3-4 h (est. @ ~1-2 pairs/s) | **8.4 min (measured, 41.6 pairs/s)** | bge-reranker-v2-m3 fp16, 1.43 GB VRAM |
| Rule features + student training + final features | ~6 min (measured, CPU) | same (CPU-bound) | no benefit from GPU |
| **Total precompute** | **~5-6 h** | **~20 min** | one-time; skippable entirely with shipped artifacts (--skip-embeddings) |

GPU is an accelerator here, never a requirement: the full chain runs CPU-only if needed,
and with the shipped artifacts_full/ it does not run at all unless candidates change.
