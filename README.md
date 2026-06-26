# redrob_ranker — v7 "JD-seam" package

The v6 ranker, re-cut so that **retargeting to a new job description is a config
edit, not a code edit.** All per-JD knowledge lives in
[`jd/jd_profile.yaml`](jd/jd_profile.yaml); the JD-stable scoring mechanism (all
numerics) lives in [`jd/method_config.yaml`](jd/method_config.yaml). The pure
core (`redrob_ranker/`) reads those two seams and the pipeline scripts
orchestrate the lifecycle.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the module map / pinned interfaces
and [`jd/RETARGETING.md`](jd/RETARGETING.md) for the "fill these 6 fields from a
new JD" guide.

## Lifecycle (run in this order)

| # | Command | Cadence / hardware |
|---|---------|--------------------|
| 1 | `python embed_candidates.py --candidates ./candidates.jsonl` | per candidate POOL — **GPU**, minutes |
| 2 | `python jd_compile.py` | per JD — CPU, seconds (re-embeds only the signal queries) |
| 3 | `python rank.py --candidates ./candidates.jsonl --features-only` | builds `artifacts_v7/features_v7.parquet` for training — CPU |
| 4 | `python train.py` | per JD (optional) — **GPU**, minutes (CE teacher -> LambdaMART student) |
| 5 | `python rank.py --candidates ./candidates.jsonl --out ./submission.csv` | every rank — **CPU only, < 5 min** |
| 6 | `python validate_submission.py ./submission.csv` | check the output shape |

Steps 1 only re-run when the candidate pool changes; steps 2–4 only when the JD
changes (edit `jd_profile.yaml` first); step 5 is the constrained submission
step. The single Stage-3 command is unchanged from v6:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Validate the JD seam without running anything:

```bash
python -m redrob_ranker.profile --check jd/jd_profile.yaml
```

## CPU / GPU split (by construction)

- **rank step** (`rank.py` + `redrob_ranker/`): imports only
  numpy/pandas/pyarrow/orjson/psutil/lightgbm/pyyaml — **torch and
  sentence-transformers are never imported**, so the constrained step cannot
  reach a GPU. Install [`requirements-rank.txt`](requirements-rank.txt).
- **precompute steps** (`embed_candidates.py`, `jd_compile.py`, `train.py`):
  add torch / sentence-transformers / transformers. Install
  [`requirements-precompute.txt`](requirements-precompute.txt). These produce
  the artifacts in `artifacts_v7/`; the rank step only reads them.

## v5 parity + the BM25 lexical channel

The deterministic composite, integrity ladder, availability/notice/location
gates and the LightGBM blend are numerically faithful to v5/v6 — every constant
now comes from `jd_profile.yaml` / `method_config.yaml` (where the v7 split and
the old single yaml disagree, the v7 yaml wins). The **BM25 lexical channel** is
restored and faithful to root `rank.py`: `jd_compile.py` now also emits
`bm25_facets.parquet` (one min-max-normalized `<id>__bm25` column per signal,
same tokenizer and per-candidate evidence doc as root), `features.py` loads it,
and `rules.py` mixes `lex_fit = mean(<id>__bm25)` into the additive composite.
Keeping rank_bm25 in the per-JD step means the CPU rank path never imports it.

## Artifacts (`artifacts_v7/`)

JD-independent (survive a JD change): `job_embeddings.npy`,
`summary_embeddings.npy`, `job_offsets.npy`, `evidence_texts.parquet`,
`intrinsic.parquet`. JD-compiled: `jd_vectors.npy`, `bm25_facets.parquet` (the
lexical channel, `<id>__bm25` per signal), `jd_profile.yaml` (copy).
Trained: `model.txt`, `feature_cols.json`. The root is
`${RANKER_ROOT:-<repo root>}/artifacts_v7/`.
