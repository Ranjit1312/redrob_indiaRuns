# RedRob Candidate Ranker — v7 ("JD-seam")

Ranks the top-100 candidates from `candidates.jsonl` for the released job
description and writes a spec-compliant `submission.csv` (`candidate_id, rank,
score, reasoning`). The constrained ranking step runs **CPU-only, offline, in
~39 seconds** against the full 100K pool — well inside the 5-minute / 16 GB
budget.

> **Design in one line:** retargeting to a *new* JD is a **config edit, not a
> code edit.** All per-JD knowledge lives in [`jd/jd_profile.yaml`](jd/jd_profile.yaml);
> the JD-stable scoring mechanism (every numeric constant) lives in
> [`jd/method_config.yaml`](jd/method_config.yaml); the pure core
> ([`redrob_ranker/`](redrob_ranker)) reads those two seams. See
> [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`jd/RETARGETING.md`](jd/RETARGETING.md).

---

## 1. The single reproduce command (Stage 3)

The constrained step that produces the CSV from the candidates file:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

CPU-only, no network, no GPU. It reads the precomputed artifacts in
`artifacts_v7/` and never imports torch (so a GPU is unreachable by
construction). Validate the output shape:

```bash
python validate_submission.py --submission ./submission.csv
```

### Setup (rank step)

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-rank.txt              # numpy/pandas/pyarrow/lightgbm/pyyaml — no torch
```

---

## 2. Pre-computation (offline, GPU optional, unbudgeted)

Per the spec, pre-computation may exceed the 5-minute window; only the rank step
above is budgeted. **This repo ships every artifact the rank step needs except
the two large embedding matrices** (`job_embeddings.npy` 440 MB,
`summary_embeddings.npy` 147 MB — above GitHub's file limit). Regenerate them
(and refresh all pool artifacts) with the precompute pipeline before ranking
from a fresh clone:

```bash
pip install -r requirements-precompute.txt        # adds torch / sentence-transformers / transformers
python embed_candidates.py --candidates ./candidates.jsonl   # -> embeddings, offsets, evidence, intrinsic
python jd_compile.py                                         # -> jd_vectors, bm25_facets (JD seam)
python rank.py --candidates ./candidates.jsonl --features-only   # -> features_v7.parquet (for training)
python train.py                                             # cross-encoder teacher -> LightGBM student
```

| Step | Cadence | Hardware |
|---|---|---|
| `embed_candidates.py` | per candidate **pool** | GPU (CPU works), minutes |
| `jd_compile.py` | per **JD** | CPU, seconds |
| `rank.py --features-only` | per JD (training input) | CPU |
| `train.py` | per JD (optional) | GPU, minutes |
| **`rank.py --out …`** | **every rank — the judged step** | **CPU only, < 5 min** |

A reproducible containerized path for all of the above is in
[`DOCKER_RUNBOOK.md`](DOCKER_RUNBOOK.md) (one `Dockerfile`, modes
`RANK` / `PRECOMPUTE` / `SERVE` via `--build-arg ENV_MODE`).

---

## 3. Compute budget & telemetry (measured)

**Constrained rank step** — Docker, `--network none --memory 16g --cpus 8`, the
full 100K pool (`rank.docker.log`):

| Metric | Measured | Limit | Used |
|---|---|---|---|
| Wall clock | **38.8 s** | 300 s | **12.9 %** |
| Peak RAM | **0.31 GB** | 16 GB | **2 %** |
| Compute | CPU only (torch not installed) | CPU only | ✓ |
| Network | off | off | ✓ |
| Artifacts read | 627 MB | 5 GB disk | ✓ |

Stage breakdown: `build_features` 38.3 s (parallel across 12 workers) dominates;
`compute_rules` 0.05 s, `lgbm_predict_blend` 0.35 s, `argpartition_topk` 0.01 s,
`reasoning_write_csv` 0.10 s. The feature build shards candidates across
processes (`RANK_WORKERS`, default `os.cpu_count()`); output is bitwise-identical
to the serial build. `validate_submission.py` reports **PASS on all 11 checks**
(exactly 100 rows, ranks 1..100 unique, monotonic scores, no empty reasoning, …).

**Pre-compute** — GPU (`device=cuda:0`), offline (`precompute.log`):
`embed_candidates` 686 s (100K candidates → 300,171 chunks), feature pass 98.5 s,
`train` 480 s (cross-encoder teacher 428 s + LightGBM student 50 s) ≈ **21 min**.

**Model quality** — held-out split, student vs. cross-encoder teacher
(`artifacts_v7/train_eval.json`): NDCG@10 **0.447**, NDCG@50 **0.703**, Spearman
**0.740** (best iteration 16). These measure the student's fidelity to the
teacher used for distillation, not the hidden ground truth.

---

## 4. How it works (short)

Two-phase system. **Offline:** the JD is split into six facet queries
(`ranking, retrieval, vectordb, evaluation, applied_ml, llm_ft`), embedded with a
bi-encoder (`bge-small-en-v1.5`) and scored against every recency-weighted career
chunk; combined with per-facet **BM25** (the restored lexical channel,
`<id>__bm25`) and structured fit features; a JD-rule evidence layer credits
must-have categories with ownership/context/recency weighting; a continuous
**integrity** score (not a binary honeypot gate) damps synthetic/contradictory
profiles. A cross-encoder teacher (`bge-reranker-v2-m3`) labels a shortlist +
random negatives; labels are distilled into a tuned **LightGBM LambdaMART**
student. **Online (rank step):** load ~627 MB of artifacts, predict the student,
recompute the deterministic rules composite in numpy, blend (rules-dominant),
then apply integrity / availability / notice-period gates **outside** the blend
so no learned score can override a disqualifier. Honeypots are avoided by reading
profiles, not special-cased. Full prose: `submission_metadata.yaml`
(`methodology_summary`) and [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 5. Sandbox

A hosted demo (HuggingFace Space) ranks a small pre-loaded candidate pool
end-to-end on CPU and returns the top-100 CSV + live telemetry — exactly as the
judged container runs `rank.py`. Link: see `sandbox_link` in
[`submission_metadata.yaml`](submission_metadata.yaml). Local equivalent:
`docker build -t redrob:serve --build-arg ENV_MODE=SERVE . && docker run -p 7860:7860 redrob:serve`.

---

## 6. Repository layout

```
rank.py                      the judged step (CPU, artifact-only)
embed_candidates.py          precompute: embeddings + evidence + intrinsic (GPU)
jd_compile.py                precompute: JD vectors + BM25 facets (CPU)
train.py                     precompute: CE teacher -> LightGBM student (GPU)
validate_submission.py       Stage-1 format checks
redrob_ranker/               pure core: profile, features, rules, intrinsic, bm25
jd/                          jd_profile.yaml + method_config.yaml (the two seams) + schema
artifacts_v7/                precomputed artifacts (large embeddings regenerated by precompute)
app.py                       Gradio sandbox (SERVE mode)
Dockerfile, entrypoint.sh    multi-mode container (RANK/PRECOMPUTE/SERVE)
requirements-rank.txt        rank-step deps (CPU, no torch)
requirements-precompute.txt  precompute deps (torch / sentence-transformers)
submission_metadata.yaml     portal metadata (team, contacts, methodology)
ARCHITECTURE.md, DOCKER_RUNBOOK.md, jd/RETARGETING.md
```

## 7. AI tools

Built with **Claude** (Claude Code) as a development assistant; all architecture,
scoring design, and engineering decisions are the team's own. Declared in
`submission_metadata.yaml`.
```
