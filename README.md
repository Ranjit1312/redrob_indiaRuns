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

## 4. Scoring methodology & rationale

The score is a **deterministic, auditable composite** (every constant lives in
[`jd/method_config.yaml`](jd/method_config.yaml) / [`jd/jd_profile.yaml`](jd/jd_profile.yaml)),
blended with a distilled LightGBM student. The design answers the JD's actual ask —
*has this person **shipped** retrieval/ranking, hands-on, at a product company* —
and is built to resist the dataset's keyword-stuffers and honeypots.

**The formula (per candidate).**

```
dense_fit = 0.28·ranking + 0.22·retrieval + 0.12·vectordb + 0.10·evaluation
          + 0.10·applied_ml + 0.08·yoe_fit + 0.10·domain_nlp_ratio   (recency-weighted facet cosines)
lex_fit   = mean(per-facet BM25)                                     (exact terms embeddings smear: NDCG, Qdrant…)
fit       = 0.38·dense_fit + 0.09·lex_fit + 0.08·depth_bonus         (ownership/scale depth)

fit ×= (0.15 + 0.85·evidence_coverage)          # (1) evidence GATE — no shipped evidence really costs
fit ×= (0.50 + 0.50·skill_corroboration)        # (2) keyword-stuffer DISCOUNT (only if AI skills are claimed)
fit ×= (1 + 0.25·assess_strength·min(1, cov/0.25))  # (3) assessment BONUS — upside-only, evidence-gated
fit ×= recency_ladder(months_since_IC_role)     # (4) 0.35 / 0.70 / 0.90 — "this role writes code"
fit ×= (0.60 + 0.40·yoe_fit)                     # (5) experience band (Gaussian around the JD's 5–9 yrs)
fit ×= cv_primary(0.60) · hopper(0.55) · consulting(≤0.30) · location(0.55+0.45·loc2)   # red-flag damps

final = minmax( 0.8·minmax(fit) + 0.2·minmax(student) ) · integrity · availability · notice
```

(`student` = LightGBM LambdaMART distilled from a `bge-reranker-v2-m3` cross-encoder
teacher; α=0.2, so the deterministic rules dominate 80/20.)

**Why it's built this way — three principles:**

1. **Applied experience is the ground truth, not skill tags.** Relevance is earned
   in the **career history**: each must-have signal is matched by regex over job
   descriptions, weighted by **ownership** (led/owned/designed ×1.0 vs participation
   ×0.7), **context** (the "internal dashboard/KB" template trap ×0.4), and
   **recency** (≈30-month half-life), then max-pooled across roles. *Summaries are
   excluded* (self-promotion isn't evidence). The dense bi-encoder facets and BM25
   lexical channel corroborate; the heavy weight sits on demonstrated, shipped work.

2. **Self-reported skills can't buy rank (anti-keyword-stuffer).** A candidate's
   `skills` array — proficiency, `duration_months`, endorsements — **never adds
   positive score on its own.** Claimed AI skills only count when **corroborated by
   the career text** (gate (2): an uncorroborated stuffer is damped toward ×0.5,
   never lifted above 1.0). The skills array's *internal impossibilities* (durations
   exceeding the career, etc.) feed only the **soft** side of the integrity ladder —
   because in this synthetic pool those values are pervasive noise, so we damp them
   (×0.85–0.97), we do **not** trust them as ground truth or hard-gate on them.

3. **`skill_assessment_scores` is the one trusted out-of-experience skill signal.**
   The platform's proctored assessments are the single skill channel a stuffer
   cannot fake, so they're the *only* skill data that can lift a score — and even
   then conservatively: mean of the top-3 JD-relevant scores mapped `(s−40)/50`
   (a 50 isn't validation; a 90 is), **confidence-discounted by test count**
   `×n/(n+0.5)`, and **evidence-gated** (`min(1, coverage/0.25)` — full credit only
   once the career narrative already supports it). Upside-only: having no
   assessments costs nothing.

**Integrity & behavioral gates sit *outside* the learned blend** (`× integrity ×
availability × notice`) so no learned score can override a disqualifier: the
integrity ladder hard-fails (×0.05) egregious contradictions (career-sum ≫ stated
YoE, single role > career, YoE ≫ history, expert-with-0-months, ≥8 "expert" skills)
and soft-damps synthetic noise; availability rewards the JD's behavioral mandate
(active, responsive, open-to-work); notice-period and location/relocation apply the
JD's hard preferences. Full prose: `submission_metadata.yaml` (`methodology_summary`)
and [`ARCHITECTURE.md`](ARCHITECTURE.md).

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
