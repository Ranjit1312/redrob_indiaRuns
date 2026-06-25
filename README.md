# v4 — Teacher-student distillation + full-profile-audit rules (final)

v3 plus a GPU cross-encoder teacher distilled into a tuned LightGBM LambdaMART
student blended at alpha=0.2, and the base-rate-verified rules from the
full-profile audit. This is the submitted configuration (`rank.py`).

## The rules this version applies (on top of v3)

1. **Cross-encoder teacher: bge-reranker-v2-m3** — chosen with **8K context
   after measuring 27.3% truncation at 512 tokens** with the default
   bge-reranker-base (evidence p99 ~641 tokens, max 861 — truncation cut exactly
   the early-career pre-LLM ML the JD asks about). Positive-evidence-only JD
   query (a relevance CE can't follow "NOT a researcher"). Bias-validated on
   corpus templates and probes before trusting: the plain-language Tier-5
   (0.691) outranked the keyword-claimer (0.671).
2. **Gated pseudo-labels** — `pseudo_label = sigmoid(CE logit) x anti-stuffer
   corroboration gate x integrity`, so residual CE bias cannot propagate into
   the student. 8K shortlist + 4K random negatives.
3. **Tuned LightGBM LambdaMART student** — lambdarank over 16 qcut label bins
   (`label_gain = 2^i - 1`), ~500-doc pseudo-groups, **stratified** 20% holdout
   and 4-fold CV, **coordinate descent** from the proven incumbent (21 configs;
   winning moves were more regularization: lr 0.08, min-leaf 80,
   feature-fraction 0.6). Holdout NDCG@10 0.6613 vs 0.6151 untuned.
4. **Blend at alpha=0.2** — `final = minmax(0.2*LGBM + 0.8*rules-fit) x integrity
   x availability x notice`; the alpha-sweep peaked at 0.2 (0.9277 vs 0.8992
   rules-only, +2.9 pts) and 0.2 is the conservative pick.
5. **Full-profile-audit rules** (each base-rate-verified before encoding):
   - dormant (inactive > 6 months AND recruiter response < 0.2): availability
     **x0.5** — the JD's own worked exclusion example (3.7% of pool);
   - low-response-alone (< 0.2, not dormant): **x0.8**;
   - certification anachronism (LangChain dated < 2022 / LLaMA < 2023):
     integrity **x0.30** (45 in pool — rare authenticity fingerprint);
   - activity-before-signup: integrity **x0.97** soft (7.5% of pool = generator noise);
   - concurrent same-window degrees: integrity **x0.93** soft (0.73%);
   - remote-pref location caps: remote+no-reloc+non-target-city capped at 0.25,
     any remote-pref x0.9 (JD is hybrid Tue/Thu); India non-target no-reloc 0.40 -> 0.33;
   - notice tiers **1.0 / 0.93 / 0.90 / 0.85** (<=90d / <120d / =120d / >120d);
   - **evidence-coverage tie-break** before candidate_id in the top-100 sort.

## Why (from the audits — see ../audit_report.md, Rounds 4-5)

- The **full-profile audit (complete JD + every field) returned verdict MAJOR**:
  the JD's literal dormant-and-unresponsive exclusion example sat at rank 17;
  LangChain-2018 cert anachronisms, education impossibilities, and
  signup-vs-activity contradictions were invisible to the field-subset audits.
- **Base-rate verification demoted 3 of the auditor's 5 proposed removals to
  soft penalties**: activity-before-signup hits 7,496 candidates,
  career-before-education 19,499, concurrent degrees 728 — pervasive synthetic
  noise, none can be among ~80 honeypots. The rare fingerprints (45 anachronisms,
  3.7% dormant) became strong rules.
- The final top-50 verification was MINOR only: the reasoning label
  "ranking-evaluation rigor (NDCG/MRR/A-B)" was softened to
  "ranking-evaluation work" for 2 candidates whose profiles don't name those
  metrics, and the coverage tie-break was added. Final state: zero
  disqualifiers, zero hallucinations, zero impossible/dormant profiles in the
  top-100.

## How to run

One-time precompute (unconstrained; teacher needs a GPU, ~8.4 min on RTX 4050):

```
python run_step35.py --candidates ..\..\candidates.jsonl --out-dir ..\..\artifacts_full
python precompute_teacher.py
python train_student.py
```

(`rank.py` from versions\v3 must have produced
`features_refined_v3.parquet` / `scores_step35_v4.parquet`, and this
version's `rescore.py` produces `features_v4.parquet`, before the first
`rank.py` run here.)

Constrained rank step (CPU only, ~2 s — writes `submission.csv` and
`rank_telemetry.json` in THIS directory):

```
python rank.py
```

Validate before submitting:

```
python validate_submission.py --submission submission.csv
```

If the repo root is elsewhere, set `RANKER_ROOT` to the directory containing
`candidates.jsonl` and `artifacts_full\`.

## Telemetry

Per-stage wall / CPU / RSS / peak-RAM and artifact disk footprint are printed
on every run and recorded in `rank_telemetry.json` (results summarized in
`telemetry.json` / `RESULTS.md` after the run): reference run = 2.0 s wall
(0.7% of budget), 0.40 GB peak RAM (2.5%), 16.5 MB artifacts (0.3% of 5 GB).
