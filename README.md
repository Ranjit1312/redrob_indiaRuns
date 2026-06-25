# v1 — Baseline: bi-encoder + BM25 + structured composite

The first complete ranker: 6-facet dense similarity + BM25 hybrid, structured
fit features, a transparent hand-set composite, and binary gates. No evidence
regexes, no assessments, no ML.

## The rules this version applies

1. **6-facet bi-encoder composite** — the JD is split into 6 facet queries
   (retrieval, vectordb, ranking, evaluation, applied_ml, llm_ft) embedded with
   bge-small-en-v1.5; each candidate's job chunks are scored by cosine to each
   facet and pooled (recency-weighted). `dense_fit = 0.28*ranking + 0.22*retrieval
   + 0.12*vectordb + 0.10*evaluation + 0.10*applied_ml + 0.08*yoe_fit +
   0.10*domain_nlp_ratio`.
2. **BM25 hybrid** — per-facet BM25 lexical scores, averaged;
   `fit = 0.8*dense + 0.2*lexical` (exact terms like NDCG/Qdrant that embeddings smear).
3. **Structured damps** (soft multipliers on fit):
   - anti keyword-stuffer: `fit *= 0.4 + 0.6*ai_skill_corroboration` (skills tags
     only count when corroborated by career text);
   - services-only career: `fit *= 1 - 0.30*only_consulting`;
   - no recent hands-on IC role (>18 months): `fit *= 0.85`;
   - location: `fit *= 0.70 + 0.30*location_fit` (Pune/Noida/T1/relocate).
4. **Binary honeypot gate** — `final = minmax(fit) * (1 - honeypot_flag)`:
   profiles failing internal-consistency checks are zeroed outright.
5. **Availability multiplier** — `final *= availability_mult` from login recency,
   response rate, open-to-work, profile completeness (the JD's behavioral mandate).

Top-100 via exact `np.argpartition`, ties broken by candidate_id ascending;
scores forced monotone non-increasing; reasoning templated from real profile
fields (title, YoE, top-2 facet strengths, honest concerns).

Known outcome (see `../audit_report.md`, Round 1): the top-30 audit found
13/30 WEAK or RED-FLAG — this version exists as the measured starting point
the later versions fix.

## How to run

One-time precompute (unconstrained, may use GPU; ~2h CPU / ~4 min GPU embed):

```
python run_step35.py --candidates ..\..\candidates.jsonl --out-dir ..\..\artifacts_full
```

Constrained rank step (CPU only, seconds — writes `submission.csv` and
`telemetry.json` in THIS directory):

```
python rank.py
```

Validate before submitting:

```
python validate_submission.py --submission submission.csv
# optionally also: --candidates ..\..\candidates.jsonl
```

If the repo root is elsewhere, set `RANKER_ROOT` to the directory containing
`candidates.jsonl` and `artifacts_full\`.

## Telemetry

Per-stage wall / CPU / RSS / peak-RAM telemetry is printed on every run and
recorded in `telemetry.json` (and summarized in `RESULTS.md` after the run) as
proof of compute-budget compliance (<=300 s, <=16 GB).
