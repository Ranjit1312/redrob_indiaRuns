# v2 — Evidence layer: ownership/context/recency regexes + integrity ladder

v1 plus a career-text evidence layer and a continuous integrity score —
everything the Round 1-2 audits demanded, base-rate verified.

## The rules this version applies (on top of v1's facet composite)

1. **Career-text evidence regexes** — 4 must-have categories (retrieval,
   vectordb, rankeval, ltr_recsys) scored per job as
   `hit x ownership x context x recency`, **max-pooled over jobs**:
   - ownership verbs (led/owned/designed/architected/from scratch) x1.0 vs
     participation-only x0.7;
   - "internal knowledge base / internal dashboard" context x0.4 (the template trap);
   - recency decay (halflife 30 months);
   - **summaries are excluded** — evidence comes from career descriptions only;
   - ranking-eval credit requires real ranking metrics (NDCG/MRR/MAP/recall@k)
     or offline-online / A-B language; BLEU/ROUGE chatbot eval no longer qualifies.
   `fit = 0.40*dense + 0.10*lex + 0.42*evid_coverage + 0.08*depth_bonus`.
2. **Continuous integrity ladder replacing the binary honeypot gate** — hard
   contradictions (career-sum >> stated YoE, single role > career, stated YoE >>
   history with the threshold tightened to catch a 15.2yr/86mo honeypot that
   slipped a 1.9x bound, summary-stated-years cross-check, expert-with-0-months,
   >=8 experts) each multiply integrity x0.05; pervasive synthetic noise
   (skill durations > career, inverted salary range) gets soft x0.85-0.97.
   `final = minmax(fit) * availability * notice_pen * integrity`.
3. **Job-hopper damp x0.55** — 4+ jobs with mean stint < 19 months (the JD's
   "switches companies every 1.5 years" red flag).
4. **JD-city location model** — Pune/Noida 1.0; Hyderabad/Mumbai/Delhi-NCR 0.9;
   rest of India 0.75 if willing to relocate else 0.40; outside India 0.50/0.12
   with an extra x0.60 damp for abroad + no-relocate (no visa sponsorship).
   `fit *= 0.55 + 0.45*loc_fit2`.
5. **Notice-period penalty** — <=90d x1.0, <=120d x0.93, >120d x0.88.
6. **CV-primary damp x0.60** — predominantly computer-vision/speech/robotics
   careers without NLP/IR (a JD disqualifier).

## Why (from the audits — see ../audit_report.md, Rounds 1-2)

- The v1 top-30 audit found **13/30 WEAK or RED-FLAG**: cosine similarity alone
  could not separate churn-model/dashboard profiles from genuine retrieval engineers.
- The known **plain-language Tier-5 sat at rank 91**, punished by v1's own
  anti-stuffer penalty (no skill tags to corroborate) — evidence coverage now
  feeds the corroboration max so plain-language proof counts.
- A **CV-primary disqualifier sat at rank 10** -> the CV-primary damp.
- The rank-10 case claimed FAISS **only in its boilerplate summary** -> summaries
  excluded from evidence scoring.
- A **honeypot at rank 9** (15.2yr stated vs 86mo history) escaped the 1.9x
  yoe bound -> integrity threshold tightened + summary cross-check.
- Result after v2: 1 WEAK left in top-30; that honeypot fell to rank ~99,952.

## How to run

One-time precompute (unconstrained, may use GPU):

```
python run_step35.py --candidates ..\..\candidates.jsonl --out-dir ..\..\artifacts_full
```

Constrained rank step (CPU only — re-scans the JSONL for evidence regexes,
reuses saved embeddings; writes `submission.csv` and `top100candidates.jsonl`
in THIS directory):

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

Wall / CPU / peak-RAM figures are printed at the end of every run and recorded
in `telemetry.json` / `RESULTS.md` after the run, along with the 14 tracked
probe-candidate ranks used to catch regressions.
