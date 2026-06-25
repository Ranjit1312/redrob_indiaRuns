# v3 — Validated potential: proctored skill assessments, evidence-gated

v2 plus the Redrob skill-assessment signal — proctored test scores as
corroboration a keyword-stuffer cannot fake, gated so a test can never
substitute for career proof.

## The rules this version applies (on top of v2's evidence layer)

1. **JD-relevant assessment strength** — `skill_assessment_scores` keys matched
   against a JD-relevance regex (embeddings/vector/retrieval/ranking/NLP/LLM/
   Python/PyTorch/...; CV/speech excluded); mean of top-3 scores mapped via
   `(score - 40) / 50` clipped to [0,1] (a 50 is not validation; a 90 is),
   then **confidence-discounted by test count**: `strength *= n / (n + 0.5)`
   (1 test = 0.67x, 3 tests = 0.86x).
2. **Corroboration becomes a 3-way max** —
   `fit *= 0.4 + 0.6 * max(narrative corroboration, evidence coverage, assess_corr)`
   where `assess_corr = assess_strength * min(1, evid_coverage / 0.25)`:
   assessment credit scales with narrative evidence and reaches full credit only
   at coverage >= 0.25 — the evidence gate.
3. **+0.05 additive "validated potential" term** —
   `fit = 0.38*dense + 0.09*lex + 0.40*coverage + 0.08*depth + 0.05*assess_strength`;
   upside-only, absence of assessments costs nothing.
4. Everything else identical to v2 (evidence regexes, integrity ladder, hopper
   x0.55, CV-primary x0.60, location model, notice penalty).

## Why (from the audits — see ../audit_report.md, Round 3)

- The pool contains **skilled-but-not-yet-applied candidates** ("validated
  potential") whose career text undersells them; a proctored assessment is the
  one corroboration channel a stuffer cannot fake.
- But the first attempt let **ungated assessments lift coverage-0.15 profiles
  ~200 ranks** on a single test score — the diff-audit against v2 showed
  single-test jumpers entering the top-50. The fix: the `min(1, coverage/0.25)`
  evidence gate plus the `n/(n+0.5)` confidence discount.
- After gating, top-50 churn dropped from 8 to 3 entrants, all legitimate
  tie-breaks.

## How to run

One-time precompute (unconstrained, may use GPU):

```
python run_step35.py --candidates ..\..\candidates.jsonl --out-dir ..\..\artifacts_full
```

Note: the diff-audit section at the end of the rank step compares against v2's
saved scores (`artifacts_full\scores_step35_v3.parquet`), so run v2's
`rank.py` once first if that file does not exist yet.

Constrained rank step (CPU only — writes `submission.csv` and
`top100candidates.jsonl` in THIS directory):

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
in `telemetry.json` / `RESULTS.md` after the run, together with the v2 -> v3
diff-audit (top-50 enter/exit list) and the 14 tracked probe ranks.
