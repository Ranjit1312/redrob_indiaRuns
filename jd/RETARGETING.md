# Retargeting the ranker to a new job description

The whole point of the **JD seam** is that swapping the role is a *config* edit,
not a *code* edit. Everything the ranker knows about a JD lives in
[`jd_profile.yaml`](jd_profile.yaml); the scoring mechanism lives in
[`method_config.yaml`](method_config.yaml) and you usually leave it alone.

```
new JD (prose)  ──fill──►  jd_profile.yaml  ──validate──►  jd_compile  ──►  same rank.py  ──►  new ranking
```

## The 6 fields you extract from any JD

Read the JD once and fill these. Each maps to a block in `jd_profile.yaml`.

| From the JD, find…                                            | Goes to                          |
|---------------------------------------------------------------|----------------------------------|
| 1. Role title, company, domain, ideal years, notice tolerance | `role`                           |
| 2. Where the role is / who can relocate / remote-ok           | `locations`                      |
| 3. The **must-have capabilities** (the "you absolutely need") | `signals[]`                      |
| 4. What makes someone in- vs out-of-domain                    | `domain`                         |
| 5. Which platform skills/assessments are relevant             | `relevant_skill_regex`           |
| 6. The explicit **"we do NOT want"** list                     | `red_flags`                      |

Plus one derived line: a positive-only paraphrase of the ideal hire →
`cross_encoder_query`.

## Writing a `signals[]` entry (the heart of it)

Each must-have capability becomes one signal with three parts:

```yaml
- id: retrieval                      # stable, lowercase, becomes a column prefix
  label: "embeddings-based retrieval in production"   # used in candidate reasoning
  query: >-                          # a short dense query — what good looks like
    production embeddings based retrieval semantic search deployed to real users
  evidence_regex: 'semantic search|\bfaiss\b|sentence[- ]transformers?|...'  # or null
  dense_weight: 0.22                 # how much this axis matters (0.0 = model-only)
```

- **`query`** is embedded once by `jd_compile`; it is *what you're looking for*,
  phrased positively. Keep it a phrase, not a sentence.
- **`evidence_regex`** is matched live over each candidate's career history. Use
  it for capabilities that leave a textual fingerprint (named tools, techniques).
  Set it to `null` for soft axes (e.g. "applied ML") that only the learned
  student should weigh.
- **`dense_weight`** is this axis's share of the hand-weighted `dense_fit`. The
  weights across signals need not sum to 1 (they're combined, then normalised).
  Set `0.0` to keep an axis as a model-only feature (e.g. a nice-to-have).

## Writing `red_flags`

The JD's "things we explicitly do NOT want" become toggles. The *magnitude* of
each penalty is fixed in `method_config` (audited); the *decision to apply it*
to this role is the JD's:

```yaml
red_flags:
  cv_primary:      {enabled: true}    # this role rejects CV/speech-primary careers
  only_consulting: {enabled: true}
  job_hopper:      {enabled: true}
  stale_ic_role:   {enabled: true}
```

To turn a flag off for a role that doesn't care, set `enabled: false`. To add a
genuinely new kind of red flag you do need a code change (a new gate) — but the
four above cover the common recruiting anti-patterns.

## Validate before you run

`jd_profile.yaml` is checked against [`jd_profile.schema.json`](jd_profile.schema.json)
at load time, and `redrob_ranker.profile.load()` raises a clear error if a
required field is missing or mistyped. To check without running the pipeline:

```bash
python -m redrob_ranker.profile --check jd/jd_profile.yaml
```

## What you do NOT touch when retargeting

`method_config.yaml`: recency half-lives, the integrity ladder, availability and
notice penalty shapes, the blend `alpha`, generic recruiting lexicons, the
location scoring ladder, world facts, model ids. These are properties of the
*algorithm*, audited across v1→v5. Change them only when you deliberately
re-tune the method (and version that change).

## Then: compile and rank (unchanged across JDs)

```bash
python -m redrob_ranker.jd_compile                          # re-embed the 6 queries (CPU, seconds)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv   # the constrained step
```

The expensive candidate-side embedding pass is **never** repeated for a JD
change — only the handful of JD queries are re-embedded. See
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) for the full lifecycle.
