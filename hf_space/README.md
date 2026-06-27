---
title: RedRob v7 Candidate Ranker
emoji: 🎯
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# RedRob v7 — Candidate Ranker (sandbox)

A live sandbox for the RedRob hackathon submission. Click **Rank candidates** to
run the constrained ranking step (`rank.py`) over a bundled ~150-candidate demo
pool and get the spec-compliant top-100 `submission.csv` — **CPU-only, offline,
in seconds**, exactly as the judged container runs it.

## What this is (and isn't)

The real submission ranks the full 100K pool. Producing the embeddings for that
pool is the **GPU precompute phase**, which a hosted Space can't run. So this
sandbox ranks a **small precomputed demo pool** (spec §10.5 asks for ≤100–200
candidates and small-sample reproducibility — not the full pool). The ranking
*mechanism*, the trained student (`model.txt`), and the JD seam are identical to
the full submission; only the candidate set is smaller.

- **Input:** bundled `pool.jsonl` (demo candidates) — no upload needed.
- **Compute:** CPU only, no network, well under the 5-minute / 16 GB budget.
- **Output:** ranked `submission.csv` (candidate_id, rank, score, reasoning),
  downloadable, plus the live telemetry (wall / peak RAM / budget headroom).

Full code, the precompute pipeline, and reproduction instructions are in the
GitHub repository linked from the submission.
