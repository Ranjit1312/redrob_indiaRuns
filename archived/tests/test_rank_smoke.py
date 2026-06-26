"""
Smoke + constraint tests for the ranking step itself.

  * the rank step must be CPU-only BY CONSTRUCTION — it must not import torch or
    sentence-transformers (spec §3: CPU only, no network);
  * rank.py must run end-to-end and emit a spec-valid submission.csv.
"""
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORBIDDEN = ("torch", "sentence_transformers", "transformers",
             "openai", "anthropic", "cohere", "google.generativeai", "requests", "httpx")


def test_rank_imports_are_cpu_only_and_offline():
    """Static guard: rank.py must not pull in any GPU/network/LLM library."""
    src = open(os.path.join(ROOT, "rank.py"), encoding="utf-8").read()
    hits = []
    for mod in FORBIDDEN:
        pat = re.compile(rf"^\s*(?:import|from)\s+{re.escape(mod)}\b", re.MULTILINE)
        if pat.search(src):
            hits.append(mod)
    assert not hits, f"rank.py imports forbidden (GPU/network) modules: {hits}"


@pytest.mark.slow
def test_rank_runs_and_produces_valid_submission(tmp_path, candidates_path, artifacts_dir):
    """End-to-end: rank.py exits 0 and writes a 100-row, monotone, well-formed CSV."""
    out = tmp_path / "submission_smoke.csv"
    env = dict(os.environ, RANKER_ROOT=ROOT)
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "rank.py"),
         "--candidates", candidates_path, "--out", str(out)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"rank.py failed:\nSTDOUT{r.stdout[-2000:]}\nSTDERR{r.stderr[-2000:]}"
    assert out.is_file()

    import csv
    rows = list(csv.reader(open(out, encoding="utf-8", newline="")))
    assert rows[0] == ["candidate_id", "rank", "score", "reasoning"]
    data = rows[1:]
    assert len(data) == 100
    ranks = sorted(int(r[1]) for r in data)
    assert ranks == list(range(1, 101))
    scores = [float(r[2]) for r in sorted(data, key=lambda r: int(r[1]))]
    assert all(scores[i] >= scores[i + 1] - 1e-12 for i in range(len(scores) - 1))
