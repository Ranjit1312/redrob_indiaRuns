"""
Determinism: the spec's Stage-3 reproduction must equal the development run
("100/100 identical rows"). Running rank.py twice must yield byte-identical CSVs.
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(out_path, candidates):
    env = dict(os.environ, RANKER_ROOT=ROOT)
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "rank.py"),
         "--candidates", candidates, "--out", str(out_path)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    return out_path


@pytest.mark.slow
def test_two_runs_are_byte_identical(tmp_path, candidates_path, artifacts_dir):
    a = _run(tmp_path / "a.csv", candidates_path).read_bytes()
    b = _run(tmp_path / "b.csv", candidates_path).read_bytes()
    assert a == b, "rank.py is not deterministic — two runs differ"


@pytest.mark.slow
def test_matches_committed_submission(tmp_path, candidates_path, artifacts_dir, submission_path):
    """A fresh run reproduces the committed submission.csv exactly."""
    fresh = _run(tmp_path / "fresh.csv", candidates_path).read_bytes()
    committed = open(submission_path, "rb").read()
    assert fresh == committed, "fresh rank.py output differs from the committed submission.csv"
