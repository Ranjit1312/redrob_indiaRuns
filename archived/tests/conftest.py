"""
Shared pytest fixtures for the RedRob ranker test-suite.

The tests run against whatever version is currently promoted to the repo root
(rank.py / submission.csv / artifacts_full / candidates.jsonl). Run them at each
version commit; set RANK_VERSION=<n> so the version-gated behavioural tests assert
the right expectations (e.g. the CV-primary disqualifier is only excluded from
v4 onward — that is the bug the audit loop fixed).

    set RANK_VERSION=5 && pytest -q            # Windows
    RANK_VERSION=5 pytest -q                   # bash

Tests skip gracefully when candidates.jsonl / artifacts_full are absent so they
do not hard-fail in a thin checkout.
"""
import csv
import os
import sys
from datetime import date

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reference "today" used by the pipeline's availability rules (build_final_features.py).
PIPELINE_REF_DATE = date(2026, 6, 11)

# Probe candidates documented in the audit loop (versions/audit_report.md).
CV_PRIMARY_DISQUALIFIER = "CAND_0039983"   # computer-vision career, JD disqualifier
PLAIN_LANGUAGE_TIER5 = "CAND_0000031"      # strong fit, zero buzzwords (must be rescued)


def rank_version():
    """Active version under test (default 5 = the submission)."""
    try:
        return int(os.environ.get("RANK_VERSION", "5"))
    except ValueError:
        return 5


def _path(*parts):
    return os.path.join(ROOT, *parts)


@pytest.fixture(scope="session")
def submission_path():
    p = _path("submission.csv")
    if not os.path.isfile(p):
        pytest.skip("submission.csv not present at repo root — run rank.py first")
    return p


@pytest.fixture(scope="session")
def candidates_path():
    p = _path("candidates.jsonl")
    if not os.path.isfile(p):
        pytest.skip("candidates.jsonl not present — organiser-provided input, not committed")
    return p


@pytest.fixture(scope="session")
def artifacts_dir():
    p = _path("artifacts_full")
    if not os.path.isdir(p) or not os.path.isfile(os.path.join(p, "features.parquet")):
        pytest.skip("artifacts_full/ not present — copy the rank artifacts in first")
    return p


@pytest.fixture(scope="session")
def submission_rows(submission_path):
    """Parsed submission as a list of dicts: candidate_id, rank(int), score(float), reasoning."""
    with open(submission_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    out = []
    for r in data:
        out.append({"candidate_id": r[0], "rank": int(r[1]),
                    "score": float(r[2]), "reasoning": r[3] if len(r) > 3 else ""})
    return {"header": header, "rows": out}


def _load_json_lines(path, wanted_ids):
    """Stream candidates.jsonl once; return {candidate_id: record} for ids in wanted_ids."""
    try:
        import orjson as _j
        loads = _j.loads
    except ImportError:
        import json as _j
        loads = _j.loads
    found = {}
    wanted = set(wanted_ids)
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            # cheap pre-filter before full parse
            if not any(w.encode() in line for w in wanted if w not in found):
                continue
            rec = loads(line)
            cid = rec.get("candidate_id")
            if cid in wanted:
                found[cid] = rec
                if len(found) == len(wanted):
                    break
    return found


@pytest.fixture(scope="session")
def top_records(submission_rows, candidates_path):
    """Full candidate records for the candidate_ids in the submission (top-100)."""
    ids = [r["candidate_id"] for r in submission_rows["rows"]]
    recs = _load_json_lines(candidates_path, ids)
    if len(recs) != len(ids):
        pytest.skip(f"only resolved {len(recs)}/{len(ids)} candidate records from candidates.jsonl")
    return recs
