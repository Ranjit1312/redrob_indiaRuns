"""
Stage-1 submission contract (submission_spec.docx §2-3). These must pass for
EVERY version — they encode the auto-rejection rules the portal applies.
"""
import re

import pytest

CAND_ID_RE = re.compile(r"^CAND_\d+$")
EXPECTED_HEADER = ["candidate_id", "rank", "score", "reasoning"]


def test_header_is_exact(submission_rows):
    assert submission_rows["header"] == EXPECTED_HEADER


def test_exactly_100_rows(submission_rows):
    assert len(submission_rows["rows"]) == 100


def test_ranks_are_1_to_100_each_once(submission_rows):
    ranks = sorted(r["rank"] for r in submission_rows["rows"])
    assert ranks == list(range(1, 101))


def test_candidate_ids_unique(submission_rows):
    ids = [r["candidate_id"] for r in submission_rows["rows"]]
    assert len(set(ids)) == len(ids)


def test_candidate_ids_well_formed(submission_rows):
    bad = [r["candidate_id"] for r in submission_rows["rows"]
           if not CAND_ID_RE.match(r["candidate_id"])]
    assert not bad, f"malformed ids: {bad[:5]}"


def test_scores_monotonically_non_increasing(submission_rows):
    by_rank = [r["score"] for r in sorted(submission_rows["rows"], key=lambda r: r["rank"])]
    bad = [(i + 1, by_rank[i], by_rank[i + 1])
           for i in range(len(by_rank) - 1) if by_rank[i] < by_rank[i + 1] - 1e-12]
    assert not bad, f"score increases with rank at {bad[:3]}"


def test_scores_not_all_identical(submission_rows):
    assert len({r["score"] for r in submission_rows["rows"]}) > 1


def test_reasoning_present(submission_rows):
    # optional per spec, but we always ship it (scored at Stage 4)
    empty = [r["rank"] for r in submission_rows["rows"] if not r["reasoning"].strip()]
    assert not empty, f"empty reasoning at ranks {empty[:5]}"


def test_all_ids_exist_in_candidates(submission_rows, candidates_path):
    ids = {r["candidate_id"] for r in submission_rows["rows"]}
    needles = {c: f'"{c}"'.encode() for c in ids}
    missing = set(ids)
    with open(candidates_path, "rb") as f:
        for line in f:
            for c in list(missing):
                if needles[c] in line:
                    missing.discard(c)
            if not missing:
                break
    assert not missing, f"ids not found in candidates.jsonl: {sorted(missing)[:5]}"
