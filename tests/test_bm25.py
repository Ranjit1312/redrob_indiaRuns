"""
test_bm25.py — the BM25 lexical facet channel, on a tiny synthetic corpus.

No heavy artifacts / models. We load the real Profile, build ~5 short evidence
docs (one of which clearly matches a signal's query terms), and assert the
root-faithful contract: a column per signal id, each min-max normalized into
[0,1], and the obviously-matching doc scores highest for the relevant signal.

If rank_bm25 is not importable (it is precompute-only — see
requirements-precompute.txt), these tests skip. A second tiny test exercises
the features.py "bm25_facets.parquet missing" guard without fabricating the
heavy artifact set.
"""
import os

import numpy as np
import pytest

from redrob_ranker import profile as P
from redrob_ranker import bm25 as B

HERE = os.path.dirname(os.path.abspath(__file__))
JD_PATH = os.path.normpath(os.path.join(HERE, "..", "jd", "jd_profile.yaml"))
METHOD_PATH = os.path.normpath(os.path.join(HERE, "..", "jd", "method_config.yaml"))


def _has_rank_bm25() -> bool:
    try:
        import rank_bm25  # noqa: F401
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def profile():
    prof, _ = P.load(JD_PATH, METHOD_PATH)
    return prof


def test_tokenizer_matches_root():
    # root: _tok = re.compile(r"[a-z0-9\+\#\.]+"); lowercased findall
    assert B.tokenize("C++ and C# v2.0  FAISS") == ["c++", "and", "c#", "v2.0", "faiss"]
    assert B.tokenize("") == []
    assert B.tokenize(None) == []


def test_evidence_doc_joins_summary_and_career():
    # SEP-joined chunks collapse to spaces; summary prefixes the body
    doc = B.evidence_doc("headline here", "job a\x1fjob b")
    assert doc == "headline here job a job b"


@pytest.mark.skipif(not _has_rank_bm25(), reason="rank_bm25 not installed (precompute-only dep)")
def test_bm25_facet_scores_contract(profile):
    sids = profile.signal_ids()
    assert "ranking" in sids  # sanity: the JD's first signal

    # ~5 short docs; doc 0 is saturated with the `ranking` signal's query terms
    # (learning-to-rank / recommendation / search ranking system at scale).
    docs = [
        "built and shipped an end to end learning to rank recommendation "
        "system and search ranking to real users at scale in production",
        "frontend react engineer building dashboards and ui components",
        "data analyst writing sql reports and spreadsheets for finance",
        "devops kubernetes terraform aws cloud infrastructure pipelines",
        "mobile android kotlin app developer shipping consumer apps",
    ]
    out = B.bm25_facet_scores(docs, profile)

    # one column per signal id
    assert set(out.keys()) == set(sids)
    # each min-max normalized into [0,1]
    for sid in sids:
        s = np.asarray(out[sid], dtype=float)
        assert s.shape == (len(docs),)
        assert s.min() >= -1e-9
        assert s.max() <= 1.0 + 1e-9
    # the obviously-matching doc (index 0) tops the `ranking` facet
    ranking = np.asarray(out["ranking"], dtype=float)
    assert int(np.argmax(ranking)) == 0
    assert ranking[0] == pytest.approx(1.0)  # min-max => the max maps to 1.0


@pytest.mark.skipif(not _has_rank_bm25(), reason="rank_bm25 not installed (precompute-only dep)")
def test_bm25_lex_fit_reproduces_mean_over_facets(profile):
    # lex_fit (rules.py) = mean over ALL signals of <id>__bm25; replicate here.
    docs = [
        "learning to rank recommendation search ranking system in production",
        "vector database faiss pinecone hybrid search embeddings retrieval",
        "ndcg mrr map offline online a/b test relevance evaluation framework",
        "generic backend services with no ml signal whatsoever here at all",
        "lora qlora peft llm fine tuning large language models prompt",
    ]
    out = B.bm25_facet_scores(docs, profile)
    cols = np.column_stack([np.asarray(out[s.id], float) for s in profile.signals])
    lex_fit = cols.mean(axis=1)
    assert lex_fit.shape == (len(docs),)
    assert np.all(np.isfinite(lex_fit))


def test_features_required_path_missing_bm25(tmp_path, profile):
    # features.build_features must raise a clear error if the heavy artifacts
    # are absent — skip cleanly rather than fabricate them. We only assert that
    # an empty art_dir raises (FileNotFoundError for a missing artifact), which
    # exercises the "artifacts absent" path without heavy fixtures.
    from redrob_ranker import features as F
    _, method = P.load(JD_PATH, METHOD_PATH)
    with pytest.raises((FileNotFoundError, ValueError)):
        F.build_features(profile, method, art_dir=str(tmp_path))
