"""
bm25.py — the BM25 lexical facet channel (v7), numerically faithful to root.

This is the per-JD lexical scorer. It reproduces root `features.py`'s
`_tok`/`_tokens` tokenizer, its per-candidate evidence-doc construction
(summary + all career texts) and its per-facet min-max normalization EXACTLY,
generalized to iterate `profile.signals` instead of the hardcoded FACETS dict.

Used ONLY by the per-JD precompute step (`jd_compile.py`): the corpus is
JD-independent but the queries are JD-dependent, so the BM25 pass runs once per
JD and is persisted to `bm25_facets.parquet`. The CPU rank path never imports
this module (or rank_bm25) — it just loads the parquet via features.py.

Root reference (features.py):
    _tok = re.compile(r"[a-z0-9\\+\\#\\.]+")
    def _tokens(s): return _tok.findall(s.lower())
    bm25 = BM25Okapi([_tokens(d) for d in docs])
    scores = bm25.get_scores(_tokens(q)); min-max per facet -> [0,1]
    evidence_docs[i] = summary_text(c) + " " + " ".join(career_text(j) ...)
"""
from __future__ import annotations

import numpy as np

# Tokenizer — verbatim from root features.py (_tok / _tokens).
_TOK = __import__("re").compile(r"[a-z0-9\+\#\.]+")


def tokenize(s: str) -> list:
    """Lowercase + findall, exactly as root `_tokens`."""
    return _TOK.findall((s or "").lower())


def evidence_doc(headline_summary: str, jobs_text: str, sep: str = "\x1f") -> str:
    """Per-candidate evidence doc, the v7 equivalent of root's
    `summary_text(c) + " " + " ".join(career_text(j) for j in career_history)`.

    In the v7 artifact layout this is:
        headline_summary + " " + jobs_text.replace(SEP, " ")
    where jobs_text is the per-job chunks joined by the unit separator.
    """
    head = headline_summary or ""
    body = (jobs_text or "").replace(sep, " ")
    return head + " " + body


def bm25_facet_scores(docs: list, profile) -> dict:
    """Return {signal_id: minmax-normalized BM25 scores} for EVERY signal.

    Faithful to root `bm25_facet_scores`: one BM25Okapi over the tokenized
    corpus, `get_scores` per signal query, min-max normalized per facet into
    [0,1] (zeros when the score range is degenerate).

    Raises ImportError (actionable) if rank_bm25 is not importable — jd_compile
    surfaces it; the rank step never calls this.
    """
    try:
        from rank_bm25 import BM25Okapi
    except Exception as exc:  # noqa: BLE001 — re-raise as actionable ImportError
        raise ImportError(
            "rank_bm25 is required to compute the BM25 lexical channel "
            "(bm25_facet_scores). Install it with `pip install rank_bm25` "
            "(it is listed in requirements-precompute.txt). Only the per-JD "
            "precompute step needs it; the rank step does not."
        ) from exc

    tokenized = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized)

    out = {}
    for s in profile.signals:                      # ALL signals (each has a query)
        scores = np.asarray(bm25.get_scores(tokenize(s.query)), dtype=float)
        rng = scores.max() - scores.min() if scores.size else 0.0
        if rng > 0:
            out[s.id] = (scores - scores.min()) / rng
        else:
            out[s.id] = np.zeros_like(scores)
    return out
