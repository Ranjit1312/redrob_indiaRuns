"""
jd_compile.py — v7 component 2: compile the JD profile (CPU, seconds).

Loads jd/jd_profile.yaml (via the validated Profile seam), embeds the signal
queries IN PROFILE ORDER with the same bi-encoder used for the candidate side,
and writes to <RANKER_ROOT|repo>/artifacts_v7/:

    jd_vectors.npy     (n_signals, 384) signal-query embeddings, rows in
                       profile.signal_ids() order (so features.py's matmul lines
                       up column i <-> signal i)
    jd_profile.yaml    a verbatim copy, so the artifacts are self-contained and
                       rank.py can read the JD seam from the artifact dir

This is the ONLY step (besides the optional train.py re-run) needed when the JD
changes — the candidate embedding pass is untouched.

Reads the embedder id from method_config.yaml (models.embedder). CPU-fast (only
a handful of query texts). Prints the torch device.

    python jd_compile.py
"""
import argparse, os, shutil, time

import numpy as np
import pandas as pd

from redrob_ranker import profile as rprofile
from redrob_ranker import bm25 as rbm25

BASE = os.environ.get("RANKER_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(BASE, "artifacts_v7")
HERE = os.path.dirname(os.path.abspath(__file__))
JD_DEFAULT = os.path.join(HERE, "jd", "jd_profile.yaml")
METHOD_DEFAULT = os.path.join(HERE, "jd", "method_config.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jd", default=JD_DEFAULT)
    ap.add_argument("--method", default=METHOD_DEFAULT)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(ART, exist_ok=True)

    profile, method = rprofile.load(args.jd, args.method)
    order = profile.signal_ids()
    queries = [s.query for s in profile.signals]
    embedder = method.models["embedder"]["name"]
    print(f"[jd] {len(queries)} signal queries (order): {order}")

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"   # CPU is fine: few texts
    model = SentenceTransformer(embedder, device=dev)
    print(f"[jd] {embedder} on device={model.device}")

    vecs = model.encode(queries, normalize_embeddings=True,
                        show_progress_bar=False).astype(np.float32)
    np.save(os.path.join(ART, "jd_vectors.npy"), vecs)
    shutil.copyfile(args.jd, os.path.join(ART, "jd_profile.yaml"))

    print(f"[jd] wrote jd_vectors.npy {vecs.shape} + jd_profile.yaml copy to "
          f"{ART} ({time.time()-t0:.1f}s)")

    # -- BM25 lexical facet channel (per-JD; loaded by features.py) ----------
    # The corpus (evidence docs) is JD-independent but the queries are
    # JD-dependent, so this lexical pass belongs in the per-JD step and is
    # persisted; the CPU rank step just loads the parquet (no rank_bm25 there).
    # NOTE: this adds a ~corpus-size BM25 pass (tens of seconds for ~100K docs).
    t1 = time.time()
    evid_path = os.path.join(ART, "evidence_texts.parquet")
    evid_df = pd.read_parquet(evid_path)
    docs = [rbm25.evidence_doc(h, j)
            for h, j in zip(evid_df["headline_summary"].values,
                            evid_df["jobs_text"].values)]
    facets = rbm25.bm25_facet_scores(docs, profile)        # {id: scores}, ALL signals
    bm25_df = pd.DataFrame(index=evid_df.index)
    for sid in order:
        bm25_df[f"{sid}__bm25"] = facets[sid]
    bm25_out = os.path.join(ART, "bm25_facets.parquet")
    bm25_df.to_parquet(bm25_out)
    print(f"[jd] wrote bm25_facets.parquet {bm25_df.shape} "
          f"(cols: {list(bm25_df.columns)}) over {len(docs)} docs "
          f"({time.time()-t1:.1f}s)")


if __name__ == "__main__":
    main()
