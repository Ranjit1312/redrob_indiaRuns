"""
precompute_teacher.py — Step 4a: cross-encoder TEACHER run (offline, GPU).

  1. scans candidates.jsonl once:
       - evidence text per candidate (headline+summary+career; never skills)
       - extended redrob-signal features for LGBM -> artifacts_full/signals_features.parquet
  2. shortlist = top --shortlist by v3 final score + --negatives random others
     (negatives teach the student what irrelevance looks like)
  3. BAAI/bge-reranker-v2-m3 (8K context; nothing truncates) scores
     (positive-only JD query, evidence) pairs on CUDA
  4. BIAS VALIDATION before trusting labels: scores the corpus's most common
     description templates + audit probes, prints separation
  5. labels gated by anti-stuffer corroboration + integrity
     -> artifacts_full/pseudo_labels_v2.parquet

    python precompute_teacher.py
"""
import argparse, collections, os, time

import numpy as np
import pandas as pd
import psutil

BASE = os.environ.get("RANKER_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "artifacts_full")

# positive-evidence-only query (negated clauses removed: a relevance CE can't
# follow instructions, and negated terms just add lexical overlap)
JD_QUERY = (
    "Senior AI engineer for an HR-tech product company who has shipped "
    "embeddings-based retrieval and semantic search to production for real users, "
    "built hybrid search over a vector database such as FAISS, Pinecone, Qdrant or "
    "Elasticsearch, owned an end-to-end ranking, search or recommendation system at "
    "scale at a product company, designs ranking evaluation with NDCG, MRR, MAP, "
    "offline-to-online calibration and A/B testing, writes strong production Python, "
    "5-9 years of experience with recent hands-on coding."
)

SIGNAL_COLS = ["profile_views_received_30d", "applications_submitted_30d",
               "avg_response_time_hours", "connection_count",
               "endorsements_received", "search_appearance_30d",
               "saved_by_recruiters_30d", "offer_acceptance_rate"]
BOOL_COLS = ["verified_email", "verified_phone", "linkedin_connected"]

def evidence_text(c):
    p = c["profile"]
    parts = [f"{p.get('headline','')}. {p.get('summary') or ''}"]
    for j in c["career_history"]:
        parts.append(f"{j.get('title','')} at {j.get('company','')}: "
                     f"{j.get('description') or ''}")
    return "\n".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortlist", type=int, default=8000)
    ap.add_argument("--negatives", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    t0, proc = time.time(), psutil.Process()
    import orjson, torch

    # ---- 1. scan: evidence + extended signal features -----------------------
    ids, evid, sig_rows = [], {}, []
    desc_counter = collections.Counter()
    with open(os.path.join(BASE, "candidates.jsonl"), "rb") as f:
        for line in f:
            if not line.strip():
                continue
            c = orjson.loads(line)
            cid = c["candidate_id"]
            ids.append(cid)
            evid[cid] = evidence_text(c)
            for j in c["career_history"]:
                d = (j.get("description") or "").strip()
                if d:
                    desc_counter[d] += 1
            s = c["redrob_signals"]
            row = {"candidate_id": cid}
            for k in SIGNAL_COLS:
                row[k] = s.get(k, -1)
            for k in BOOL_COLS:
                row[k] = int(bool(s.get(k)))
            sig_rows.append(row)
    sig_df = pd.DataFrame(sig_rows).set_index("candidate_id")
    sig_df.to_parquet(os.path.join(ART, "signals_features.parquet"))
    print(f"[teacher] scanned {len(ids)} candidates "
          f"({time.time()-t0:.0f}s); signals_features.parquet written")

    # ---- 2. shortlist ---------------------------------------------------------
    scores = pd.read_parquet(os.path.join(ART, "scores_step35_v4.parquet"))
    order = scores["final"].sort_values(ascending=False)
    short_ids = list(order.index[:args.shortlist])
    rest = order.index[args.shortlist:]
    rng = np.random.default_rng(42)
    neg_ids = list(rest[rng.choice(len(rest), size=args.negatives, replace=False)])
    todo = short_ids + neg_ids
    print(f"[teacher] CE will score {len(todo)} candidates "
          f"({args.shortlist} shortlist + {args.negatives} random negatives)")

    # ---- 3. cross-encoder -----------------------------------------------------
    from sentence_transformers import CrossEncoder
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ce = CrossEncoder("BAAI/bge-reranker-v2-m3", device=dev, max_length=1024,
                      model_kwargs={"torch_dtype": torch.float16})
    print(f"[teacher] bge-reranker-v2-m3 loaded on {dev}")

    # ---- 4. bias validation BEFORE labeling -----------------------------------
    print("\n[bias-check] CE scores for the corpus's most common templates:")
    templates = desc_counter.most_common(14)
    tpl_scores = ce.predict([(JD_QUERY, t) for t, _ in templates],
                            batch_size=args.batch)
    tpl_scores = 1 / (1 + np.exp(-np.asarray(tpl_scores, dtype=np.float64)))
    for (t, n), sc in sorted(zip(templates, tpl_scores), key=lambda x: -x[1]):
        print(f"  {sc:.3f} (x{n:>6}) {t[:110]}")

    probes = {"CAND_0046525": "elite #1", "CAND_0011687": "deep stack",
              "CAND_0055905": "elite content (London)", "CAND_0083879": "summary-only claims",
              "CAND_0027691": "churn-current", "CAND_0000031": "plain-language",
              "CAND_0069638": "rec-lite hedger"}
    pr = ce.predict([(JD_QUERY, evid[p]) for p in probes], batch_size=args.batch)
    pr = 1 / (1 + np.exp(-np.asarray(pr, dtype=np.float64)))
    print("\n[bias-check] probe candidates:")
    for (pid, note), sc in zip(probes.items(), pr):
        print(f"  {sc:.3f} {pid} ({note})")

    # ---- score the shortlist ---------------------------------------------------
    t1 = time.time()
    pairs = [(JD_QUERY, evid[cid]) for cid in todo]
    raw = ce.predict(pairs, batch_size=args.batch, show_progress_bar=True)
    raw = np.asarray(raw, dtype=np.float64)
    ce_sig = 1 / (1 + np.exp(-raw))
    print(f"[teacher] scored {len(todo)} pairs in {time.time()-t1:.0f}s "
          f"({len(todo)/(time.time()-t1):.1f}/s)")

    # ---- 5. gate labels ---------------------------------------------------------
    ref = pd.read_parquet(os.path.join(ART, "features_refined_v3.parquet"))
    feats = pd.read_parquet(os.path.join(ART, "features.parquet"))
    sub = ref.loc[todo]
    assess_corr = (sub["assess_strength"].values
                   * np.minimum(1.0, sub["evid_coverage"].values / 0.25))
    stuffer_gate = 0.4 + 0.6 * np.maximum.reduce(
        [feats.loc[todo, "ai_skill_corroboration"].values,
         sub["evid_coverage"].values, assess_corr])
    pseudo = ce_sig * stuffer_gate * sub["integrity"].values
    out = pd.DataFrame({"candidate_id": todo, "ce_raw": raw, "ce_sigmoid": ce_sig,
                        "pseudo_label": pseudo, "is_negative_sample":
                        [0]*len(short_ids) + [1]*len(neg_ids)}
                       ).set_index("candidate_id")
    out.to_parquet(os.path.join(ART, "pseudo_labels_v2.parquet"))

    mem, cpu = proc.memory_info(), proc.cpu_times()
    print(f"\n[teacher] pseudo_labels_v2.parquet written ({len(out)} rows)")
    print(f"[teacher] label dist: p10={np.percentile(pseudo,10):.3f} "
          f"p50={np.percentile(pseudo,50):.3f} p90={np.percentile(pseudo,90):.3f} "
          f"max={pseudo.max():.3f}")
    print(f"[teacher] wall={time.time()-t0:.0f}s cpu={cpu.user+cpu.system:.0f}s "
          f"peak_ram={getattr(mem,'peak_wset',mem.rss)/2**30:.2f}GB "
          f"gpu_peak={torch.cuda.max_memory_allocated()/2**30:.2f}GB" if dev == "cuda" else "")

if __name__ == "__main__":
    main()
