"""
run_step35.py — Steps 1–3.5 on the FULL candidates.jsonl. NO cross-encoder, NO LightGBM.

    Step 1   dense bi-encoder facet sims (bge-small-en-v1.5, recency-weighted)
    Step 2   BM25 lexical facet scores
    Step 3   structured fit features + honeypot gate + availability multiplier
    Step 3.5 transparent reference composite (the features_v2 baseline, completed)
             -> gates -> argpartition top-100
             -> submission.csv + top100candidates.jsonl

Also saves all artifacts (features, embeddings, scores) to --out-dir so the later
LightGBM phase and any query/gate refinement never re-embeds, and logs wall time,
CPU time and peak memory per stage to resource_log.json.

    python run_step35.py --candidates ../../candidates.jsonl --out-dir ../../artifacts_full
"""
import argparse, csv, json, os, time

import numpy as np
import pandas as pd
import psutil

import features as F
from features import (FACETS, FACET_ORDER, EmbeddingBackend, career_text,
                      summary_text, dense_facet_features, bm25_facet_scores,
                      structured_features, honeypot_flags, availability_multiplier)

# --------------------------------------------------------------------------- #
# Resource logging                                                            #
# --------------------------------------------------------------------------- #
PROC = psutil.Process()

def _snapshot():
    cpu = PROC.cpu_times()
    mem = PROC.memory_info()
    return {"wall": time.time(), "cpu": cpu.user + cpu.system,
            "rss_gb": mem.rss / 2**30,
            "peak_gb": getattr(mem, "peak_wset", mem.rss) / 2**30}

class StageTimer:
    def __init__(self):
        self.stages, self._t0 = [], _snapshot()
    def mark(self, name):
        t1 = _snapshot()
        self.stages.append({"stage": name,
                            "wall_s": round(t1["wall"] - self._t0["wall"], 2),
                            "cpu_s":  round(t1["cpu"]  - self._t0["cpu"], 2),
                            "rss_gb_end": round(t1["rss_gb"], 3),
                            "peak_gb_so_far": round(t1["peak_gb"], 3)})
        print(f"[time] {name:34} wall={self.stages[-1]['wall_s']:8.1f}s  "
              f"cpu={self.stages[-1]['cpu_s']:8.1f}s  "
              f"rss={t1['rss_gb']:.2f}GB  peak={t1['peak_gb']:.2f}GB", flush=True)
        self._t0 = t1

# --------------------------------------------------------------------------- #
# Step 3.5 — transparent reference composite (features_v2 baseline, completed) #
# --------------------------------------------------------------------------- #
def reference_composite(df):
    """Interpretable fit = dense facet evidence + structured fit, then soft
    multiplicative penalties (anti-stuffer, services-only, not-hands-on,
    location). Hard gates (honeypot, availability) are applied OUTSIDE."""
    dense_fit = (
        0.28 * df["ranking__recencywt"] +
        0.22 * df["retrieval__recencywt"] +
        0.12 * df["vectordb__recencywt"] +
        0.10 * df["evaluation__recencywt"] +
        0.10 * df["applied_ml__recencywt"] +
        0.08 * df["yoe_fit"] +
        0.10 * df["domain_nlp_ratio"]
    )
    bm25_cols = [f"{f}__bm25" for f in FACET_ORDER if f"{f}__bm25" in df.columns]
    lex_fit = df[bm25_cols].mean(axis=1) if bm25_cols else 0.0
    fit = 0.8 * dense_fit + 0.2 * lex_fit                      # hybrid dense+lexical

    fit = fit * (0.4 + 0.6 * df["ai_skill_corroboration"])     # anti keyword-stuffer
    fit = fit * (1.0 - 0.30 * df["only_consulting"])           # services-only career
    fit = fit * np.where(df["months_since_ic_role"] > 18, 0.85, 1.0)  # no recent hands-on
    fit = fit * (0.70 + 0.30 * df["location_fit"])             # Pune/Noida/T1/relocate
    return fit

# --------------------------------------------------------------------------- #
# Reasoning (same grounded template as rank.py, no lightgbm import)            #
# --------------------------------------------------------------------------- #
FACET_LABEL = {
    "retrieval": "embeddings-based retrieval", "vectordb": "vector/hybrid search",
    "ranking": "ranking/recommender systems", "evaluation": "ranking evaluation",
    "applied_ml": "applied ML in production", "llm_ft": "LLM fine-tuning",
}

def reasoning_for(row, prof):
    p = prof["profile"]; sig = prof["redrob_signals"]
    facets = [(f, row[f + "__recencywt"]) for f in FACET_LABEL]
    best = sorted(facets, key=lambda t: -t[1])[:2]
    strengths = " and ".join(FACET_LABEL[f] for f, v in best if v > 0) or "adjacent ML work"
    s = (f"{p['current_title']} with {p['years_of_experience']:.0f} yrs; "
         f"evidence strongest in {strengths}")
    concerns = []
    if row["only_consulting"]: concerns.append("services-only background")
    if row["months_since_ic_role"] > 18: concerns.append("no recent hands-on coding role")
    if sig.get("recruiter_response_rate", 1) < 0.2: concerns.append("low recruiter response rate")
    if row["ai_skill_corroboration"] < 0.2 and row["ai_skills_claimed"] > 3:
        concerns.append("AI skills not backed by career history")
    np_days = sig.get("notice_period_days")
    if np_days and np_days > 90: concerns.append(f"{np_days}-day notice period")
    if concerns:
        s += "; concern: " + ", ".join(concerns[:2])
    return s + "."

# --------------------------------------------------------------------------- #
def load_jsonl(path):
    import orjson
    out = []
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                out.append(orjson.loads(line))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--out-dir", default="artifacts_full")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    base = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--submission", default=os.path.join(base, "submission.csv"))
    ap.add_argument("--top100", default=os.path.join(base, "top100candidates.jsonl"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    T = StageTimer()

    # ---- load --------------------------------------------------------------
    cands = load_jsonl(args.candidates)
    n = len(cands)
    print(f"[step35] {n} candidates loaded")
    T.mark("load_jsonl")

    # ---- Step 1: embeddings -------------------------------------------------
    job_offsets, all_job_texts, all_summaries = [], [], []
    for c in cands:
        start = len(all_job_texts)
        for j in c["career_history"]:
            all_job_texts.append(career_text(j))
        job_offsets.append((start, len(all_job_texts)))
        all_summaries.append(summary_text(c))
    print(f"[step35] {len(all_job_texts)} job texts, {n} summaries to embed")

    backend = EmbeddingBackend(all_job_texts + all_summaries + list(FACETS.values()))
    print(f"[step35] embedding backend = {backend.kind}")
    if backend.kind == "st":
        enc = lambda texts: backend.model.encode(
            [t if t else " " for t in texts], normalize_embeddings=True,
            batch_size=args.batch_size, show_progress_bar=True).astype(np.float32)
    else:
        enc = backend.encode
    job_matrix  = enc(all_job_texts) if all_job_texts else np.zeros((0, 1), np.float32)
    summ_matrix = enc(all_summaries)
    facet_vecs  = {f: v for f, v in zip(FACET_ORDER, enc(list(FACETS.values())))}
    np.save(os.path.join(args.out_dir, "job_embeddings.npy"), job_matrix)
    np.save(os.path.join(args.out_dir, "summary_embeddings.npy"), summ_matrix)
    np.save(os.path.join(args.out_dir, "job_offsets.npy"), np.asarray(job_offsets))
    T.mark("step1_embeddings")

    # ---- Step 2: BM25 -------------------------------------------------------
    evidence_docs = [summary_text(c) + " " + " ".join(career_text(j)
                     for j in c["career_history"]) for c in cands]
    bm25 = bm25_facet_scores(evidence_docs)
    T.mark("step2_bm25")

    # ---- Step 3: per-candidate features + gates -----------------------------
    rows, meta = [], []
    for i, c in enumerate(cands):
        s, e = job_offsets[i]
        feats = {"candidate_id": c["candidate_id"]}
        feats.update(dense_facet_features(c, job_matrix[s:e], summ_matrix[i], facet_vecs))
        if bm25 is not None:
            for fname in FACET_ORDER:
                feats[f"{fname}__bm25"] = float(bm25[fname][i])
        feats.update(structured_features(c, feats))
        flags = honeypot_flags(c)
        feats["honeypot_flag"] = int(len(flags) > 0)
        feats["availability_mult"] = availability_multiplier(c)
        rows.append(feats)
        meta.append({"candidate_id": c["candidate_id"],
                     "title": c["profile"]["current_title"],
                     "yoe": c["profile"]["years_of_experience"],
                     "honeypot_reasons": ";".join(flags)})
    df = pd.DataFrame(rows).set_index("candidate_id").fillna(0.0)
    meta = pd.DataFrame(meta).set_index("candidate_id")
    df.to_parquet(os.path.join(args.out_dir, "features.parquet"))
    meta.to_parquet(os.path.join(args.out_dir, "meta.parquet"))
    print(f"[step35] features {df.shape}, honeypots flagged = {int(df['honeypot_flag'].sum())}")
    T.mark("step3_structured_gates")

    # ---- Step 3.5: composite -> gates -> top-100 -----------------------------
    fit = reference_composite(df).values.astype(float)
    fit = (fit - fit.min()) / (np.ptp(fit) + 1e-9)
    final = fit * df["availability_mult"].values * (1 - df["honeypot_flag"].values)
    pd.DataFrame({"candidate_id": df.index, "fit": fit, "final": final}
                 ).set_index("candidate_id").to_parquet(
                 os.path.join(args.out_dir, "scores_step35.parquet"))

    k = min(args.topk, n)
    idx = np.argpartition(-final, k - 1)[:k]
    idx = idx[np.lexsort((df.index.values[idx], -final[idx]))]   # tiebreak: id asc
    ids, scores = df.index.values[idx], final[idx]
    T.mark("step35_score_topk")

    # ---- outputs -------------------------------------------------------------
    by_id = {cid: cands[i] for cid, i in zip(ids, idx)}
    prev, rows_out = 1.0, []
    for rank, (cid, sc) in enumerate(zip(ids, scores), start=1):
        sc = float(min(sc, prev)); prev = sc
        rows_out.append((cid, rank, round(sc, 6),
                         reasoning_for(df.loc[cid], by_id[cid])))
    with open(args.submission, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        w.writerows(rows_out)

    import orjson
    with open(args.top100, "wb") as f:
        for cid, rank, sc, reason in rows_out:
            rec = {"_rank": rank, "_score": sc, "_reasoning": reason}
            rec.update(by_id[cid])
            f.write(orjson.dumps(rec) + b"\n")
    T.mark("write_outputs")

    log = {"n_candidates": n, "n_job_texts": len(all_job_texts),
           "embedding_backend": backend.kind, "stages": T.stages,
           "total_wall_s": round(sum(s["wall_s"] for s in T.stages), 1),
           "total_cpu_s": round(sum(s["cpu_s"] for s in T.stages), 1),
           "peak_memory_gb": max(s["peak_gb_so_far"] for s in T.stages)}
    with open(os.path.join(args.out_dir, "resource_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(json.dumps(log, indent=2))
    print(f"[step35] wrote submission.csv + top100candidates.jsonl "
          f"(honeypots in top-100 = {int(df.loc[ids, 'honeypot_flag'].sum())})")

if __name__ == "__main__":
    main()
