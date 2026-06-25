"""
rank.py — v1 CONSTRAINED ranking step (baseline steps 1-3 + transparent
composite). NO evidence regexes, NO assessments, NO ML — exactly the
reference_composite() of run_step35.py, replayed from saved artifacts.

Loads artifacts_full/features.parquet + meta.parquet (built once by
run_step35.py), computes the v1 composite, gates by honeypot flag and
availability, takes the top-100 and streams candidates.jsonl for those 100
profiles to write grounded reasoning.

    python rank.py
"""
import csv, json, os, time

import numpy as np
import pandas as pd
import psutil

BASE = os.environ.get("RANKER_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "artifacts_full")

FACET_ORDER = ["retrieval", "vectordb", "ranking", "evaluation", "applied_ml", "llm_ft"]
FACET_LABEL = {
    "retrieval": "embeddings-based retrieval", "vectordb": "vector/hybrid search",
    "ranking": "ranking/recommender systems", "evaluation": "ranking evaluation",
    "applied_ml": "applied ML in production", "llm_ft": "LLM fine-tuning",
}

# --------------------------------------------------------------------------- #
# telemetry (same snap()/T pattern as v4's rank.py)                           #
# --------------------------------------------------------------------------- #
PROC = psutil.Process()
def snap():
    cpu, mem = PROC.cpu_times(), PROC.memory_info()
    return {"wall": time.time(), "cpu": cpu.user + cpu.system,
            "rss_gb": mem.rss / 2**30,
            "peak_gb": getattr(mem, "peak_wset", mem.rss) / 2**30}

class T:
    def __init__(self): self.stages, self._p = [], snap()
    def mark(self, name):
        c = snap()
        self.stages.append({"stage": name,
                            "wall_s": round(c["wall"] - self._p["wall"], 3),
                            "cpu_s": round(c["cpu"] - self._p["cpu"], 3),
                            "rss_gb": round(c["rss_gb"], 3),
                            "peak_gb": round(c["peak_gb"], 3)})
        s = self.stages[-1]
        print(f"[t] {name:32} wall={s['wall_s']:7.2f}s cpu={s['cpu_s']:7.2f}s "
              f"rss={s['rss_gb']:.2f}GB peak={s['peak_gb']:.2f}GB", flush=True)
        self._p = c

# --------------------------------------------------------------------------- #
# v1 composite — EXACTLY run_step35.py's reference_composite()                #
# --------------------------------------------------------------------------- #
def reference_composite(df):
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

def reasoning_for(row, c):
    p, sig = c["profile"], c["redrob_signals"]
    facets = [(f, row[f + "__recencywt"]) for f in FACET_LABEL]
    best = sorted(facets, key=lambda t: -t[1])[:2]
    strengths = " and ".join(FACET_LABEL[f] for f, v in best if v > 0) or "adjacent ML work"
    s = (f"{p['current_title']} with {p['years_of_experience']:.0f} yrs; "
         f"evidence strongest in {strengths}")
    concerns = []
    if row["only_consulting"]: concerns.append("services-only background")
    if row["months_since_ic_role"] > 18: concerns.append("no recent hands-on coding role")
    if sig.get("recruiter_response_rate", 1) < 0.2: concerns.append("low recruiter response rate")
    if concerns:
        s += "; concern: " + ", ".join(concerns[:2])
    return s + "."

def main():
    t = T(); t_all = time.time()
    import orjson

    # ---- 1. load artifacts ---------------------------------------------------
    df = pd.read_parquet(os.path.join(ART, "features.parquet"))
    meta = pd.read_parquet(os.path.join(ART, "meta.parquet"))
    t.mark("load_artifacts")

    # ---- 2. composite -> gates ------------------------------------------------
    fit = reference_composite(df).values.astype(float)
    fit = (fit - fit.min()) / (np.ptp(fit) + 1e-9)
    final = fit * df["availability_mult"].values * (1 - df["honeypot_flag"].values)
    t.mark("composite_gates")

    # ---- 3. top-100 ------------------------------------------------------------
    k = min(100, len(df))
    idx = np.argpartition(-final, k - 1)[:k]
    idx = idx[np.lexsort((df.index.values[idx], -final[idx]))]   # tiebreak: id asc
    top_ids, top_scores = df.index.values[idx], final[idx]
    t.mark("argpartition_top100")

    # ---- 4. stream candidates.jsonl for the 100 profiles (early exit) ----------
    wanted, prof = set(top_ids), {}
    with open(os.path.join(BASE, "candidates.jsonl"), "rb") as f:
        for line in f:
            if not line.strip():
                continue
            c = orjson.loads(line)
            if c["candidate_id"] in wanted:
                prof[c["candidate_id"]] = c
                if len(prof) == len(wanted):
                    break
    t.mark("stream_top100_profiles")

    # ---- 5. reasoning + write ---------------------------------------------------
    prev, rows_out = 1.0, []
    for rank, (cid, sc) in enumerate(zip(top_ids, top_scores), start=1):
        sc = float(min(sc, prev)); prev = sc                     # monotone non-increasing
        rows_out.append((cid, rank, round(sc, 6), reasoning_for(df.loc[cid], prof[cid])))
    sub_path = os.path.join(OUT, "submission.csv")
    with open(sub_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        w.writerows(rows_out)
    t.mark("reasoning_write_csv")

    total_wall = time.time() - t_all
    tele = {"version": "v1", "total_wall_s": round(total_wall, 2),
            "total_cpu_s": round(sum(s["cpu_s"] for s in t.stages), 2),
            "peak_memory_gb": max(s["peak_gb"] for s in t.stages),
            "n_candidates": int(len(df)),
            "honeypots_in_top100": int(df.loc[top_ids, "honeypot_flag"].sum()),
            "budget": {"wall_limit_s": 300, "ram_limit_gb": 16},
            "headroom": {"wall_pct_used": round(total_wall / 300 * 100, 1),
                         "ram_pct_used": round(max(s["peak_gb"] for s in t.stages) / 16 * 100, 1)},
            "stages": t.stages}
    with open(os.path.join(OUT, "telemetry.json"), "w") as f:
        json.dump(tele, f, indent=2)
    print(json.dumps(tele, indent=2))
    print(f"[v1] wrote {sub_path} (100 rows) + telemetry.json "
          f"(wall={total_wall:.1f}s, {total_wall/300*100:.1f}% of 5-min budget)")

if __name__ == "__main__":
    main()
