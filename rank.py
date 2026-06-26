"""
rank.py — the CONSTRAINED ranking step (target: <=5 min wall, <=16 GB RAM,
CPU only, no network). No limits are enforced here — instead it records full
telemetry per stage (wall, CPU, RSS, peak RAM) plus the disk footprint of every
artifact it depends on, written to rank_telemetry.json in this directory.

STRUCTURALLY CPU-only: this file imports numpy/pandas/pyarrow/orjson/psutil/
lightgbm and nothing else — torch and sentence-transformers are never imported.

What it does (the single Stage-3 reproduction command):
    1. load precomputed feature artifacts + LightGBM student   (no transformers)
    2. predict student scores for all 100K
    3. recompute the deterministic composite (pure numpy) and blend (a=0.2)
    4. apply gates: integrity x availability x notice
    5. top-100 via argpartition; stream candidates.jsonl for the 100 profiles
    6. write submission.csv with grounded reasoning

    python rank.py --candidates ..\\..\\candidates.jsonl --out .\\submission.csv
"""
import argparse, csv, json, os, re, time

import numpy as np
import pandas as pd
import psutil
import lightgbm as lgb

# Fast candidate_id extractor — avoids a full JSON parse of every line when we
# only need the id set for the coverage check (keeps the constrained step cheap).
_CID_RE = re.compile(rb'"candidate_id"\s*:\s*"(CAND_\d+)"')

def scan_candidate_ids(path):
    """Stream candidates.jsonl once and return the set of candidate_ids in it."""
    ids = set()
    with open(path, "rb") as f:
        for line in f:
            m = _CID_RE.search(line)
            if m:
                ids.add(m.group(1).decode())
    return ids

BASE = os.environ.get("RANKER_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "artifacts_full")
ALPHA = 0.2

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

def mm(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (np.ptp(x) + 1e-12)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=os.path.join(BASE, "candidates.jsonl"))
    ap.add_argument("--out", default=os.path.join(OUT, "submission.csv"))
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--no-coverage-check", action="store_true",
                    help="skip the candidate_id coverage scan (restores the minimal "
                         "~3.6s path; only safe when candidates.jsonl is the precomputed pool)")
    ap.add_argument("--strict-coverage", action="store_true",
                    help="abort (exit 2) if ANY supplied candidate_id is missing from "
                         "the precomputed artifacts instead of warning + ranking the covered subset")
    args = ap.parse_args()
    t = T(); t_all = time.time()
    import orjson

    # ---- artifact disk footprint -----------------------------------------------
    ARTIFACTS = ["features.parquet", "features_refined_v3.parquet",
                 "features_v4.parquet", "signals_features.parquet",
                 "model_v3.txt", "feature_cols_v2.json"]
    sizes = {a: os.path.getsize(os.path.join(ART, a)) / 2**20 for a in ARTIFACTS}
    print("[rank] artifact footprint (MB):",
          {k: round(v, 1) for k, v in sizes.items()},
          f"TOTAL={sum(sizes.values()):.1f}MB")

    # ---- 1. load artifacts -------------------------------------------------------
    feats = pd.read_parquet(os.path.join(ART, "features.parquet"))

    # ---- 1a. coverage check: only candidates with precomputed features can be ranked.
    # On the official pool (candidates.jsonl == the precomputed pool) this is a no-op:
    # `covered == feat_ids` so no restriction happens and the output stays byte-identical.
    # On a NEW/partial dataset it restricts ranking to the candidates we actually have
    # features for and tells the operator how to precompute the missing ones.
    if not args.no_coverage_check:
        wanted = scan_candidate_ids(args.candidates)
        feat_ids = set(feats.index)
        covered = wanted & feat_ids
        missing = wanted - feat_ids
        if not wanted:
            print(f"[rank][WARN] no candidate_ids read from {args.candidates}; "
                  f"skipping coverage check and ranking the full precomputed pool.")
        else:
            if missing:
                print(f"[rank][WARN] {len(missing)} of {len(wanted)} supplied candidate_ids "
                      f"have NO precomputed features and cannot be ranked (new candidates).")
                print(f"[rank][WARN] Run the one-time UNCONSTRAINED precompute for just the new ids:")
                print(f"[rank][WARN]     python precompute.py --candidates {args.candidates}")
                print(f"[rank][WARN] then re-run this command. (Precompute may use GPU/network; "
                      f"the rank step never does.)")
            if not covered:
                raise SystemExit("[rank] FATAL: none of the supplied candidates have "
                                 "precomputed features — run precompute.py first.")
            if covered != feat_ids:           # supplied set is a subset → rank only those
                feats = feats.loc[feats.index.isin(covered)]
                print(f"[rank] ranking the {len(feats)} supplied candidates that have features.")
            if missing and args.strict_coverage:
                raise SystemExit("[rank] --strict-coverage set and candidates are missing; aborting.")
        t.mark("coverage_check")

    ref = pd.read_parquet(os.path.join(ART, "features_refined_v3.parquet"))
    v4 = pd.read_parquet(os.path.join(ART, "features_v4.parquet")).loc[feats.index]
    sig = pd.read_parquet(os.path.join(ART, "signals_features.parquet"))
    feat_cols = json.load(open(os.path.join(ART, "feature_cols_v2.json")))
    booster = lgb.Booster(model_file=os.path.join(ART, "model_v3.txt"))
    X = feats.join(ref).join(sig)
    idx = X.index
    t.mark("load_artifacts")

    # ---- 2. student predictions ---------------------------------------------------
    lgbm_score = booster.predict(X[feat_cols].astype("float32").values)
    t.mark("lgbm_predict_100k")

    # ---- 3. deterministic composite (pure numpy, replicates the rule pass) --------
    # Evidence-gated scoring. The JD is explicit that listed AI keywords are a trap
    # ("all the AI keywords as skills but title is Marketing Manager is not a fit")
    # and that career history is the truth. So career EVIDENCE is a necessary gate,
    # and listed-skill corroboration / platform assessments MULTIPLY in as bounded
    # modifiers (neutral when absent) -- they can confirm or discount evidence, but
    # can never substitute for it. (Earlier versions used max(ai_corr, evid_coverage,
    # assess_corr), which let a keyword signal override missing career evidence.)
    FACETS = ["retrieval", "vectordb", "ranking", "evaluation", "applied_ml", "llm_ft"]
    dense_fit = (0.28 * X["ranking__recencywt"] + 0.22 * X["retrieval__recencywt"] +
                 0.12 * X["vectordb__recencywt"] + 0.10 * X["evaluation__recencywt"] +
                 0.10 * X["applied_ml__recencywt"] + 0.08 * X["yoe_fit"] +
                 0.10 * X["domain_nlp_ratio"])
    lex_fit = X[[f"{f}__bm25" for f in FACETS]].mean(axis=1)

    # additive evidence channels: dense semantic match + lexical + ownership depth
    fit = 0.38 * dense_fit + 0.09 * lex_fit + 0.08 * X["depth_bonus"]

    # (1) evidence GATE -- the necessary condition. Low floor (0.15) so "no career
    #     evidence" really costs; full credit (1.0) at evid_coverage == 1.
    g_evid = 0.15 + 0.85 * X["evid_coverage"]
    # (2) claim-consistency -- DISCOUNT-only. Neutral (x1) when the candidate lists
    #     no AI skills; ~0.5 when they list AI buzzwords the career text doesn't
    #     support (the keyword-stuffer signal). Never lifts above 1.0.
    m_claim = np.where(X["ai_skills_claimed"].values == 0,
                       1.0, 0.5 + 0.5 * X["ai_skill_corroboration"].values)
    # (3) assessment BONUS -- already coverage-gated (full credit only at cov>=0.25),
    #     so it can only reward a candidate who ALSO has career evidence. Absence of
    #     assessments (94% of the pool) is neutral, never zero.
    assess_corr = (X["assess_strength"].values
                   * np.minimum(1.0, X["evid_coverage"].values / 0.25))
    m_assess = 1.0 + 0.25 * assess_corr
    fit = fit * g_evid * m_claim * m_assess

    # (4) recent-coding ladder ("this role writes code"). months_since_ic_role == 999
    #     is the never-held-an-IC-role sentinel -> falls in the >36 bucket (heavy
    #     floor), no longer collapsed with "coded 19 months ago".
    msi = X["months_since_ic_role"].values
    recency_mult = np.where(msi > 36, 0.35,
                   np.where(msi > 18, 0.70,
                   np.where(msi > 1.0, 0.90, 1.0)))
    fit = fit * recency_mult
    # (5) experience band (JD 5-9 yrs) as a real multiplier, not a ~3% additive nudge.
    fit = fit * (0.60 + 0.40 * X["yoe_fit"])

    # structured damps (unchanged)
    fit *= np.where(X["cv_primary"] == 1, 0.60, 1.0)
    fit *= np.where(X["hopper"] == 1, 0.55, 1.0)
    fit *= 1.0 - 0.30 * X["only_consulting"]
    # v4 location: India non-target no-reloc 0.40->0.33; remote-pref damps
    loc2 = X["loc_fit2"].where(X["loc_fit2"] != 0.40, 0.33)
    capm = (v4["remote_pref"] == 1) & (v4["no_reloc"] == 1) & (v4["city_ok"] == 0)
    loc2[capm] = np.minimum(loc2[capm], 0.25)
    loc2[v4["remote_pref"] == 1] *= 0.9
    fit *= 0.55 + 0.45 * loc2
    fit *= np.where(loc2 <= 0.12, 0.60, 1.0)

    integ = (X["integrity"]
             * np.where(v4["anach"] == 1, 0.30, 1.0)
             * np.where(v4["la_lt_signup"] == 1, 0.97, 1.0)
             * np.where(v4["concurrent_deg"] == 1, 0.93, 1.0))
    avail = (X["availability_mult"]
             * np.where(v4["dormant"] == 1, 0.5, 1.0)
             * np.where((v4["low_rr"] == 1) & (v4["dormant"] == 0), 0.8, 1.0))
    notice_pen = np.where(v4["notice"] <= 90, 1.0,
                          np.where(v4["notice"] < 120, 0.93,
                                   np.where(v4["notice"] == 120, 0.90, 0.85)))

    blend = ALPHA * mm(lgbm_score) + (1 - ALPHA) * mm(fit.values)
    final = mm(blend) * integ.values * avail.values * notice_pen
    t.mark("composite_blend_gates")

    # ---- 4. top-100 -----------------------------------------------------------------
    k = min(args.topk, len(idx))          # fewer candidates than topk (e.g. demo uploads) is fine
    pos = np.argpartition(-final, k - 1)[:k]
    cov = X["evid_coverage"].values
    pos = pos[np.lexsort((idx.values[pos], -cov[pos], -final[pos]))]
    top_ids, top_scores = idx.values[pos], final[pos]
    t.mark("argpartition_top100")

    # ---- 5. stream candidates.jsonl for the 100 profiles ------------------------------
    wanted, prof = set(top_ids), {}
    with open(args.candidates, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            c = orjson.loads(line)
            if c["candidate_id"] in wanted:
                prof[c["candidate_id"]] = c
                if len(prof) == len(wanted):
                    break
    t.mark("stream_top100_profiles")

    # ---- 6. reasoning + write ----------------------------------------------------------
    EVID_LABEL = {"retrieval": "embeddings/semantic-search work",
                  "vectordb": "vector-database/hybrid-search experience",
                  "rankeval": "ranking-evaluation work",
                  "ltr_recsys": "shipped ranking/recommendation systems"}
    def reasoning_for(cid):
        row, c = ref.loc[cid], prof[cid]
        p, s_ = c["profile"], c["redrob_signals"]
        named = sorted(((kk, row[f"evid_{kk}"]) for kk in EVID_LABEL),
                       key=lambda x: -x[1])
        strengths = ", ".join(EVID_LABEL[kk] for kk, v in named[:3] if v >= 0.4) \
                    or "adjacent applied-ML work"
        out = (f"{p['current_title']} with {p['years_of_experience']:.0f} yrs; "
               f"career history shows {strengths}")
        if row["assess_strength"] > 0.5:
            out += "; platform-validated assessments in relevant skills"
        cc = []
        if v4.loc[cid, "dormant"]: cc.append("dormant and unresponsive on platform")
        if row["hopper"]: cc.append("frequent job changes")
        if not s_.get("open_to_work_flag"): cc.append("not flagged open-to-work")
        if v4.loc[cid, "remote_pref"]: cc.append("prefers remote vs hybrid role")
        nd = s_.get("notice_period_days")
        if nd and nd > 90: cc.append(f"{nd}-day notice period")
        if loc2[cid] <= 0.4: cc.append("location/relocation friction")
        if s_.get("recruiter_response_rate", 1) < 0.2:
            cc.append("low recruiter response rate")
        if cc:
            out += "; concern: " + ", ".join(cc[:2])
        return out + "."

    prev, rows_out = 1.0, []
    for rank, (cid, sc) in enumerate(zip(top_ids, top_scores), start=1):
        sc = float(min(sc, prev)); prev = sc
        rows_out.append((cid, rank, round(sc, 6), reasoning_for(cid)))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        w.writerows(rows_out)
    t.mark("reasoning_write_csv")

    total_wall = time.time() - t_all
    tele = {"total_wall_s": round(total_wall, 2),
            "total_cpu_s": round(sum(s["cpu_s"] for s in t.stages), 2),
            "peak_memory_gb": max(s["peak_gb"] for s in t.stages),
            "artifact_mb": {k: round(v, 1) for k, v in sizes.items()},
            "artifact_total_mb": round(sum(sizes.values()), 1),
            "candidates_jsonl_mb": round(os.path.getsize(args.candidates) / 2**20, 1),
            "n_features": len(feat_cols), "alpha": ALPHA,
            "budget": {"wall_limit_s": 300, "ram_limit_gb": 16, "disk_limit_gb": 5},
            "headroom": {"wall_pct_used": round(total_wall / 300 * 100, 1),
                         "ram_pct_used": round(max(s["peak_gb"] for s in t.stages) / 16 * 100, 1)},
            "stages": t.stages}
    json.dump(tele, open(os.path.join(OUT, "rank_telemetry.json"), "w"), indent=2)
    print(f"\n[rank] TOTAL wall={total_wall:.1f}s ({total_wall/300*100:.1f}% of 5-min budget) "
          f"peak_ram={tele['peak_memory_gb']:.2f}GB ({tele['headroom']['ram_pct_used']:.0f}% of 16GB) "
          f"artifacts={tele['artifact_total_mb']:.0f}MB")
    print(f"[rank] wrote {args.out} (100 rows) + rank_telemetry.json")

if __name__ == "__main__":
    main()
