"""
rank.py — v7 component 4: the constrained ranking step (CPU ONLY, <5 min).

STRUCTURALLY CPU-only: imports numpy / pandas / pyarrow / psutil / lightgbm /
pyyaml (transitively, via redrob_ranker) and nothing else — torch and
sentence-transformers are never imported, so GPU use is impossible by
construction.

ARTIFACT-ONLY: rank.py never opens the raw candidate file. Scoring uses the
precomputed parquet/npy artifacts, and reasoning is sourced from the feature /
intrinsic frames (the v4 decision — candidate facts live in parquet, not
re-parsed from JSON). --candidates is accepted only to report the pool size in
telemetry; it can be omitted.

Unlike v6's monolith, v7 delegates to the pure core:

    profile, method = redrob_ranker.profile.load(...)             # the JD seam
    df              = redrob_ranker.features.build_features(...)   # live features
    rr              = redrob_ranker.rules.compute_rules(df, ...)   # composite+gates

then blends the LightGBM student over the rules composite and writes the
submission. The composite/gate formula lives ONCE (rules.py); the feature
arithmetic lives ONCE (features.py); rank.py is the orchestrator + I/O +
telemetry + reasoning, kept faithful to the v6 working tree.

Artifacts (in <RANKER_ROOT or repo-root>/artifacts_v7/):
    job_embeddings.npy + job_offsets.npy + summary_embeddings.npy
    evidence_texts.parquet + intrinsic.parquet     (JD-independent)
    jd_vectors.npy + jd_profile.yaml               (JD-compiled)
    model.txt + feature_cols.json                  (trained student)

NOTE — BM25 lexical channel: the v5 BM25 channel is preserved. The per-JD pass
runs in jd_compile.py (rank_bm25 over the evidence docs) and is persisted to
artifacts_v7/bm25_facets.parquet; features.py loads it and rules.py mixes
lex_fit = mean(<id>__bm25) into the additive composite at additive_weights.lexical
(0.09). Keeping rank_bm25 in the per-JD step means this CPU rank path never
imports it — torch/sentence-transformers/rank_bm25 are all absent here.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py --candidates ./candidates.jsonl --features-only   (for train.py)
"""
import argparse, csv, json, os, time

import numpy as np
import pandas as pd
import psutil

from redrob_ranker import profile as rprofile
from redrob_ranker import features as rfeatures
from redrob_ranker import rules as rrules
from redrob_ranker.rules import mm

BASE = os.environ.get("RANKER_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "artifacts_v7")
HERE = OUT
JD_DEFAULT = os.path.join(HERE, "jd", "jd_profile.yaml")
METHOD_DEFAULT = os.path.join(HERE, "jd", "method_config.yaml")

# columns that are gates / outputs, never LGBM features (mirror of train.py)
GATES = {"availability_mult", "integrity", "notice_pen", "loc2_v4",
         "fit_rules", "final_rules", "dormant", "low_rr", "anach",
         "la_lt_signup", "concurrent_deg", "remote_pref", "no_reloc",
         "city_ok", "notice_days"}

# ---------------------------------------------------------------------------
# telemetry (same snap/T pattern as v5/v6 rank.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=os.path.join(BASE, "candidates.jsonl"))
    ap.add_argument("--out", default=os.path.join(OUT, "submission.csv"))
    ap.add_argument("--features-only", action="store_true",
                    help="write artifacts_v7/features_v7.parquet and exit "
                         "(used by train.py; no model required)")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--jd", default=JD_DEFAULT, help="path to jd_profile.yaml")
    ap.add_argument("--method", default=METHOD_DEFAULT,
                    help="path to method_config.yaml")
    args = ap.parse_args()
    t = T(); t_all = time.time()

    # ---- 0. artifact footprint --------------------------------------------
    ARTIFACTS = ["job_embeddings.npy", "summary_embeddings.npy",
                 "job_offsets.npy", "jd_vectors.npy", "jd_profile.yaml",
                 "evidence_texts.parquet", "intrinsic.parquet"]
    if os.path.exists(os.path.join(ART, "model.txt")):
        ARTIFACTS += ["model.txt", "feature_cols.json"]
    sizes = {a: os.path.getsize(os.path.join(ART, a)) / 2**20
             for a in ARTIFACTS if os.path.exists(os.path.join(ART, a))}
    print("[rank] artifact footprint (MB):",
          {k: round(v, 1) for k, v in sizes.items()},
          f"TOTAL={sum(sizes.values()):.1f}MB")

    # ---- 1. load the JD seam (Profile + Method) ---------------------------
    # Prefer the artifact-dir jd_profile.yaml (jd_compile copies it there so the
    # artifacts are self-contained); fall back to the source tree.
    art_jd = os.path.join(ART, "jd_profile.yaml")
    jd_path = art_jd if os.path.exists(art_jd) else args.jd
    profile, method = rprofile.load(jd_path, args.method)
    print(f"[rank] JD '{profile.role.title}' — {len(profile.signals)} signals "
          f"{profile.signal_ids()} ({len(profile.evidence_signals())} with evidence)")
    t.mark("load_profile")

    # ---- 2-9. live feature build (features.py) ----------------------------
    df = rfeatures.build_features_parallel(profile, method, art_dir=ART)
    idx = df.index
    N = len(idx)
    t.mark("build_features")

    # ---- composite + gates (the single rules engine) ----------------------
    rr = rrules.compute_rules(df, profile, method)
    fit = rr.fit
    integ, avail, notice_pen = rr.integrity, rr.availability, rr.notice_pen
    loc2 = rr.loc2
    df["fit_rules"] = fit
    df["final_rules"] = rr.final_rules
    evid_coverage = df["evid_coverage"].values.astype(np.float64)
    t.mark("compute_rules")

    # ---- 10. features-only mode (feeds train.py) --------------------------
    if args.features_only:
        out_path = os.path.join(ART, "features_v7.parquet")
        df.to_parquet(out_path)
        print(f"[rank] --features-only: wrote {out_path} "
              f"({df.shape[0]} x {df.shape[1]}) in {time.time()-t_all:.1f}s")
        return

    # ---- 11. student predict + blend --------------------------------------
    import lightgbm as lgb
    booster = lgb.Booster(model_file=os.path.join(ART, "model.txt"))
    feat_cols = json.load(open(os.path.join(ART, "feature_cols.json")))
    lgbm_score = booster.predict(df[feat_cols].astype("float32").values)
    alpha = method.alpha
    blend = alpha * mm(lgbm_score) + (1 - alpha) * mm(fit)
    final = mm(blend) * integ * avail * notice_pen
    t.mark("lgbm_predict_blend")

    # ---- 12. top-k --------------------------------------------------------
    k = min(args.topk, N)
    pos = np.argpartition(-final, k - 1)[:k]
    pos = pos[np.lexsort((idx.values[pos], -evid_coverage[pos], -final[pos]))]
    top_ids, top_scores = idx.values[pos], final[pos]
    t.mark("argpartition_topk")

    # ---- 13. grounded reasoning + write -----------------------------------
    # rank.py never opens the raw candidate file: scoring is artifact-only, and
    # reasoning is sourced entirely from the precomputed feature/intrinsic frames
    # below (the v4 decision — facts live in parquet, not re-parsed from JSON).
    # EVID_LABEL is now sourced from the profile's evidence signals (their
    # human label), not a hardcoded dict — a JD swap re-labels reasoning too.
    EVID_LABEL = {s.id: s.label for s in profile.evidence_signals()}
    ALL_LABEL = {s.id: s.label for s in profile.signals}  # incl. dense-only axes
    loc2_s = pd.Series(loc2, index=idx)
    # Reasoning facts come from the precomputed intrinsic table (the single
    # source of candidate facts) — NOT a re-parse of raw JSON. The feature frame
    # already carries yoe / notice_days / remote_pref / dormant / hopper /
    # assess_strength / evid_*; only these three aren't on it, so pull just them.
    intr_reason = pd.read_parquet(
        os.path.join(ART, "intrinsic.parquet"),
        columns=["current_title", "open_to_work_flag", "recruiter_response_rate"])

    def reasoning_for(cid):
        row, ir = df.loc[cid], intr_reason.loc[cid]
        named = sorted(((kk, row[f"evid_{kk}"]) for kk in EVID_LABEL),
                       key=lambda x: -x[1])
        hit = [EVID_LABEL[kk] for kk, v in named[:3] if v >= 0.4]
        if hit:
            strengths = ", ".join(hit)
        else:
            # No career-text evidence cleared the bar — name the strongest
            # SEMANTIC (dense) alignment instead of a bland catch-all, so
            # genuinely-relevant but text-light profiles read accurately (and
            # reasonings vary). Uses the per-signal dense columns already on row.
            top = max(profile.signals,
                      key=lambda s: row.get(f"{s.id}__recencywt", 0.0))
            strengths = (f"applied-ML background; closest alignment is "
                         f"{ALL_LABEL[top.id]} (semantic match; career text "
                         f"light on explicit terms)")
        out = (f"{ir['current_title']} with {row['yoe']:.0f} yrs; "
               f"career history shows {strengths}")
        if row["assess_strength"] > 0.5:
            out += "; platform-validated assessments in relevant skills"
        cc = []
        if row["dormant"]: cc.append("dormant and unresponsive on platform")
        if row["hopper"]: cc.append("frequent job changes")
        if not ir["open_to_work_flag"]: cc.append("not flagged open-to-work")
        if row["remote_pref"]: cc.append("prefers remote vs hybrid role")
        nd = int(row["notice_days"])
        if nd and nd > profile.role.notice_preference_days:
            cc.append(f"{nd}-day notice period")
        if loc2_s[cid] <= 0.4: cc.append("location/relocation friction")
        rrr = float(ir["recruiter_response_rate"])     # -1 == missing (NOT "low")
        if 0 <= rrr < 0.2:
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

    # ---- 15. telemetry ----------------------------------------------------
    J = int(np.load(os.path.join(ART, "job_embeddings.npy"), mmap_mode="r").shape[0])
    total_wall = time.time() - t_all
    tele = {"total_wall_s": round(total_wall, 2),
            "total_cpu_s": round(sum(s["cpu_s"] for s in t.stages), 2),
            "peak_memory_gb": max(s["peak_gb"] for s in t.stages),
            "artifact_mb": {k: round(v, 1) for k, v in sizes.items()},
            "artifact_total_mb": round(sum(sizes.values()), 1),
            "candidates_jsonl_mb": (round(os.path.getsize(args.candidates) / 2**20, 1)
                                    if os.path.exists(args.candidates) else None),
            "n_candidates": int(N), "n_job_chunks": int(J),
            "alpha": alpha,
            "budget": {"wall_limit_s": 300, "ram_limit_gb": 16, "disk_limit_gb": 5},
            "headroom": {"wall_pct_used": round(total_wall / 300 * 100, 1),
                         "ram_pct_used": round(max(s["peak_gb"] for s in t.stages) / 16 * 100, 1)},
            "stages": t.stages}
    json.dump(tele, open(os.path.join(OUT, "telemetry.json"), "w"), indent=2)
    print(f"\n[rank] TOTAL wall={total_wall:.1f}s ({total_wall/300*100:.1f}% of 5-min budget) "
          f"peak_ram={tele['peak_memory_gb']:.2f}GB ({tele['headroom']['ram_pct_used']:.0f}% of 16GB) "
          f"artifacts={tele['artifact_total_mb']:.0f}MB")
    print(f"[rank] wrote {args.out} ({k} rows) + telemetry.json")


if __name__ == "__main__":
    main()
