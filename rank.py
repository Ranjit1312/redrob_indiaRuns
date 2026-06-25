"""
rank.py — v2 CONSTRAINED ranking step: second refinement pass after the
top-50 audit. Reuses saved embeddings/features (artifacts_full).
No re-embedding / teacher / LGBM.

Audit findings encoded (2026-06-11, top-50 review):
  V1  evidence is scored on career_history[].description ONLY — templated
      summaries make unsupported "I built FAISS search" claims (rank-10 case).
  V2  per-job evidence with CONTEXT and OWNERSHIP:
        - "internal knowledge base / internal dashboard" scale -> 0.4x credit
        - ownership verbs (led/owned/designed/architected/from scratch) -> 1.0,
          participation only -> 0.7x
        - recency-weighted (stale retrieval work counts less), max over jobs
  V3  ranking-eval credit requires ranking metrics (NDCG/MRR/MAP/recall@k) or
      offline-online / A-B language; "human relevance judgments" alone and
      BLEU/ROUGE chatbot eval no longer qualify.
  V4  job-hopper disqualifier: 4+ jobs with mean stint < 19 months -> 0.55x.
  V5  location per the JD's actual city lists: Pune/Noida preferred;
      Hyderabad/Mumbai/Delhi-NCR ok; everything else needs willing_to_relocate;
      outside India + not relocating is near-disqualifying.
  V6  integrity: yoe-vs-career threshold tightened (rank-9 honeypot at 15.2yr
      vs 86mo slipped the 1.9x bound) + summary-stated years cross-check.

    python rank.py
"""
import csv, json, os, re, time

import numpy as np
import pandas as pd
import psutil

BASE = os.environ.get("RANKER_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "artifacts_full")

from features import FACET_ORDER, months_since_end, recency_weight

# --------------------------------------------------------------------------- #
# telemetry (same snap()/T pattern as v1's rank.py)                           #
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
# evidence regexes (career descriptions only)                                 #
# --------------------------------------------------------------------------- #
EVID = {
    "retrieval": re.compile(
        r"embedding[- ]based (?:retrieval|search)|semantic search|dense retrieval|"
        r"vector (?:search|recall)|\bfaiss\b|sentence[- ]transformers?|"
        r"bi[- ]encoder|fine[- ]tuned? (?:bge|e5|embedding)|"
        r"keyword[- ]based.{0,40}embedding", re.I),
    "vectordb": re.compile(
        r"\bfaiss\b|pinecone|weaviate|qdrant|milvus|opensearch|elasticsearch|"
        r"vector (?:database|db|index|store)|hybrid (?:search|retrieval)|"
        r"\bhnsw\b|approximate nearest|bm25.{0,40}(?:dense|vector|embedding)|"
        r"(?:dense|vector|embedding).{0,40}bm25", re.I),
    "rankeval": re.compile(
        r"ndcg|\bmrr\b|\bmap\b|mean average precision|recall@|precision@|"
        r"a/b[- ]test|ab[- ]test|offline.{0,50}online|online.{0,50}offline|"
        r"relevance label|interleav", re.I),
    "ltr_recsys": re.compile(
        r"learning[- ]to[- ]rank|\bltr\b|lambdamart|ranking (?:model|layer|"
        r"function|pipeline)|re[- ]?rank|recommendation (?:system|engine|model)|"
        r"recsys|discovery feed|search (?:system|engine|product|ranking)", re.I),
}
INTERNAL_RE  = re.compile(r"internal (?:knowledge base|dashboard|tool|search|stakeholders)|"
                          r"internal[- ]facing", re.I)
OWNER_RE     = re.compile(r"\bled\b|\bowned?\b|designed|architected|from scratch|"
                          r"built (?:the|an?|our)\b|drove\b|spearhead", re.I)
SCALE_RE     = re.compile(r"\d+\s?m\+|\bqps\b|queries per (?:month|second|day)|"
                          r"\bp9[59]\b|million|\d+k\+? (?:users|queries|docs)", re.I)
CV_RE  = re.compile(r"computer vision|image (?:classification|moderation|recognition)|"
                    r"\byolo\b|opencv|object detection|segmentation|resnet|"
                    r"\bspeech\b|\basr\b|robotics", re.I)
NLP_RE = re.compile(r"\bnlp\b|natural language|retrieval|search|ranking|recommend|"
                    r"embedding|semantic|information retrieval|text|\bllm\b|transformer", re.I)
YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*years of (?:hands[- ]on )?experience", re.I)

PREFERRED = ("pune", "noida")
OK_CITIES = ("hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "ncr")

def evidence_scores(c):
    """Per-category evidence in [0,1]: max over jobs of
    hit * ownership * context * recency."""
    out = {k: 0.0 for k in EVID}
    depth_bonus = 0.0
    for j in c["career_history"]:
        desc = f"{j.get('title','')}. {j.get('description','')}"
        rec = recency_weight(months_since_end(j), halflife=30.0)   # gentler decay
        ctx = 0.4 if INTERNAL_RE.search(desc) else 1.0
        own = 1.0 if OWNER_RE.search(desc) else 0.7
        scale = 1.0 if SCALE_RE.search(desc) else 0.85
        for k, rx in EVID.items():
            if rx.search(desc):
                out[k] = max(out[k], ctx * own * scale * rec)
        if ctx == 1.0 and OWNER_RE.search(desc) and SCALE_RE.search(desc):
            depth_bonus = max(depth_bonus, rec)
    coverage = float(np.mean(list(out.values())))
    return out, coverage, depth_bonus

def location_fit(c):
    p, sig = c["profile"], c["redrob_signals"]
    loc = ((p.get("location") or "") + " " + (p.get("country") or "")).lower()
    relocate = bool(sig.get("willing_to_relocate"))
    in_india = ("india" in loc or any(x in loc for x in
                ("pradesh", "maharashtra", "karnataka", "telangana", "tamil",
                 "bengal", "kerala", "gujarat", "rajasthan", "punjab", "delhi",
                 "haryana", "bihar", "odisha", "assam")) or
                any(x in loc for x in PREFERRED + OK_CITIES))
    if any(x in loc for x in PREFERRED):
        return 1.0
    if any(x in loc for x in OK_CITIES):
        return 0.9
    if in_india:
        return 0.75 if relocate else 0.40
    return 0.50 if relocate else 0.12     # no visa sponsorship, case-by-case

def integrity_and_hops(c):
    p, jobs, skills = c["profile"], c["career_history"], c.get("skills", [])
    sig = c["redrob_signals"]
    yoe = float(p.get("years_of_experience", 0))
    career_m = sum(j.get("duration_months", 0) for j in jobs)

    integrity, reasons = 1.0, []
    def hard(name):
        nonlocal integrity; integrity *= 0.05; reasons.append(name)
    def soft(name, f):
        nonlocal integrity; integrity *= f; reasons.append(name)

    if career_m > 0 and career_m / 12.0 > yoe + 3.5:
        hard("career_sum_exceeds_stated_yoe")
    if any(j.get("duration_months", 0) / 12.0 > yoe + 1.5 for j in jobs):
        hard("single_role_exceeds_stated_yoe")
    if career_m > 0 and yoe * 12 > career_m * 1.6 + 18:                   # V6
        hard("stated_yoe_far_exceeds_career_history")
    m = YEARS_RE.search(p.get("summary") or "")
    if m and abs(yoe - float(m.group(1))) > 3.0:                          # V6
        hard("stated_yoe_contradicts_summary")
    if any(s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
           for s in skills):
        hard("expert_proficiency_with_zero_usage")
    if sum(s.get("proficiency") == "expert" for s in skills) >= 8:
        hard("implausibly_many_expert_skills")
    n_imp = sum(1 for s in skills
                if (s.get("duration_months") or 0) > career_m * 1.25 + 6)
    if career_m > 0 and n_imp >= 2:
        soft("skill_durations_exceed_career", 1.0 - min(0.15, 0.04 * n_imp))
    sal = sig.get("expected_salary_range_inr_lpa") or {}
    if sal.get("min", 0) > sal.get("max", 1e9):
        soft("salary_range_inverted", 0.97)

    durs = [j.get("duration_months", 0) for j in jobs]
    hopper = int(len(durs) >= 4 and np.mean(durs) < 19)                   # V4
    return integrity, reasons, hopper

def main():
    t = T(); t_all = time.time()
    t0, proc = time.time(), psutil.Process()
    import orjson
    feats = pd.read_parquet(os.path.join(ART, "features.parquet"))

    ids, rows, cand_by_id = [], [], {}
    with open(os.path.join(BASE, "candidates.jsonl"), "rb") as f:
        for line in f:
            if not line.strip():
                continue
            c = orjson.loads(line)
            cid = c["candidate_id"]
            ev, cov, depth = evidence_scores(c)
            integ, reasons, hopper = integrity_and_hops(c)
            text = " ".join(f"{j.get('title','')}. {j.get('description','')}"
                            for j in c["career_history"])
            cv_n, nlp_n = len(CV_RE.findall(text)), len(NLP_RE.findall(text))
            notice = c["redrob_signals"].get("notice_period_days") or 0
            rows.append({**{f"evid_{k}": v for k, v in ev.items()},
                         "evid_coverage": cov, "depth_bonus": depth,
                         "cv_primary": int(cv_n >= 3 and cv_n > nlp_n),
                         "hopper": hopper, "integrity": integ,
                         "integrity_reasons": ";".join(reasons),
                         "loc_fit2": location_fit(c),
                         "notice_pen": 1.0 if notice <= 90 else
                                       (0.93 if notice <= 120 else 0.88)})
            ids.append(cid)
            cand_by_id[cid] = c
    t.mark("jsonl_scan")
    ref = pd.DataFrame(rows, index=pd.Index(ids, name="candidate_id"))
    df = feats.join(ref)
    t.mark("feature_build")
    print(f"[v2] scanned {len(df)} candidates in {time.time()-t0:.0f}s")
    print(f"[v2] integrity<0.1: {int((df['integrity'] < 0.1).sum())}; "
          f"hoppers: {int(df['hopper'].sum())}; cv_primary: {int(df['cv_primary'].sum())}")
    rc = ref.loc[ref["integrity_reasons"] != "", "integrity_reasons"]
    print(f"[v2] integrity reasons: {rc.str.split(';').explode().value_counts().to_dict()}")
    print(f"[v2] coverage>=0.5: {int((df['evid_coverage'] >= 0.5).sum())}, "
          f">=0.25: {int((df['evid_coverage'] >= 0.25).sum())}")

    # ---- composite v3 --------------------------------------------------------
    dense_fit = (0.28 * df["ranking__recencywt"] + 0.22 * df["retrieval__recencywt"] +
                 0.12 * df["vectordb__recencywt"] + 0.10 * df["evaluation__recencywt"] +
                 0.10 * df["applied_ml__recencywt"] + 0.08 * df["yoe_fit"] +
                 0.10 * df["domain_nlp_ratio"])
    bm25_cols = [f"{f}__bm25" for f in FACET_ORDER if f"{f}__bm25" in df.columns]
    lex_fit = df[bm25_cols].mean(axis=1)

    fit = (0.40 * dense_fit + 0.10 * lex_fit +
           0.42 * df["evid_coverage"] + 0.08 * df["depth_bonus"])
    fit *= 0.4 + 0.6 * np.maximum(df["ai_skill_corroboration"], df["evid_coverage"])
    fit *= np.where(df["cv_primary"] == 1, 0.60, 1.0)
    fit *= np.where(df["hopper"] == 1, 0.55, 1.0)                          # V4
    fit *= 1.0 - 0.30 * df["only_consulting"]
    fit *= np.where(df["months_since_ic_role"] > 18, 0.85, 1.0)
    fit *= 0.55 + 0.45 * df["loc_fit2"]                                    # V5
    # outside India + unwilling to relocate: case-by-case per JD, no visa
    # sponsorship -> too risky for top ranks regardless of content
    fit *= np.where(df["loc_fit2"] <= 0.12, 0.60, 1.0)
    fit = (fit - fit.min()) / (np.ptp(fit.values) + 1e-9)

    final = (fit * df["availability_mult"] * df["notice_pen"] * df["integrity"]).values

    # ---- top-100 --------------------------------------------------------------
    k = 100
    idx = np.argpartition(-final, k - 1)[:k]
    idx = idx[np.lexsort((df.index.values[idx], -final[idx]))]
    top_ids, top_scores = df.index.values[idx], final[idx]
    t.mark("scoring_topk")

    EVID_LABEL = {"retrieval": "embeddings/semantic-search work",
                  "vectordb": "vector-database/hybrid-search experience",
                  "rankeval": "ranking-evaluation rigor (NDCG/MRR/A-B)",
                  "ltr_recsys": "shipped ranking/recommendation systems"}

    def reasoning_for(cid):
        row, c = df.loc[cid], cand_by_id[cid]
        p, sig = c["profile"], c["redrob_signals"]
        named = sorted(((k, row[f"evid_{k}"]) for k in EVID_LABEL),
                       key=lambda t: -t[1])
        strengths = ", ".join(EVID_LABEL[k] for k, v in named[:3] if v >= 0.4) \
                    or "adjacent applied-ML work"
        s = (f"{p['current_title']} with {p['years_of_experience']:.0f} yrs; "
             f"career history shows {strengths}")
        concerns = []
        if row["hopper"]: concerns.append("frequent job changes")
        if row["only_consulting"]: concerns.append("services-only background")
        if row["months_since_ic_role"] > 18: concerns.append("no recent hands-on coding role")
        if sig.get("recruiter_response_rate", 1) < 0.2: concerns.append("low recruiter response rate")
        if row["cv_primary"]: concerns.append("primarily computer-vision background")
        nd = sig.get("notice_period_days")
        if nd and nd > 90: concerns.append(f"{nd}-day notice period")
        if row["loc_fit2"] <= 0.4: concerns.append("location/relocation friction")
        if concerns:
            s += "; concern: " + ", ".join(concerns[:2])
        return s + "."

    prev, rows_out = 1.0, []
    for rank, (cid, sc) in enumerate(zip(top_ids, top_scores), start=1):
        sc = float(min(sc, prev)); prev = sc
        rows_out.append((cid, rank, round(sc, 6), reasoning_for(cid)))

    with open(os.path.join(OUT, "submission.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        w.writerows(rows_out)
    with open(os.path.join(OUT, "top100candidates.jsonl"), "wb") as f:
        for cid, rank, sc, reason in rows_out:
            rec = {"_rank": rank, "_score": sc, "_reasoning": reason}
            rec.update(cand_by_id[cid])
            f.write(orjson.dumps(rec) + b"\n")

    ref.to_parquet(os.path.join(ART, "features_refined_v2.parquet"))
    pd.DataFrame({"candidate_id": df.index, "fit": fit, "final": final}
                 ).set_index("candidate_id").to_parquet(
                 os.path.join(ART, "scores_step35_v3.parquet"))
    t.mark("outputs")

    mem, cpu = proc.memory_info(), proc.cpu_times()
    print(f"[v2] wall={time.time()-t0:.1f}s cpu={cpu.user+cpu.system:.1f}s "
          f"peak={getattr(mem,'peak_wset',mem.rss)/2**30:.2f}GB")
    print(f"[v2] min integrity in top-100 = {df.loc[top_ids,'integrity'].min():.2f}")
    ranks = pd.Series(final, index=df.index).rank(ascending=False, method="min")
    PROBES = {
        "CAND_0046525": "audit #1 (keep ~1)", "CAND_0005260": "audit -> top3",
        "CAND_0011687": "audit -> top3 (Indore)", "CAND_0046064": "audit -> top5",
        "CAND_0041669": "audit #5", "CAND_0071974": "audit -> top10 (Vizag)",
        "CAND_0039383": "audit #7", "CAND_0064326": "audit #9 (behavioral best)",
        "CAND_0010770": "HONEYPOT -> out", "CAND_0083879": "summary-only -> out",
        "CAND_0030953": "hopper -> out", "CAND_0027691": "churn-current -> down",
        "CAND_0055905": "London no-reloc -> down", "CAND_0000031": "watch",
    }
    for pid, note in PROBES.items():
        print(f"[v2] {pid}: rank {int(ranks[pid]):6}  ({note}; "
              f"cov={df.loc[pid,'evid_coverage']:.2f} integ={df.loc[pid,'integrity']:.2f} "
              f"loc={df.loc[pid,'loc_fit2']:.2f} hop={int(df.loc[pid,'hopper'])})")

    total_wall = time.time() - t_all
    tele = {"version": "v2", "total_wall_s": round(total_wall, 2),
            "total_cpu_s": round(sum(s["cpu_s"] for s in t.stages), 2),
            "peak_memory_gb": max(s["peak_gb"] for s in t.stages),
            "n_candidates": int(len(df)),
            "budget": {"wall_limit_s": 300, "ram_limit_gb": 16},
            "headroom": {"wall_pct_used": round(total_wall / 300 * 100, 1),
                         "ram_pct_used": round(max(s["peak_gb"] for s in t.stages) / 16 * 100, 1)},
            "stages": t.stages}
    with open(os.path.join(OUT, "telemetry.json"), "w") as f:
        json.dump(tele, f, indent=2)
    print(f"[v2] wrote submission.csv + top100candidates.jsonl + telemetry.json "
          f"(wall={total_wall:.1f}s, {total_wall/300*100:.1f}% of 5-min budget)")

if __name__ == "__main__":
    main()
