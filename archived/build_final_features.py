"""
build_final_features.py — pipeline stage 5 (offline, CPU): the full-profile
audit rules + final blend (the v4 refinement logic).

Its role in this package: produce artifacts_full/features_v4.parquet (dormant /
anachronism / remote-pref / notice flags) and
artifacts_full/scores_v4_final.parquet. It also writes a blended
submission_blend.csv / top100candidates.jsonl in this directory as
diagnostics — the shipped, reproducible ranking comes from rank.py
(which recomputes exactly this blend from the saved artifacts).

Base-rate-validated rules (unchanged from the audited v4 pass):
  B1 dormant+unresponsive (inactive>6mo AND rrr<0.2): availability x0.5 (JD's
     own exclusion example; 3.7% of pool)
  B2 cert anachronisms (LangChain<2022 / LLaMA<2023): integrity x0.30 (45 in
     pool — rare authenticity fingerprint)
  B3 last_active<signup (7.5% noise): integrity x0.97 soft
  B4 concurrent same-window degrees (0.73%): integrity x0.93 soft
  B5 remote-pref + no-relocate + not Pune/Noida/OK-city: loc capped 0.25;
     any remote-pref: loc x0.9 (JD is hybrid Tue/Thu)
  B6 notice>120d: pen 0.88 -> 0.85
Recomputes fit (v3 formula + B5), blends with saved LGBM (a=0.2), gates.

    python build_final_features.py
"""
import csv, json, os, time
import numpy as np
import pandas as pd

BASE = os.environ.get("RANKER_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "artifacts_full")
ALPHA = 0.2
PREFERRED = ("pune", "noida")
OK_CITIES = ("hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "ncr")

def mm(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (np.ptp(x) + 1e-12)

def main():
    t0 = time.time()
    import orjson
    feats = pd.read_parquet(os.path.join(ART, "features.parquet"))
    ref = pd.read_parquet(os.path.join(ART, "features_refined_v3.parquet"))
    lgbm = pd.read_parquet(os.path.join(ART, "lgbm_scores_v3.parquet"))  # tuned student

    rows, prof_cache = {}, {}
    with open(os.path.join(BASE, "candidates.jsonl"), "rb") as f:
        for line in f:
            if not line.strip(): continue
            c = orjson.loads(line)
            cid = c["candidate_id"]; s = c["redrob_signals"]; p = c["profile"]
            la, su = s.get("last_active_date") or "", s.get("signup_date") or ""
            mode = s.get("preferred_work_mode")
            reloc = bool(s.get("willing_to_relocate"))
            loc = ((p.get("location") or "") + " " + (p.get("country") or "")).lower()
            if la:
                from datetime import date
                d = date(int(la[:4]), int(la[5:7]), int(la[8:10]))
                months_inactive = (date(2026, 6, 11) - d).days / 30.44
            else:
                months_inactive = 12.0
            anach = any((("langchain" in (ct.get("name") or "").lower() and (ct.get("year") or 9999) < 2022)
                         or ("llama" in (ct.get("name") or "").lower() and (ct.get("year") or 9999) < 2023))
                        for ct in (c.get("certifications") or []))
            edu = [(e.get("start_year"), e.get("end_year")) for e in (c.get("education") or [])]
            edu = [e for e in edu if e[0] and e[1]]
            concurrent = any(edu[i] == edu[j] for i in range(len(edu))
                             for j in range(i + 1, len(edu)))
            rows[cid] = {
                "dormant": int(months_inactive > 6 and s.get("recruiter_response_rate", 1) < 0.2),
                "low_rr": int(s.get("recruiter_response_rate", 1) < 0.2),
                "anach": int(anach),
                "la_lt_signup": int(bool(la and su and la < su)),
                "concurrent_deg": int(concurrent),
                "remote_pref": int(mode == "remote"),
                "no_reloc": int(not reloc),
                "city_ok": int(any(x in loc for x in PREFERRED + OK_CITIES)),
                "notice": s.get("notice_period_days") or 0}
    v4 = pd.DataFrame.from_dict(rows, orient="index").loc[feats.index]
    print(f"[v4] scan {time.time()-t0:.0f}s; dormant={v4['dormant'].sum()} "
          f"anach={v4['anach'].sum()} la<signup={v4['la_lt_signup'].sum()} "
          f"concur={v4['concurrent_deg'].sum()}")

    X = feats.join(ref)
    # B5 location adjustment
    loc2 = ref["loc_fit2"].copy()
    # India, non-target city, unwilling to relocate (encoded as 0.40 by the rule
    # pass): hybrid Tue/Thu office makes this a real blocker -> 0.33
    loc2 = loc2.where(loc2 != 0.40, 0.33)
    capmask = (v4["remote_pref"] == 1) & (v4["no_reloc"] == 1) & (v4["city_ok"] == 0)
    loc2[capmask] = np.minimum(loc2[capmask], 0.25)
    loc2[v4["remote_pref"] == 1] *= 0.9
    # integrity v4
    integ = (ref["integrity"]
             * np.where(v4["anach"] == 1, 0.30, 1.0)
             * np.where(v4["la_lt_signup"] == 1, 0.97, 1.0)
             * np.where(v4["concurrent_deg"] == 1, 0.93, 1.0))
    # availability v4 (B1)
    avail = (feats["availability_mult"]
             * np.where(v4["dormant"] == 1, 0.5, 1.0)
             * np.where((v4["low_rr"] == 1) & (v4["dormant"] == 0), 0.8, 1.0))
    # notice v4 (B6)
    notice_pen = np.where(v4["notice"] <= 90, 1.0,
                          np.where(v4["notice"] < 120, 0.93,
                                   np.where(v4["notice"] == 120, 0.90, 0.85)))

    FAC = ["retrieval", "vectordb", "ranking", "evaluation", "applied_ml", "llm_ft"]
    dense_fit = (0.28 * X["ranking__recencywt"] + 0.22 * X["retrieval__recencywt"] +
                 0.12 * X["vectordb__recencywt"] + 0.10 * X["evaluation__recencywt"] +
                 0.10 * X["applied_ml__recencywt"] + 0.08 * X["yoe_fit"] +
                 0.10 * X["domain_nlp_ratio"])
    lex_fit = X[[f"{f}__bm25" for f in FAC]].mean(axis=1)
    fit = (0.38 * dense_fit + 0.09 * lex_fit + 0.40 * X["evid_coverage"] +
           0.08 * X["depth_bonus"] + 0.05 * X["assess_strength"])
    assess_corr = (X["assess_strength"].values
                   * np.minimum(1.0, X["evid_coverage"].values / 0.25))
    fit *= 0.4 + 0.6 * np.maximum.reduce([X["ai_skill_corroboration"].values,
                                          X["evid_coverage"].values, assess_corr])
    fit *= np.where(X["cv_primary"] == 1, 0.60, 1.0)
    fit *= np.where(X["hopper"] == 1, 0.55, 1.0)
    fit *= 1.0 - 0.30 * X["only_consulting"]
    fit *= np.where(X["months_since_ic_role"] > 18, 0.85, 1.0)
    fit *= 0.55 + 0.45 * loc2
    fit *= np.where(loc2 <= 0.12, 0.60, 1.0)

    blend = ALPHA * mm(lgbm.loc[feats.index, "lgbm_score"]) + (1 - ALPHA) * mm(fit.values)
    final = (mm(blend) * integ.values * avail.values * notice_pen)

    idx = feats.index
    k = 100
    pos = np.argpartition(-final, k - 1)[:k]
    cov = X["evid_coverage"].values
    pos = pos[np.lexsort((idx.values[pos], -cov[pos], -final[pos]))]
    top_ids, top_scores = idx.values[pos], final[pos]

    ranks = pd.Series(final, index=idx).rank(ascending=False, method="min")

    wanted, prof = set(top_ids), {}
    with open(os.path.join(BASE, "candidates.jsonl"), "rb") as f:
        for line in f:
            if not line.strip(): continue
            c = orjson.loads(line)
            if c["candidate_id"] in wanted:
                prof[c["candidate_id"]] = c
                if len(prof) == len(wanted): break

    EL = {"retrieval": "embeddings/semantic-search work",
          "vectordb": "vector-database/hybrid-search experience",
          "rankeval": "ranking-evaluation work",
          "ltr_recsys": "shipped ranking/recommendation systems"}
    def reason(cid):
        row, c = ref.loc[cid], prof[cid]
        p, s = c["profile"], c["redrob_signals"]
        named = sorted(((kk, row[f"evid_{kk}"]) for kk in EL), key=lambda x: -x[1])
        st = ", ".join(EL[kk] for kk, v in named[:3] if v >= 0.4) or "adjacent applied-ML work"
        out = (f"{p['current_title']} with {p['years_of_experience']:.0f} yrs; "
               f"career history shows {st}")
        if row["assess_strength"] > 0.5:
            out += "; platform-validated assessments in relevant skills"
        cc = []
        if v4.loc[cid, "dormant"]: cc.append("dormant and unresponsive on platform")
        if row["hopper"]: cc.append("frequent job changes")
        if not s.get("open_to_work_flag"): cc.append("not flagged open-to-work")
        if v4.loc[cid, "remote_pref"]: cc.append("prefers remote vs hybrid role")
        nd = s.get("notice_period_days")
        if nd and nd > 90: cc.append(f"{nd}-day notice period")
        if loc2[cid] <= 0.4: cc.append("location/relocation friction")
        if s.get("recruiter_response_rate", 1) < 0.2: cc.append("low recruiter response rate")
        if cc: out += "; concern: " + ", ".join(cc[:2])
        return out + "."

    prev, out_rows = 1.0, []
    for rank, (cid, sc) in enumerate(zip(top_ids, top_scores), start=1):
        sc = float(min(sc, prev)); prev = sc
        out_rows.append((cid, rank, round(sc, 6), reason(cid)))
    with open(os.path.join(OUT, "submission_blend.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["candidate_id", "rank", "score", "reasoning"]); w.writerows(out_rows)
    with open(os.path.join(OUT, "top100candidates.jsonl"), "wb") as f:
        for cid, rank, sc, rs in out_rows:
            rec = {"_rank": rank, "_score": sc, "_reasoning": rs}; rec.update(prof[cid])
            f.write(orjson.dumps(rec) + b"\n")
    v4.to_parquet(os.path.join(ART, "features_v4.parquet"))
    pd.DataFrame({"candidate_id": idx, "final": final}).set_index("candidate_id"
                 ).to_parquet(os.path.join(ART, "scores_v4_final.parquet"))
    old_path = os.path.join(ART, "scores_blended.parquet")
    if os.path.exists(old_path):
        old = pd.read_parquet(old_path)["final"]
        o50 = set(old.rank(ascending=False, method="min").pipe(lambda r: r[r <= 50]).index)
        n50 = set(ranks[ranks <= 50].index)
        print(f"[v4] top-50 overlap with previous blend: {len(o50 & n50)}/50")
    else:
        print("[v4] scores_blended.parquet not present — skipping overlap diff "
              "(expected on a fresh precompute)")
    print(f"[v4] wall={time.time()-t0:.0f}s; wrote features_v4.parquet + "
          f"scores_v4_final.parquet + submission_blend.csv + top100candidates.jsonl")

if __name__ == "__main__":
    main()
