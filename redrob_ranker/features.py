"""
features.py — generic, signal-driven feature builder (v7).

Ports v6 `rank.py` sections 2-8 (parse_job_meta, dense_sims_pooling,
evidence_regexes, structured_text_features, assessments, rules_gates,
assemble_features) but GENERALIZED so a JD change is a config edit:

  * iterates `profile.signals` (no hardcoded facet names) for the dense pooling
    columns `<id>__recencywt/__peak/__nhits/__recent/__summary` and the live
    evidence columns `evid_<id>`;
  * `evid_coverage` is the mean over `profile.evidence_signals()` ONLY (the
    signals that carry an evidence_regex);
  * every constant is read from `profile` (per-JD) or `method` (JD-stable):
    recency half-lives, the facet-hit threshold, the in/out-domain regex+terms,
    the relevant-skill regex, the integrity ladder, availability weights, the
    notice tiers, the location ladder, the assessment knobs, the lexicons.

This module is the SINGLE home for the integrity-ladder / availability / notice
/ loc2 arithmetic: it materializes the `integrity, availability_mult,
notice_pen, loc2_v4` columns that `rules.compute_rules` then consumes (the
composite + gated product live in rules.py — this module does NOT recompute
them).

CPU-only by construction: numpy / pandas / orjson / re only. No torch, no
lightgbm. Loads its artifacts from `art_dir`:

    job_embeddings.npy, summary_embeddings.npy, job_offsets.npy,
    jd_vectors.npy, evidence_texts.parquet, intrinsic.parquet

Public entry point:

    build_features(profile, method, *, art_dir, ref_date=None) -> pd.DataFrame
        indexed by candidate_id; columns include every input column that
        rules.compute_rules reads (see rules.required_columns) plus the wider
        set of model features (kept identical to v6 for student parity).
"""
from __future__ import annotations

import os
import re
from datetime import date

import numpy as np
import orjson
import pandas as pd

SEP = "\x1f"   # unit separator: joins per-job chunks inside jobs_text


# ---------------------------------------------------------------------------
# date helpers (verbatim from v6 rank.py)
# ---------------------------------------------------------------------------
def _parse_date(d):
    if not d:
        return None
    try:
        y, m, day = (int(x) for x in d[:10].split("-"))
        return date(y, m, day)
    except Exception:
        return None


def _months_between(a, b):
    return (b.year - a.year) * 12 + (b.month - a.month) + (b.day - a.day) / 30.0


# ---------------------------------------------------------------------------
def build_features(profile, method, *, art_dir, ref_date=None) -> "pd.DataFrame":
    """Build the JD-dependent feature frame live from JD-independent artifacts.

    Parameters
    ----------
    profile : redrob_ranker.profile.Profile   (per-JD knowledge; regexes compiled)
    method  : redrob_ranker.profile.Method     (JD-stable mechanism; all numerics)
    art_dir : path to the artifacts dir (job_embeddings.npy, ..., intrinsic.parquet)
    ref_date: optional 'YYYY-MM-DD' override; defaults to method.ref_date.

    Returns
    -------
    pandas.DataFrame indexed by candidate_id. Pure (no I/O beyond loading the
    named artifacts from art_dir), CPU-only.
    """
    REF = _parse_date(ref_date or method.ref_date)
    if REF is None:
        raise ValueError(f"features: could not parse ref_date {ref_date or method.ref_date!r}")

    # -- per-JD signal seam -------------------------------------------------
    SIG_IDS = profile.signal_ids()
    EVID_SIGS = profile.evidence_signals()          # only those with a regex
    NSIG = len(SIG_IDS)

    # method-side regexes (compiled once in profile.load)
    INTERNAL_RE = method.context_re["internal"]
    OWNER_RE = method.context_re["owner"]
    SCALE_RE = method.context_re["scale"]
    YEARS_RE = method.years_re
    # domain regexes / terms come from the profile (per-JD)
    CV_RE = profile.domain.out_of_domain_re
    NLP_RE = profile.domain.in_domain_re
    NLPT = profile.domain.in_domain_terms
    CVT = profile.domain.out_of_domain_terms
    DESIRED_RE = profile.relevant_skill_re
    AITERMS = method.lexicons["ai_skill_terms"]

    # method-side lexicons / constants
    LEX = method.lexicons
    CONSULT = LEX["consulting"]; PRODIND = LEX["product_industries"]
    ICTOK = LEX["ic_tokens"]; MGTOK = LEX["mgmt_tokens"]
    REC = method.recency
    facet_hl = REC["facet_halflife_months"]
    evid_hl = REC["evidence_halflife_months"]
    facet_thr = method.thresholds["facet_hit"]
    EC = method.evidence_context
    IR = method.integrity

    # -- load artifacts -----------------------------------------------------
    job_matrix = np.load(os.path.join(art_dir, "job_embeddings.npy"))
    summ_matrix = np.load(os.path.join(art_dir, "summary_embeddings.npy"))
    offsets = np.load(os.path.join(art_dir, "job_offsets.npy"))
    jd_vecs = np.load(os.path.join(art_dir, "jd_vectors.npy"))          # (NSIG, d)
    evid_df = pd.read_parquet(os.path.join(art_dir, "evidence_texts.parquet"))
    intr = pd.read_parquet(os.path.join(art_dir, "intrinsic.parquet"))
    intr = intr.loc[evid_df.index]                  # enforce shared row order
    idx = evid_df.index
    N, J = len(idx), job_matrix.shape[0]

    # -- BM25 lexical facet channel (precomputed per-JD by jd_compile) -------
    # rank_bm25 is NOT imported here: the lexical scores are produced offline
    # in jd_compile.py and persisted, so the rank path stays lean.
    bm25_path = os.path.join(art_dir, "bm25_facets.parquet")
    if not os.path.exists(bm25_path):
        raise FileNotFoundError(
            "bm25_facets.parquet missing — run jd_compile.py after editing "
            "jd_profile.yaml")
    bm25_df = pd.read_parquet(bm25_path)
    missing_bm25 = [f"{sid}__bm25" for sid in SIG_IDS
                    if f"{sid}__bm25" not in bm25_df.columns]
    if missing_bm25:
        raise ValueError(
            f"bm25_facets.parquet missing columns {missing_bm25} — run "
            "jd_compile.py after editing jd_profile.yaml")
    bm25_df = bm25_df.reindex(idx)                   # align to the frame's order

    if jd_vecs.shape[0] != NSIG:
        raise ValueError(
            f"features: jd_vectors has {jd_vecs.shape[0]} rows but profile has "
            f"{NSIG} signals — re-run jd_compile after editing jd_profile.yaml")

    starts, ends = offsets[:, 0], offsets[:, 1]
    counts = ends - starts

    # ---- parse per-job meta into flat arrays (single pass) ----------------
    msince = np.zeros(J); dur = np.ones(J)
    chunks = [""] * J
    consulting_frac = np.zeros(N); only_consulting = np.zeros(N)
    product_frac = np.zeros(N); months_since_ic = np.full(N, 999.0)
    recent_is_mgmt = np.zeros(N); job_hop_rate = np.zeros(N)
    hop_tenure = method.hop_rate_tenure_months

    def _is_ic(tl):
        return any(k in tl for k in ICTOK) and not any(k in tl for k in MGTOK)

    metas_str = evid_df["jobs_meta"].values
    jobs_txt = evid_df["jobs_text"].values
    for i in range(N):
        s, e = starts[i], ends[i]
        if e <= s:
            continue
        metas = orjson.loads(metas_str[i])
        ctext = jobs_txt[i].split(SEP)
        n_cons = 0; n_prod = 0; n_hop = 0
        for k, m in enumerate(metas):
            r = s + k
            chunks[r] = ctext[k]
            d = m["d"] or 0
            dur[r] = max(1, d)
            if not m["cur"] and m["e"]:
                ed = _parse_date(m["e"])
                if ed is not None:
                    msince[r] = max(0.0, _months_between(ed, REF))
            comp = m["c"].lower(); ind = m["i"].lower(); tl = m["t"].lower()
            if any(x in comp for x in CONSULT): n_cons += 1
            if any(x in ind for x in PRODIND): n_prod += 1
            if d < hop_tenure: n_hop += 1
            if _is_ic(tl):
                months_since_ic[i] = min(months_since_ic[i], msince[r])
        nj = len(metas)
        consulting_frac[i] = n_cons / nj
        only_consulting[i] = float(n_cons == nj)
        product_frac[i] = n_prod / nj
        job_hop_rate[i] = n_hop / nj
        recent_is_mgmt[i] = float(any(k in metas[0]["t"].lower() for k in MGTOK))

    # ---- dense facet sims: ONE matmul + vectorized recency pooling --------
    simsJ = job_matrix @ jd_vecs.T                               # (J, NSIG)
    simsS = summ_matrix @ jd_vecs.T                              # (N, NSIG)
    rw = 0.5 ** (msince / facet_hl)
    W = rw * np.sqrt(dur)
    ne = counts > 0
    st_ne = starts[ne]

    recencywt = np.zeros((N, NSIG)); peak = np.zeros((N, NSIG))
    nhits = np.zeros((N, NSIG)); recent = np.zeros((N, NSIG))
    if J > 0 and ne.any():
        wsum = np.add.reduceat(W, st_ne)
        wsum[wsum == 0] = 1.0
        recencywt[ne] = np.add.reduceat(simsJ * W[:, None], st_ne) / wsum[:, None]
        peak[ne] = np.maximum.reduceat(simsJ, st_ne)
        nhits[ne] = np.add.reduceat((simsJ > facet_thr).astype(np.float32), st_ne)
        # most-recent job per candidate (argmin of msince within segment)
        cand_idx = np.repeat(np.arange(N)[ne], counts[ne])
        mr = pd.Series(msince).groupby(cand_idx).idxmin().values
        recent[ne] = simsJ[mr]

    # ---- evidence regexes per job chunk (ownership x context x recency) ---
    def _flags(rx):
        return np.fromiter((1.0 if rx.search(c) else 0.0 for c in chunks),
                           np.float64, J)
    f_int = _flags(INTERNAL_RE); f_own = _flags(OWNER_RE); f_sca = _flags(SCALE_RE)
    ctx = np.where(f_int > 0, EC["internal"], 1.0)
    own = np.where(f_own > 0, 1.0, EC["no_owner"])
    sca = np.where(f_sca > 0, 1.0, EC["no_scale"])
    rec_e = 0.5 ** (msince / evid_hl)
    mods = ctx * own * sca * rec_e

    evid = {}
    for s in EVID_SIGS:
        flat = _flags(s.evidence_re) * mods
        col = np.zeros(N)
        if J > 0 and ne.any():
            col[ne] = np.maximum.reduceat(flat, st_ne)
        evid[s.id] = col
    depth_flat = ((f_int == 0) & (f_own > 0) & (f_sca > 0)) * rec_e
    depth_bonus = np.zeros(N)
    if J > 0 and ne.any():
        depth_bonus[ne] = np.maximum.reduceat(depth_flat, st_ne)
    # evid_coverage = mean over the EVIDENCE signals only (those with a regex)
    if evid:
        evid_coverage = np.mean(np.column_stack(list(evid.values())), axis=1)
    else:
        evid_coverage = np.zeros(N)

    # ---- text-derived structured features ---------------------------------
    head = evid_df["headline_summary"].values
    cv_primary = np.zeros(N); domain_nlp_ratio = np.zeros(N)
    summary_years = np.full(N, np.nan)
    ai_corr = np.zeros(N); ai_claimed_n = np.zeros(N)
    n_skill_dur_exceed = np.zeros(N)
    skills_str = intr["skills_json"].values
    career_m = intr["career_months"].values
    SD = IR["skill_dur"]

    for i in range(N):
        jt = jobs_txt[i].replace(SEP, " ")
        cv_n = len(CV_RE.findall(jt)); nlp_n = len(NLP_RE.findall(jt))
        cv_primary[i] = float(cv_n >= 3 and cv_n > nlp_n)

        skills = orjson.loads(skills_str[i])
        names = [s["n"].lower() for s in skills]
        blob = (head[i] + " " + jt + " " + " ".join(names)).lower()
        nlp_c = sum(blob.count(tm) for tm in NLPT)
        cv_c = sum(blob.count(tm) for tm in CVT)
        domain_nlp_ratio[i] = (nlp_c + 1) / (nlp_c + cv_c + 2)

        m = YEARS_RE.search(head[i] or "")
        if m:
            summary_years[i] = float(m.group(1))

        evidence_l = (head[i] + " " + jt).lower()
        claimed = [nm for nm in names if any(tm in nm for tm in AITERMS)]
        supported = 0
        for nm in claimed:
            toks = [w for w in re.split(r"[^a-z]+", nm) if len(w) >= 4]
            if any(w[:5] in evidence_l for w in toks):
                supported += 1
        ai_corr[i] = supported / (len(claimed) + 1.0)
        ai_claimed_n[i] = len(claimed)

        cm = career_m[i]
        n_skill_dur_exceed[i] = sum(
            1 for s in skills
            if (s["d"] or 0) > cm * SD["ratio"] + SD["slack_months"])

    # ---- assessments (JD-relevant skill_assessment_scores) ----------------
    A = method.assessment
    assess_strength = np.zeros(N); n_assessed = np.zeros(N)
    for i, s in enumerate(intr["assessments_json"].values):
        sas = orjson.loads(s)
        rel = [v for k, v in sas.items() if DESIRED_RE.search(k)]
        if not rel:
            continue
        topk = sorted(rel, reverse=True)[:A["top_k"]]
        st = min(1.0, max(0.0, (float(np.mean(topk)) - A["score_floor"]) / A["score_span"]))
        assess_strength[i] = st * len(rel) / (len(rel) + A["count_shrink"])
        n_assessed[i] = len(rel)

    # ---- rules: yoe / location / integrity / availability / notice --------
    yoe = intr["yoe"].values.astype(float)
    peak_y = profile.role.peak_years; sigma_y = profile.role.sigma_years
    yoe_fit = np.exp(-((yoe - peak_y) ** 2) / (2 * sigma_y ** 2))
    avg_tenure = intr["avg_tenure_months"].values.astype(float)
    n_jobs = intr["n_jobs"].values.astype(float)
    HD = method.hopper_def
    hopper = ((n_jobs >= HD["min_jobs"]) &
              (avg_tenure < HD["mean_tenure_below_months"])).astype(float)

    # -- location ladder (mechanism in method.location_ladder; per-JD city
    #    lists in profile.locations) -----------------------------------------
    locs = intr["location"].values
    reloc = intr["willing_to_relocate"].values.astype(bool)
    LL = method.location_ladder
    LS = LL["scores"]
    LV4 = LL["v4"]
    pref_cities = profile.locations.preferred
    ok_cities = profile.locations.acceptable
    reloc_ok = profile.locations.relocation_acceptable
    pref = np.fromiter((any(x in l for x in pref_cities) for l in locs), bool, N)
    okc = np.fromiter((any(x in l for x in ok_cities) for l in locs), bool, N)
    t1c = np.fromiter((any(x in l for x in LL["india_tier1"]) for l in locs), bool, N)
    india = np.fromiter((any(x in l for x in LL["india_markers"]) for l in locs),
                        bool, N) | pref | okc
    # the JD's relocation stance gates whether candidate relocation earns credit
    eff_reloc = reloc & reloc_ok
    loc_fit2 = np.where(pref, LS["preferred"],
               np.where(okc, LS["ok_city"],
               np.where(india & eff_reloc, LS["india_relocate"],
               np.where(india, LS["india_no_reloc"],
               np.where(eff_reloc, LS["abroad_relocate"], LS["abroad_no_reloc"])))))
    # simple location_fit (v5 features.py flavor, kept as a model feature)
    location_fit = np.where(pref, 1.0, np.where(t1c, 0.8,
                   np.where(eff_reloc, 0.6, 0.2)))
    # v4 location adjustments
    remote_pref = intr["remote_pref"].values.astype(bool)
    no_reloc = ~reloc
    city_ok = pref | okc
    loc2 = np.where(np.isclose(loc_fit2, LS["india_no_reloc"]),
                    LV4["india_no_reloc_override"], loc_fit2).astype(float)
    capm = remote_pref & no_reloc & ~city_ok
    loc2[capm] = np.minimum(loc2[capm], LV4["remote_noreloc_offcity_cap"])
    loc2[remote_pref] *= LV4["remote_pref_damp"]

    # -- integrity ladder (vectorized; same compounding as v6) --------------
    max_role = intr["max_role_months"].values.astype(float)
    integ = np.ones(N)
    hard = [
        (career_m > 0) & (career_m / 12.0 > yoe + IR["career_sum_slack_years"]),
        max_role / 12.0 > yoe + IR["single_role_slack_years"],
        (career_m > 0) & (yoe * 12 > career_m * IR["yoe_vs_history"]["ratio"]
                          + IR["yoe_vs_history"]["slack_months"]),
        ~np.isnan(summary_years) & (np.abs(yoe - np.nan_to_num(summary_years))
                                    > IR["summary_yoe_tolerance"]),
        intr["n_expert_zero_dur"].values > 0,
        intr["n_expert"].values >= IR["too_many_expert"],
    ]
    for cond in hard:
        integ *= np.where(cond, IR["hard"], 1.0)
    soft_skill = (career_m > 0) & (n_skill_dur_exceed >= SD["min_count"])
    integ *= np.where(soft_skill,
                      1.0 - np.minimum(SD["max_pen"],
                                       SD["per_skill_pen"] * n_skill_dur_exceed),
                      1.0)
    sal_inv = (intr["salary_min"].values > intr["salary_max"].values).astype(float)
    integ *= np.where(sal_inv > 0, IR["salary_inverted"], 1.0)
    anach = intr["anach"].values
    integ *= np.where(anach == 1, IR["anachronism"], 1.0)
    la = intr["last_active_date"].values; su = intr["signup_date"].values
    la_lt_signup = np.fromiter(((1 if (a and b and a < b) else 0)
                                for a, b in zip(la, su)), np.int64, N)
    integ *= np.where(la_lt_signup == 1, IR["la_lt_signup"], 1.0)
    concur = intr["concurrent_deg"].values
    integ *= np.where(concur == 1, IR["concurrent_deg"], 1.0)

    # -- availability -------------------------------------------------------
    AV = method.availability; AW = AV["weights"]
    mi = np.fromiter(
        (_months_between(_parse_date(a), REF) if _parse_date(a) else
         AV["default_months_inactive"] for a in la), np.float64, N)
    mi = np.maximum(0.0, mi)
    rrr_raw = intr["recruiter_response_rate"].values.astype(float)
    rrr = np.maximum(0.0, rrr_raw)            # missing (-1) -> 0 for availability
    rrr_d = np.where(rrr_raw < 0, 1.0, rrr_raw)  # missing -> "not low" for dormancy
    raw = (AW["recency"] * np.exp(-mi / AV["inactive_halflife_months"]) +
           AW["response"] * rrr +
           AW["open_to_work"] * intr["open_to_work_flag"].values +
           AW["interview"] * intr["interview_completion_rate"].values +
           AW["completeness"] * intr["profile_completeness_score"].values / 100.0)
    avail_mult = AV["base"] + AV["span"] * raw
    DM = AV["dormancy"]
    dormant = ((mi > DM["months_inactive"]) & (rrr_d < DM["rrr_below"])).astype(int)
    low_rr = (rrr_d < DM["rrr_below"]).astype(int)
    avail = (avail_mult * np.where(dormant == 1, DM["damp"], 1.0)
             * np.where((low_rr == 1) & (dormant == 0), AV["low_rr_only_damp"], 1.0))

    # -- notice tiers -------------------------------------------------------
    days = intr["notice_period_days"].values.astype(float)
    notice_pen = np.empty(N); assigned = np.zeros(N, bool)
    for tier in method.notice_tiers:
        m_ = ~assigned & (days <= tier["max_days"])
        notice_pen[m_] = tier["mult"]; assigned |= m_
    notice_pen[~assigned] = method.notice_tiers[-1]["mult"]

    end_year = intr["min_edu_end_year"].values
    yoe_vs_grad_gap = np.where(end_year > 0, yoe - (REF.year - end_year), 0.0)

    # ---- assemble feature frame -------------------------------------------
    df = pd.DataFrame(index=idx)
    for fi, sid in enumerate(SIG_IDS):
        df[f"{sid}__recencywt"] = recencywt[:, fi]
        df[f"{sid}__peak"] = peak[:, fi]
        df[f"{sid}__nhits"] = nhits[:, fi]
        df[f"{sid}__recent"] = recent[:, fi]
        df[f"{sid}__summary"] = simsS[:, fi]
        df[f"{sid}__bm25"] = bm25_df[f"{sid}__bm25"].values   # lexical channel
    df["yoe"] = yoe; df["yoe_fit"] = yoe_fit
    df["n_jobs"] = n_jobs; df["avg_tenure_months"] = avg_tenure
    df["job_hop_rate"] = job_hop_rate
    df["consulting_frac"] = consulting_frac
    df["only_consulting"] = only_consulting
    df["product_frac"] = product_frac
    df["months_since_ic_role"] = months_since_ic
    df["recent_role_is_mgmt"] = recent_is_mgmt
    df["domain_nlp_ratio"] = domain_nlp_ratio
    df["ai_skill_corroboration"] = ai_corr
    df["ai_skills_claimed"] = ai_claimed_n
    df["location_fit"] = location_fit
    df["github_activity"] = np.maximum(0.0, intr["github_activity_score"].values)
    df["salary_inconsistent"] = sal_inv
    df["yoe_vs_grad_gap"] = yoe_vs_grad_gap
    for k, v in evid.items():
        df[f"evid_{k}"] = v
    df["evid_coverage"] = evid_coverage
    df["depth_bonus"] = depth_bonus
    df["assess_strength"] = assess_strength
    df["n_assessed_relevant"] = n_assessed
    df["cv_primary"] = cv_primary
    df["hopper"] = hopper
    df["loc_fit2"] = loc_fit2
    SIGNALS = ["profile_views_received_30d", "applications_submitted_30d",
               "avg_response_time_hours", "connection_count",
               "endorsements_received", "search_appearance_30d",
               "saved_by_recruiters_30d", "offer_acceptance_rate",
               "verified_email", "verified_phone", "linkedin_connected"]
    for k in SIGNALS:
        df[k] = intr[k].values.astype(float)
    # gate columns (rules.compute_rules consumes these; GATES excludes them
    # from the LGBM student in rank.py / train.py)
    df["integrity"] = integ
    df["availability_mult"] = avail
    df["notice_pen"] = notice_pen
    df["loc2_v4"] = loc2
    df["dormant"] = dormant; df["low_rr"] = low_rr
    df["anach"] = anach; df["la_lt_signup"] = la_lt_signup
    df["concurrent_deg"] = concur
    df["remote_pref"] = remote_pref.astype(int)
    df["no_reloc"] = no_reloc.astype(int)
    df["city_ok"] = city_ok.astype(int)
    df["notice_days"] = days
    return df
