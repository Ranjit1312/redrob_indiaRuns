"""
rules.py — the single deterministic rules engine (v7 candidate 2).

ONE place the composite + gate formula lives. v5 had it copied in three files
(build_rule_features, build_final_features, rank); v7 consolidates it here and
both the precompute label step and rank.py call `compute_rules`.

NUMERICALLY FAITHFUL TO THE (UPDATED) ROOT `rank.py` — section "3. deterministic
composite". The composite is:

    # additive channels: dense semantic + BM25 lexical + ownership depth
    fit = aw.dense*dense_fit + aw.lexical*lex_fit + aw.depth*depth_bonus
    # (1) evidence GATE (necessary condition)
    fit *= evidence_gate.floor + evidence_gate.span*evid_coverage
    # (2) claim-consistency DISCOUNT (neutral if no AI skills claimed)
    fit *= where(ai_skills_claimed==0, 1, claim.base + claim.span*ai_corr)
    # (3) assessment BONUS (coverage-gated)
    fit *= 1 + assess_bonus.weight * assess_strength*min(1, evid_coverage/cov0)
    # (4) recent-coding ladder over months_since_ic_role
    fit *= recency_ladder bucket
    # (5) experience band
    fit *= experience_band.base + experience_band.span*yoe_fit
    # red-flag damps (cv_primary / hopper / only_consulting) + location
    final_rules = mm(fit) * integrity * availability * notice_pen   (gates, a=0.2 blend outside)

where dense_fit = Σ signal.dense_weight*<id>__recencywt + yoe_fit_weight*yoe_fit
+ domain_ratio_weight*domain_nlp_ratio, and lex_fit = mean over ALL signals of
<id>__bm25 (the BM25 lexical channel; produced per-JD by jd_compile, loaded by
features.py).

ALL constants come from `profile` (per-JD) and `method` (JD-stable). Nothing is
hardcoded. Pure & vectorized; numpy/pandas only.

Input contract — `features` (a pandas.DataFrame indexed by candidate_id) MUST
provide the columns in `required_columns(profile)` (REQUIRED_COLUMNS plus, per
signal, `<id>__recencywt` and `<id>__bm25`). features.py produces them; tests
build a minimal synthetic frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
def mm(x):
    """Min-max normalize to [0,1]. Copied verbatim from root rank.py."""
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (np.ptp(x) + 1e-12)


@dataclass(frozen=True)
class RuleResult:
    fit: np.ndarray            # pre-gate composite (raw, not normalized)
    integrity: np.ndarray
    availability: np.ndarray
    notice_pen: np.ndarray
    loc2: np.ndarray           # v4-adjusted location score in [0,1]
    final_rules: np.ndarray    # mm(fit) * integrity * availability * notice_pen


# Non-signal columns compute_rules reads off `features`. (Per-signal
# __recencywt and __bm25 columns are added dynamically from profile.signals.)
REQUIRED_COLUMNS = [
    "yoe_fit", "domain_nlp_ratio", "depth_bonus", "evid_coverage",
    "ai_skills_claimed", "ai_skill_corroboration", "assess_strength",
    "cv_primary", "hopper", "only_consulting", "months_since_ic_role",
    "loc2_v4",
    "integrity", "availability_mult", "notice_pen",
]


def required_columns(profile) -> list:
    """The full set of feature-frame columns compute_rules will read for this
    profile (per-signal __recencywt + __bm25, then REQUIRED_COLUMNS)."""
    cols = []
    for sid in profile.signal_ids():
        cols.append(f"{sid}__recencywt")
        cols.append(f"{sid}__bm25")
    return cols + list(REQUIRED_COLUMNS)


def compute_rules(features: "pd.DataFrame", profile, method) -> RuleResult:
    """Deterministic composite + gated product, faithful to root rank.py.

    final_rules = mm(fit) * integrity * availability * notice_pen
    """
    df = features
    missing = [c for c in required_columns(profile) if c not in df.columns]
    if missing:
        raise KeyError(f"compute_rules: features frame missing columns: {missing}")

    N = len(df)

    def col(name):
        return df[name].values.astype(np.float64)

    # -- gate columns (already materialized by features.py; section 7 of v6) -
    integ = col("integrity")
    avail = col("availability_mult")
    notice_pen = col("notice_pen")
    loc2 = col("loc2_v4")

    # -- composite inputs ---------------------------------------------------
    yoe_fit = col("yoe_fit")
    domain_nlp_ratio = col("domain_nlp_ratio")
    depth_bonus = col("depth_bonus")
    evid_coverage = col("evid_coverage")
    ai_claimed = col("ai_skills_claimed")
    ai_corr = col("ai_skill_corroboration")
    assess_strength = col("assess_strength")
    cv_primary = col("cv_primary")
    hopper = col("hopper")
    only_consulting = col("only_consulting")
    months_since_ic = col("months_since_ic_role")

    # -- dense_fit: Σ per-signal dense_weight * <id>__recencywt -------------
    # (llm_ft has dense_weight 0.0 -> contributes nothing, matching root which
    #  omitted it from dense_fit.)
    dense_fit = np.zeros(N, dtype=np.float64)
    for s in profile.signals:
        dense_fit += s.dense_weight * col(f"{s.id}__recencywt")
    de = profile.dense_extras
    dense_fit = (dense_fit
                 + float(de.get("yoe_fit_weight", 0.0)) * yoe_fit
                 + float(de.get("domain_ratio_weight", 0.0)) * domain_nlp_ratio)

    # -- lex_fit: mean of the BM25 lexical channel over ALL signals ---------
    bm25_cols = np.column_stack([col(f"{s.id}__bm25") for s in profile.signals])
    lex_fit = bm25_cols.mean(axis=1)

    # -- additive channels (dense + lexical + depth) ------------------------
    AW = method.additive_weights
    fit = (AW["dense"] * dense_fit
           + AW["lexical"] * lex_fit
           + AW["depth"] * depth_bonus)

    # -- (1) evidence GATE --------------------------------------------------
    EG = method.evidence_gate
    g_evid = EG["floor"] + EG["span"] * evid_coverage
    # -- (2) claim-consistency DISCOUNT (neutral when no AI skills claimed) --
    CC = method.claim_consistency
    m_claim = np.where(ai_claimed == 0, 1.0, CC["base"] + CC["span"] * ai_corr)
    # -- (3) assessment BONUS (coverage-gated) ------------------------------
    AB = method.assessment_bonus
    assess_corr = assess_strength * np.minimum(1.0, evid_coverage / AB["full_credit_cov"])
    m_assess = 1.0 + AB["weight"] * assess_corr
    fit = fit * g_evid * m_claim * m_assess

    # -- (4) recent-coding ladder over months_since_ic_role -----------------
    # nested-where equivalent: iterate buckets ascending by gt so the highest
    # threshold the candidate exceeds wins (matches root's nested np.where).
    # Gated by the stale_ic_role red flag (default-on = root behavior).
    if profile.red_flag_enabled("stale_ic_role"):
        recency_mult = np.ones(N)
        for tier in sorted(method.recency_ladder, key=lambda t: t["gt"]):
            recency_mult = np.where(months_since_ic > tier["gt"],
                                    tier["mult"], recency_mult)
        fit = fit * recency_mult

    # -- (5) experience band ------------------------------------------------
    EB = method.experience_band
    fit = fit * (EB["base"] + EB["span"] * yoe_fit)

    # -- red-flag damps (apply only when the JD enables them) ---------------
    D = method.damps
    if profile.red_flag_enabled("cv_primary"):
        fit = fit * np.where(cv_primary == 1, D["cv_primary"], 1.0)
    if profile.red_flag_enabled("job_hopper"):
        fit = fit * np.where(hopper == 1, D["hopper"], 1.0)
    if profile.red_flag_enabled("only_consulting"):
        fit = fit * (1.0 - D["only_consulting"] * only_consulting)

    # -- location damp (always-on geometry; loc2 carries the per-JD geo) ----
    fit = fit * (D["loc_base"] + D["loc_span"] * loc2)
    fit = fit * np.where(loc2 <= D["loc_floor_threshold"], D["loc_floor_damp"], 1.0)

    # -- gated product ------------------------------------------------------
    final_rules = mm(fit) * integ * avail * notice_pen

    return RuleResult(fit=fit, integrity=integ, availability=avail,
                      notice_pen=notice_pen, loc2=loc2, final_rules=final_rules)
