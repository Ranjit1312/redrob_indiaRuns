"""
intrinsic.py — the JD-independent candidate-facts seam (v7 candidate 4).

The single place that knows the raw `candidates.jsonl` record shape: the
`profile` block, the 23 `redrob_signals`, `career_history`, `skills`,
`certifications`, `education`. It emits one typed table — `intrinsic.parquet`
— of facts that are TRUE OF THE CANDIDATE regardless of which JD we rank for.
Nothing downstream re-parses raw candidate JSON.

Ported verbatim from v6 `embed_candidates.py` (the intrinsic-row construction
and the anachronism world-fact). Deliberately JD-INDEPENDENT:
  - NO facet similarities, NO evidence regexes, NO JD lexicons, NO domain terms.
  - The two world-fact anachronisms (langchain<2022, llama<2023) are NOT JD
    knowledge: a cert dated before the tech existed is impossible for any role.
    (In v7 these same numbers also live in method_config.world_facts; this
    module keeps the v6 baked-in list so the candidate pass needs no Method.)

`extract_intrinsic(records) -> pd.DataFrame` is indexed by candidate_id and is
the EXACT input contract that rank.py / features.py read (see INTRINSIC_COLUMNS).
"""
from __future__ import annotations

import json
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Extended redrob signal columns kept verbatim (same set the v5/v6 teacher
# saved). These pass straight through as model features in rank.py.
# ---------------------------------------------------------------------------
SIGNAL_COLS = ["profile_views_received_30d", "applications_submitted_30d",
               "avg_response_time_hours", "connection_count",
               "endorsements_received", "search_appearance_30d",
               "saved_by_recruiters_30d", "offer_acceptance_rate"]
BOOL_COLS = ["verified_email", "verified_phone", "linkedin_connected"]

# world-fact anachronisms (NOT JD knowledge): (name_substring, first_year)
ANACHRONISMS = [("langchain", 2022), ("llama", 2023)]

# ---------------------------------------------------------------------------
# INTRINSIC_COLUMNS — the documented schema of intrinsic.parquet.
# Every column extract_intrinsic produces, in order, with what it carries and
# which rank.py / rules.py consumer reads it. (candidate_id is the index, not a
# column.) Keep this list in sync with the row dict built below.
# ---------------------------------------------------------------------------
INTRINSIC_COLUMNS = [
    # --- profile facts -----------------------------------------------------
    "current_title",            # str   : profile.current_title (reasoning)
    "yoe",                      # float : profile.years_of_experience (rules: yoe_fit, integrity)
    "location",                 # str   : "<location> <country>".lower() (rules: location)
    # --- career-shape facts (from career_history durations) ----------------
    "n_jobs",                   # int   : number of career_history entries (rules: hopper)
    "career_months",            # float : sum of job duration_months (integrity)
    "avg_tenure_months",        # float : mean job duration_months (rules: hopper)
    "max_role_months",          # float : max job duration_months (integrity: single_role)
    # --- skills / integrity inputs ----------------------------------------
    "skills_json",              # str(json): [{n,p,d}] name/proficiency/duration_months (features: domain ratio, ai corr, skill_dur)
    "n_expert",                 # int   : # skills with proficiency=="expert" (integrity: too_many_expert)
    "n_expert_zero_dur",        # int   : # expert skills with 0 duration (integrity: hard)
    "salary_min",               # float : expected_salary_range_inr_lpa.min (integrity: salary_inverted)
    "salary_max",               # float : expected_salary_range_inr_lpa.max (integrity: salary_inverted)
    "anach",                    # int   : 1 if any cert predates its tech (integrity: anachronism)
    "concurrent_deg",           # int   : 1 if overlapping full-time degrees (integrity: concurrent_deg)
    "min_edu_end_year",         # int   : min education end_year (rules: yoe_vs_grad_gap)
    # --- redrob signals: availability components + assessments -------------
    "assessments_json",         # str(json): skill_assessment_scores dict (features: assess_strength)
    "last_active_date",         # str   : signals.last_active_date (availability, integrity la_lt_signup)
    "signup_date",              # str   : signals.signup_date (integrity la_lt_signup)
    "recruiter_response_rate",  # float : -1 if missing else rate (availability / dormancy)
    "open_to_work_flag",        # int   : signals.open_to_work_flag (availability)
    "interview_completion_rate",# float : signals.interview_completion_rate (default 0.5) (availability)
    "profile_completeness_score",# float: signals.profile_completeness_score (default 50) (availability)
    "notice_period_days",       # int   : signals.notice_period_days (rules: notice_tiers)
    "github_activity_score",    # float : signals.github_activity_score (-1 missing) (model feature)
    "willing_to_relocate",      # int   : signals.willing_to_relocate (rules: location)
    "remote_pref",              # int   : 1 if preferred_work_mode=="remote" (rules: location v4)
] + SIGNAL_COLS + BOOL_COLS    # + 8 numeric signal pass-throughs + 3 bool flags


# ---------------------------------------------------------------------------
def _career_durs(jobs):
    return [j.get("duration_months") or 0 for j in jobs]


def extract_intrinsic(records: "Iterable[dict]") -> "pd.DataFrame":
    """Build the JD-independent intrinsic facts table from raw candidate dicts.

    Parameters
    ----------
    records : iterable of parsed candidate JSON objects (one per candidate),
              each with keys: candidate_id, profile, career_history,
              redrob_signals, and optionally skills / certifications /
              education.

    Returns
    -------
    pandas.DataFrame indexed by candidate_id, columns == INTRINSIC_COLUMNS.
    Pure: no I/O, no JD knowledge. Mirrors v6 embed_candidates.py.
    """
    rows = []
    for c in records:
        cid = c["candidate_id"]
        p = c.get("profile") or {}
        jobs = c.get("career_history") or []
        sig = c.get("redrob_signals") or {}
        skills = c.get("skills", []) or []

        durs = _career_durs(jobs)
        career_m = float(sum(durs))

        certs = c.get("certifications") or []
        anach = any(name in (ct.get("name") or "").lower()
                    and (ct.get("year") or 9999) < yr
                    for ct in certs for name, yr in ANACHRONISMS)

        edu = [(e.get("start_year"), e.get("end_year"))
               for e in (c.get("education") or [])]
        edu = [e for e in edu if e[0] and e[1]]
        concurrent = any(edu[i] == edu[j] for i in range(len(edu))
                         for j in range(i + 1, len(edu)))
        end_years = [e[1] for e in edu]

        sal = sig.get("expected_salary_range_inr_lpa") or {}

        row = {
            "candidate_id": cid,
            # profile facts
            "current_title": p.get("current_title") or "",
            "yoe": float(p.get("years_of_experience") or 0),
            "location": ((p.get("location") or "") + " " +
                         (p.get("country") or "")).lower(),
            # career-shape facts
            "n_jobs": len(jobs),
            "career_months": career_m,
            "avg_tenure_months": float(np.mean(durs)) if durs else 0.0,
            "max_role_months": float(max(durs)) if durs else 0.0,
            # skills / integrity inputs
            "skills_json": json.dumps(
                [{"n": s.get("name") or "", "p": s.get("proficiency") or "",
                  "d": s.get("duration_months") or 0} for s in skills]),
            "n_expert": int(sum(s.get("proficiency") == "expert" for s in skills)),
            "n_expert_zero_dur": int(sum(
                s.get("proficiency") == "expert"
                and (s.get("duration_months", 1) or 0) == 0 for s in skills)),
            "salary_min": float(sal.get("min") or 0.0),
            "salary_max": float(sal.get("max") or 0.0),
            "anach": int(anach),
            "concurrent_deg": int(concurrent),
            "min_edu_end_year": int(min(end_years)) if end_years else 0,
            # redrob signals (availability components + assessments)
            "assessments_json": json.dumps(
                sig.get("skill_assessment_scores") or {}),
            "last_active_date": sig.get("last_active_date") or "",
            "signup_date": sig.get("signup_date") or "",
            # -1 = missing (rank.py recovers both availability & dormancy senses)
            "recruiter_response_rate": float(
                -1 if sig.get("recruiter_response_rate") is None
                else sig["recruiter_response_rate"]),
            "open_to_work_flag": int(bool(sig.get("open_to_work_flag"))),
            "interview_completion_rate": float(
                0.5 if sig.get("interview_completion_rate") is None
                else sig["interview_completion_rate"]),
            "profile_completeness_score": float(
                50 if sig.get("profile_completeness_score") is None
                else sig["profile_completeness_score"]),
            "notice_period_days": int(sig.get("notice_period_days") or 0),
            "github_activity_score": float(
                sig.get("github_activity_score", -1)
                if sig.get("github_activity_score") is not None else -1),
            "willing_to_relocate": int(bool(sig.get("willing_to_relocate"))),
            "remote_pref": int(sig.get("preferred_work_mode") == "remote"),
        }
        for k in SIGNAL_COLS:
            row[k] = float(sig.get(k, -1) if sig.get(k) is not None else -1)
        for k in BOOL_COLS:
            row[k] = int(bool(sig.get(k)))
        rows.append(row)

    df = pd.DataFrame(rows, columns=["candidate_id"] + INTRINSIC_COLUMNS)
    df = df.set_index("candidate_id")
    return df
