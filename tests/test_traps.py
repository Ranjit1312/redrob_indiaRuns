"""
Challenge-specific behaviour: the top-100 must be free of the disqualifier /
honeypot / unavailability traps the dataset plants (job_description.docx +
redrob_signals_doc.docx; rules encoded in build_final_features.py and the audit
loop in versions/audit_report.md).

These are VERSION-GATED on RANK_VERSION because the audit loop introduced the
fixes incrementally — that progression is the point. A trap test that is skipped
for v1 and passes for v5 documents exactly which iteration closed the hole:
  * YoE-inflation honeypot   -> integrity ladder, v2+
  * CV-primary disqualifier  -> cv_primary damp, demoted out of top-100 by v4
  * dormant + unresponsive   -> availability x0.5 (B1), v4+
  * certification anachronism -> integrity x0.30 (B2), v4+
"""
from datetime import date

import pytest

PIPELINE_REF_DATE = date(2026, 6, 11)          # matches build_final_features.py
CV_PRIMARY_DISQUALIFIER = "CAND_0039983"        # JD disqualifier (computer-vision career)


def _rank_version():
    import os
    try:
        return int(os.environ.get("RANK_VERSION", "5"))
    except ValueError:
        return 5


def _months_inactive(rec):
    la = (rec.get("redrob_signals") or {}).get("last_active_date")
    if not la:
        return 12.0
    d = date(int(la[:4]), int(la[5:7]), int(la[8:10]))
    return (PIPELINE_REF_DATE - d).days / 30.44


def _is_dormant(rec):
    s = rec.get("redrob_signals") or {}
    return _months_inactive(rec) > 6 and s.get("recruiter_response_rate", 1) < 0.2


def _has_anachronistic_cert(rec):
    for ct in rec.get("certifications") or []:
        name = (ct.get("name") or "").lower()
        yr = ct.get("year") or 9999
        if "langchain" in name and yr < 2022:
            return True
        if "llama" in name and yr < 2023:
            return True
    return False


def _yoe_inflated(rec):
    """The v2 honeypot fingerprint: stated YoE far exceeds real career history."""
    p = rec.get("profile") or {}
    yoe = p.get("years_of_experience") or 0
    career_m = sum((j.get("duration_months") or 0) for j in (rec.get("career_history") or []))
    return yoe * 12 > career_m * 1.6 + 18


def test_no_yoe_inflation_honeypot_in_top100(top_records):
    if _rank_version() < 2:
        pytest.skip("YoE-inflation honeypot is closed by the v2 integrity ladder")
    bad = [cid for cid, rec in top_records.items() if _yoe_inflated(rec)]
    assert not bad, f"YoE-inflation honeypots in top-100: {bad[:5]}"


def test_cv_primary_disqualifier_not_in_top100(submission_rows):
    if _rank_version() < 4:
        pytest.skip("CV-primary candidate is only demoted out of top-100 by v4")
    ids = {r["candidate_id"] for r in submission_rows["rows"]}
    assert CV_PRIMARY_DISQUALIFIER not in ids, \
        f"{CV_PRIMARY_DISQUALIFIER} (CV-primary disqualifier) is in the top-100"


def test_no_dormant_unresponsive_in_top100(top_records):
    if _rank_version() < 4:
        pytest.skip("dormant+unresponsive availability gate (B1) lands in v4")
    bad = [cid for cid, rec in top_records.items() if _is_dormant(rec)]
    assert not bad, f"dormant+unresponsive candidates in top-100: {bad[:5]}"


def test_no_certification_anachronism_in_top100(top_records):
    if _rank_version() < 4:
        pytest.skip("certification-anachronism integrity rule (B2) lands in v4")
    bad = [cid for cid, rec in top_records.items() if _has_anachronistic_cert(rec)]
    assert not bad, f"anachronistic-certification candidates in top-100: {bad[:5]}"
