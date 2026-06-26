"""
test_swap_jd.py — proves a JD swap is CONFIG-ONLY (no code change).

MODELS NOT REQUIRED. We copy the real jd_profile.yaml into a tmp dir and:

  (a) load it and assert the signal id/weight/regex set is what the worked
      example declares;
  (b) write a MODIFIED variant that drops the `llm_ft` signal AND flips
      red_flags.cv_primary.enabled -> false, then assert via profile.load that
      signal_ids changed and red_flag_enabled('cv_primary') is now False — a JD
      retarget that touched only the yaml;
  (c) assert compute_rules on a tiny synthetic frame respects the cv_primary
      toggle (the disabled flag stops damping fit), reusing the pattern from
      tests/test_rules.py.

This is the regression that locks the seam: the SAME code yields different
signals, reasoning inputs and gates purely from the edited config.
"""
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from redrob_ranker import profile as P
from redrob_ranker import rules as R

HERE = os.path.dirname(os.path.abspath(__file__))
JD_PATH = os.path.normpath(os.path.join(HERE, "..", "jd", "jd_profile.yaml"))
METHOD_PATH = os.path.normpath(os.path.join(HERE, "..", "jd", "method_config.yaml"))


def _load_raw():
    with open(JD_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write(tmp_path, jd_dict, name="jd_profile.yaml"):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(jd_dict, f, sort_keys=False)
    return str(p)


def _frame(profile, n=1, **overrides):
    """Synthetic feature frame with every column compute_rules reads."""
    base = {
        "yoe_fit": 0.8, "domain_nlp_ratio": 0.7,
        "evid_coverage": 0.5, "depth_bonus": 0.4, "assess_strength": 0.3,
        "ai_skills_claimed": 0.0, "ai_skill_corroboration": 0.6,
        "cv_primary": 0.0, "hopper": 0.0, "only_consulting": 0.0,
        "months_since_ic_role": 2.0,
        "loc2_v4": 0.9,
        "integrity": 1.0, "availability_mult": 1.0, "notice_pen": 1.0,
    }
    data = {}
    for sid in profile.signal_ids():
        data[f"{sid}__recencywt"] = np.linspace(0.2, 0.9, n)
        data[f"{sid}__bm25"] = np.linspace(0.1, 0.8, n)   # BM25 lexical channel
    for k, v in base.items():
        data[k] = np.full(n, float(v))
    for k, v in overrides.items():
        data[k] = np.asarray(v, dtype=float)
    idx = pd.Index([f"c{i}" for i in range(n)], name="candidate_id")
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# (a) the unmodified worked-example JD loads with the expected seam
# ---------------------------------------------------------------------------
def test_baseline_jd_signal_set(tmp_path):
    raw = _load_raw()
    jd = _write(tmp_path, raw)
    prof, _ = P.load(jd, METHOD_PATH)

    assert prof.signal_ids() == [
        "ranking", "retrieval", "vectordb", "evaluation", "applied_ml", "llm_ft"]

    by_id = {s.id: s for s in prof.signals}
    assert by_id["ranking"].dense_weight == pytest.approx(0.28)
    assert by_id["retrieval"].dense_weight == pytest.approx(0.22)
    assert by_id["llm_ft"].dense_weight == pytest.approx(0.0)

    # the evidence (regex-carrying) signals
    assert [s.id for s in prof.evidence_signals()] == [
        "ranking", "retrieval", "vectordb", "evaluation"]
    # llm_ft / applied_ml are model-only (no compiled regex)
    assert by_id["llm_ft"].evidence_re is None
    assert by_id["applied_ml"].evidence_re is None
    # a known evidence match still fires
    assert by_id["vectordb"].evidence_re.search("we used FAISS for recall")

    assert prof.red_flag_enabled("cv_primary") is True


# ---------------------------------------------------------------------------
# (b) a CONFIG-ONLY swap: drop a signal + disable a red flag
# ---------------------------------------------------------------------------
def test_config_only_swap_changes_seam(tmp_path):
    raw = _load_raw()
    base_ids = [s["id"] for s in raw["signals"]]
    assert "llm_ft" in base_ids

    # edit ONLY the yaml: drop llm_ft, disable cv_primary
    raw["signals"] = [s for s in raw["signals"] if s["id"] != "llm_ft"]
    raw["red_flags"]["cv_primary"]["enabled"] = False
    swapped = _write(tmp_path, raw, name="jd_profile_swapped.yaml")

    prof, _ = P.load(swapped, METHOD_PATH)

    # signal set changed — llm_ft gone, the rest preserved in order
    assert "llm_ft" not in prof.signal_ids()
    assert prof.signal_ids() == [
        "ranking", "retrieval", "vectordb", "evaluation", "applied_ml"]
    # the red-flag toggle flipped
    assert prof.red_flag_enabled("cv_primary") is False
    # other flags untouched
    assert prof.red_flag_enabled("job_hopper") is True


# ---------------------------------------------------------------------------
# (c) compute_rules respects the toggle on a synthetic frame
# ---------------------------------------------------------------------------
def test_compute_rules_respects_toggle(tmp_path):
    method_obj = P.load(JD_PATH, METHOD_PATH)[1]

    # baseline: cv_primary enabled -> a cv_primary row gets damped
    raw = _load_raw()
    on_path = _write(tmp_path, raw, name="jd_on.yaml")
    prof_on, _ = P.load(on_path, METHOD_PATH)
    df_on = _frame(prof_on, n=1, cv_primary=[1.0])
    fit_on = R.compute_rules(df_on, prof_on, method_obj).fit[0]

    # swapped: cv_primary disabled -> same row is NOT damped, fit rises
    raw2 = _load_raw()
    raw2["red_flags"]["cv_primary"]["enabled"] = False
    off_path = _write(tmp_path, raw2, name="jd_off.yaml")
    prof_off, _ = P.load(off_path, METHOD_PATH)
    df_off = _frame(prof_off, n=1, cv_primary=[1.0])
    fit_off = R.compute_rules(df_off, prof_off, method_obj).fit[0]

    assert fit_off > fit_on
    # removing the damp multiplies fit back up by exactly the cv_primary factor
    assert fit_on == pytest.approx(
        fit_off * method_obj.damps["cv_primary"], rel=1e-9)
