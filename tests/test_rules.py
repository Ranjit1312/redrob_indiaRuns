"""
test_rules.py — the single deterministic engine, on a synthetic frame.

No heavy artifacts / models. We build a handful of rows with exactly the
columns compute_rules reads, load the real Profile/Method, and assert the
contract: finite shapes, gated product, red-flag toggles, integrity kill.
"""
import os
import copy

import numpy as np
import pandas as pd
import pytest

from redrob_ranker import profile as P
from redrob_ranker import rules as R

HERE = os.path.dirname(os.path.abspath(__file__))
JD_PATH = os.path.normpath(os.path.join(HERE, "..", "jd", "jd_profile.yaml"))
METHOD_PATH = os.path.normpath(os.path.join(HERE, "..", "jd", "method_config.yaml"))


@pytest.fixture(scope="module")
def loaded():
    return P.load(JD_PATH, METHOD_PATH)


def _frame(profile, n=4, **overrides):
    """Build a synthetic feature frame with every required column."""
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


def test_shapes_and_finite(loaded):
    prof, method = loaded
    df = _frame(prof, n=5)
    res = R.compute_rules(df, prof, method)
    for arr in (res.fit, res.integrity, res.availability, res.notice_pen,
                res.loc2, res.final_rules):
        assert arr.shape == (5,)
        assert np.all(np.isfinite(arr))
    # final in [0,1]: mm(fit) in [0,1] times gates each <= 1
    assert res.final_rules.min() >= 0.0
    assert res.final_rules.max() <= 1.0 + 1e-9


def test_final_is_gated_product(loaded):
    prof, method = loaded
    df = _frame(prof, n=4,
                integrity=[1.0, 0.5, 1.0, 0.05],
                availability_mult=[1.0, 0.9, 0.8, 1.0],
                notice_pen=[1.0, 1.0, 0.9, 0.85])
    res = R.compute_rules(df, prof, method)
    expected = R.mm(res.fit) * res.integrity * res.availability * res.notice_pen
    np.testing.assert_allclose(res.final_rules, expected)


def test_integrity_hard_fail_drives_near_zero(loaded):
    prof, method = loaded
    # one row clean, one row with hard integrity fail (0.05)
    df = _frame(prof, n=2, integrity=[1.0, method.integrity["hard"]])
    res = R.compute_rules(df, prof, method)
    # the hard-fail row's final must be heavily suppressed vs its mm(fit) share
    assert res.final_rules[1] <= R.mm(res.fit)[1] * method.integrity["hard"] + 1e-9
    assert res.final_rules[1] < 0.06


def test_disabling_red_flag_raises_score(loaded):
    prof, method = loaded
    # a row that trips cv_primary; everything else identical
    df = _frame(prof, n=1, cv_primary=[1.0])

    res_on = R.compute_rules(df, prof, method)
    fit_on = res_on.fit[0]

    # disable the cv_primary red flag in a copy of the profile
    rf = dict(prof.red_flags)
    rf["cv_primary"] = P.RedFlag(name="cv_primary", enabled=False)
    prof_off = _replace_profile(prof, red_flags=rf)
    res_off = R.compute_rules(df, prof_off, method)
    fit_off = res_off.fit[0]

    # damp removed -> fit rises by exactly the cv_primary damp factor
    assert fit_off > fit_on
    assert fit_on == pytest.approx(fit_off * method.damps["cv_primary"], rel=1e-9)


def test_only_consulting_damp_toggles(loaded):
    prof, method = loaded
    df = _frame(prof, n=1, only_consulting=[1.0])
    on = R.compute_rules(df, prof, method).fit[0]
    rf = dict(prof.red_flags)
    rf["only_consulting"] = P.RedFlag(name="only_consulting", enabled=False)
    off = R.compute_rules(df, _replace_profile(prof, red_flags=rf), method).fit[0]
    assert off > on
    # on = off * (1 - 0.30)
    assert on == pytest.approx(off * (1 - method.damps["only_consulting"]), rel=1e-9)


def test_missing_column_raises(loaded):
    prof, method = loaded
    df = _frame(prof, n=2)
    df = df.drop(columns=["evid_coverage"])
    with pytest.raises(KeyError):
        R.compute_rules(df, prof, method)


# dataclasses.replace works on frozen dataclasses; provide a tiny shim name
def _replace_profile(prof, **changes):
    import dataclasses
    return dataclasses.replace(prof, **changes)
