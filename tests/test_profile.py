"""
test_profile.py — the JD seam loads, validates, and compiles regexes.

Builds broken configs in a tmp dir from the REAL yaml so we test the actual
schema, not a toy. Works with or without the optional `jsonschema` dep (load()
falls back to a hand-rolled check that raises the same ValueError shape).
"""
import os
import re
import copy

import pytest
import yaml

from redrob_ranker import profile as P

HERE = os.path.dirname(os.path.abspath(__file__))
JD_PATH = os.path.normpath(os.path.join(HERE, "..", "jd", "jd_profile.yaml"))
METHOD_PATH = os.path.normpath(os.path.join(HERE, "..", "jd", "method_config.yaml"))


def _load_raw():
    with open(JD_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write(tmp_path, jd_dict):
    p = tmp_path / "jd_profile.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(jd_dict, f)
    return str(p)


# ---------------------------------------------------------------------------
def test_valid_load_succeeds_and_exposes_signals():
    prof, method = P.load(JD_PATH, METHOD_PATH)

    # expected signal ids in order
    assert prof.signal_ids() == [
        "ranking", "retrieval", "vectordb", "evaluation", "applied_ml", "llm_ft"]

    # dense weights from jd_profile.signals[].dense_weight
    by_id = {s.id: s for s in prof.signals}
    assert by_id["ranking"].dense_weight == pytest.approx(0.28)
    assert by_id["retrieval"].dense_weight == pytest.approx(0.22)
    assert by_id["llm_ft"].dense_weight == pytest.approx(0.0)

    # evidence_signals = only those with a regex (the 4 evidence axes)
    ev_ids = [s.id for s in prof.evidence_signals()]
    assert ev_ids == ["ranking", "retrieval", "vectordb", "evaluation"]

    # dense_extras
    assert prof.dense_extras["yoe_fit_weight"] == pytest.approx(0.08)
    assert prof.dense_extras["domain_ratio_weight"] == pytest.approx(0.10)

    # role
    assert prof.role.peak_years == pytest.approx(7.0)
    assert prof.role.sigma_years == pytest.approx(2.5)
    assert prof.role.notice_preference_days == 90

    # red flags
    assert prof.red_flag_enabled("cv_primary") is True
    assert prof.red_flag_enabled("job_hopper") is True


def test_regexes_are_compiled():
    prof, method = P.load(JD_PATH, METHOD_PATH)

    # signal evidence regexes compiled (None only for model-only axes)
    for s in prof.signals:
        if s.evidence_regex is None:
            assert s.evidence_re is None
        else:
            assert isinstance(s.evidence_re, re.Pattern)
    # llm_ft / applied_ml are model-only
    by_id = {s.id: s for s in prof.signals}
    assert by_id["applied_ml"].evidence_re is None
    assert by_id["llm_ft"].evidence_re is None
    # a known evidence match
    assert by_id["vectordb"].evidence_re.search("we used FAISS for recall")

    # domain regexes compiled
    assert isinstance(prof.domain.in_domain_re, re.Pattern)
    assert isinstance(prof.domain.out_of_domain_re, re.Pattern)
    assert prof.domain.out_of_domain_re.search("trained a YOLO detector")

    # relevant_skill regex compiled
    assert isinstance(prof.relevant_skill_re, re.Pattern)
    assert prof.relevant_skill_re.search("Vector Search")

    # method regexes compiled
    assert isinstance(method.years_re, re.Pattern)
    for k in ("internal", "owner", "scale"):
        assert isinstance(method.context_re[k], re.Pattern)


def test_missing_required_field_names_the_field(tmp_path):
    raw = _load_raw()
    del raw["role"]["notice_preference_days"]
    bad = _write(tmp_path, raw)
    with pytest.raises(ValueError) as ei:
        P.load(bad, METHOD_PATH)
    assert "notice_preference_days" in str(ei.value)


def test_bad_type_names_the_field(tmp_path):
    raw = _load_raw()
    raw["signals"][0]["dense_weight"] = "heavy"   # should be a number
    bad = _write(tmp_path, raw)
    with pytest.raises(ValueError) as ei:
        P.load(bad, METHOD_PATH)
    assert "dense_weight" in str(ei.value)


def test_missing_signals_field(tmp_path):
    raw = _load_raw()
    del raw["signals"]
    bad = _write(tmp_path, raw)
    with pytest.raises(ValueError) as ei:
        P.load(bad, METHOD_PATH)
    assert "signals" in str(ei.value)


def test_bad_signal_id_pattern(tmp_path):
    raw = _load_raw()
    raw["signals"][0]["id"] = "Bad Id!"   # violates ^[a-z][a-z0-9_]*$
    bad = _write(tmp_path, raw)
    with pytest.raises(ValueError) as ei:
        P.load(bad, METHOD_PATH)
    assert "id" in str(ei.value).lower()


def test_invalid_regex_names_the_field(tmp_path):
    raw = _load_raw()
    raw["signals"][1]["evidence_regex"] = "(unbalanced"
    bad = _write(tmp_path, raw)
    with pytest.raises(ValueError) as ei:
        P.load(bad, METHOD_PATH)
    assert "evidence_regex" in str(ei.value)
