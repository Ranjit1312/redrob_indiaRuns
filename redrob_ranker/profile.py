"""
profile.py — the JD-seam adapter (v7 candidate 1).

Parses + validates the two config seams and hands the rest of the pipeline two
frozen, regex-precompiled objects:

    Profile  <- jd/jd_profile.yaml   (the per-JD knowledge; schema-validated)
    Method   <- jd/method_config.yaml (the JD-stable mechanism / all numerics)

Nothing else in the pipeline parses these yamls. Every regex (signal evidence,
domain in/out, relevant-skill, method context/years) is compiled exactly once
here and exposed on the returned objects.

Import-light by design: pyyaml + re + dataclasses, and OPTIONALLY jsonschema
(if importable). When jsonschema is absent, load() falls back to a hand-rolled
check of the schema's required fields/types so validation still works.

CLI:
    python -m redrob_ranker.profile --check jd/jd_profile.yaml
    (prints OK or a precise ValueError message; exits 0 / 1)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

# Path to the JSON schema for jd_profile.yaml (sibling jd/ dir).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_JD = os.path.normpath(os.path.join(_HERE, "..", "jd", "jd_profile.yaml"))
_DEFAULT_METHOD = os.path.normpath(os.path.join(_HERE, "..", "jd", "method_config.yaml"))
_SCHEMA_PATH = os.path.normpath(os.path.join(_HERE, "..", "jd", "jd_profile.schema.json"))


# ---------------------------------------------------------------------------
# Dataclasses (the pinned interface from ARCHITECTURE.md)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Signal:
    id: str
    label: str
    query: str
    evidence_regex: Optional[str]          # raw pattern (None = model-only axis)
    dense_weight: float
    evidence_re: Optional["re.Pattern"] = None   # compiled (None if no regex)


@dataclass(frozen=True)
class RoleSpec:
    title: str
    company: str
    domain: str
    summary: str
    min_years: Optional[float]
    max_years: Optional[float]
    peak_years: float
    sigma_years: float
    notice_preference_days: int


@dataclass(frozen=True)
class LocSpec:
    preferred: list                        # list[str], lowercase city tokens
    acceptable: list                       # list[str]
    relocation_acceptable: bool
    remote_acceptable: bool


@dataclass(frozen=True)
class DomainSpec:
    in_domain_terms: list
    out_of_domain_terms: list
    in_domain_regex: str
    out_of_domain_regex: str
    in_domain_re: "re.Pattern" = None
    out_of_domain_re: "re.Pattern" = None


@dataclass(frozen=True)
class RedFlag:
    name: str
    enabled: bool


@dataclass(frozen=True)
class Profile:
    """Parsed + validated jd_profile.yaml."""
    schema_version: int
    role: RoleSpec
    locations: LocSpec
    signals: list                          # list[Signal]
    dense_extras: dict                     # {yoe_fit_weight, domain_ratio_weight}
    cross_encoder_query: str
    domain: DomainSpec
    relevant_skill_regex: str
    red_flags: dict                        # dict[str, RedFlag]
    relevant_skill_re: "re.Pattern" = None

    # -- convenience --------------------------------------------------------
    def signal_ids(self) -> list:
        return [s.id for s in self.signals]

    def evidence_signals(self) -> list:
        """Signals that carry an evidence_regex (contribute to evid_coverage)."""
        return [s for s in self.signals if s.evidence_regex is not None]

    def red_flag_enabled(self, name: str) -> bool:
        rf = self.red_flags.get(name)
        return bool(rf and rf.enabled)


@dataclass(frozen=True)
class Method:
    """Parsed method_config.yaml — the JD-stable mechanism (all numerics)."""
    schema_version: int
    ref_date: str
    recency: dict
    thresholds: dict
    # composite structure (matches root rank.py): additive channels + the
    # multiplicative evidence gate / claim discount / assessment bonus /
    # recency ladder / experience band.
    additive_weights: dict
    evidence_gate: dict
    claim_consistency: dict
    assessment_bonus: dict
    recency_ladder: list
    experience_band: dict
    damps: dict
    hopper_def: dict
    hop_rate_tenure_months: float
    integrity: dict
    world_facts: dict
    availability: dict
    notice_tiers: list
    assessment: dict
    alpha: float
    years_regex: str
    context_regex: dict
    evidence_context: dict
    lexicons: dict
    location_ladder: dict
    models: dict
    # compiled regexes
    years_re: "re.Pattern" = None
    context_re: dict = field(default_factory=dict)   # {internal,owner,scale} -> Pattern


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _err(field_path: str, msg: str) -> "ValueError":
    return ValueError(f"jd_profile.yaml: field '{field_path}' {msg}")


def _validate_with_jsonschema(data: dict) -> None:
    """Use jsonschema if available; raise ValueError naming the field on fail."""
    import json
    import jsonschema  # may raise ImportError -> caller falls back
    with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        loc = ".".join(str(p) for p in e.path) or "<root>"
        raise ValueError(f"jd_profile.yaml: field '{loc}' {e.message}")


def _require(data: dict, key: str, path: str) -> Any:
    if not isinstance(data, dict):
        raise _err(path, f"expected an object, got {type(data).__name__}")
    if key not in data:
        raise _err(f"{path}.{key}" if path else key, "is required but missing")
    return data[key]


def _check_type(val: Any, types, path: str, allow_none: bool = False) -> Any:
    if allow_none and val is None:
        return val
    if not isinstance(val, types):
        names = types.__name__ if isinstance(types, type) else \
            "/".join(t.__name__ for t in types)
        raise _err(path, f"must be {names}, got {type(val).__name__}")
    return val


def _validate_hand_rolled(d: dict) -> None:
    """Schema-equivalent required-field/type check used when jsonschema is absent.

    Mirrors jd/jd_profile.schema.json: same required fields, same coarse types.
    Raises ValueError naming the exact offending field.
    """
    if not isinstance(d, dict):
        raise ValueError("jd_profile.yaml: top level must be a mapping/object")

    sv = _require(d, "schema_version", "")
    if sv != 1:
        raise _err("schema_version", "must equal 1")

    # role -----------------------------------------------------------------
    role = _check_type(_require(d, "role", ""), dict, "role")
    _check_type(_require(role, "title", "role"), str, "role.title")
    if not role["title"]:
        raise _err("role.title", "must be a non-empty string")
    _check_type(_require(role, "domain", "role"), str, "role.domain")
    _check_type(_require(role, "notice_preference_days", "role"),
                int, "role.notice_preference_days")
    ie = _check_type(_require(role, "ideal_experience", "role"),
                     dict, "role.ideal_experience")
    _check_type(_require(ie, "peak_years", "role.ideal_experience"),
                (int, float), "role.ideal_experience.peak_years")
    sigma = _check_type(_require(ie, "sigma_years", "role.ideal_experience"),
                        (int, float), "role.ideal_experience.sigma_years")
    if sigma <= 0:
        raise _err("role.ideal_experience.sigma_years", "must be > 0")

    # locations ------------------------------------------------------------
    loc = _check_type(_require(d, "locations", ""), dict, "locations")
    _check_type(_require(loc, "preferred", "locations"), list, "locations.preferred")
    _check_type(_require(loc, "acceptable", "locations"), list, "locations.acceptable")
    _check_type(_require(loc, "relocation_acceptable", "locations"),
                bool, "locations.relocation_acceptable")
    _check_type(_require(loc, "remote_acceptable", "locations"),
                bool, "locations.remote_acceptable")

    # signals --------------------------------------------------------------
    signals = _check_type(_require(d, "signals", ""), list, "signals")
    if len(signals) < 1:
        raise _err("signals", "must contain at least one signal")
    seen_ids = set()
    for i, s in enumerate(signals):
        p = f"signals[{i}]"
        _check_type(s, dict, p)
        sid = _check_type(_require(s, "id", p), str, f"{p}.id")
        if not re.match(r"^[a-z][a-z0-9_]*$", sid):
            raise _err(f"{p}.id", "must match ^[a-z][a-z0-9_]*$")
        if sid in seen_ids:
            raise _err(f"{p}.id", f"duplicate signal id '{sid}'")
        seen_ids.add(sid)
        _check_type(_require(s, "label", p), str, f"{p}.label")
        _check_type(_require(s, "query", p), str, f"{p}.query")
        _check_type(_require(s, "dense_weight", p), (int, float), f"{p}.dense_weight")
        if s["dense_weight"] < 0:
            raise _err(f"{p}.dense_weight", "must be >= 0")
        # evidence_regex optional: string or null
        er = s.get("evidence_regex", None)
        _check_type(er, str, f"{p}.evidence_regex", allow_none=True)

    # cross_encoder_query --------------------------------------------------
    ceq = _check_type(_require(d, "cross_encoder_query", ""),
                      str, "cross_encoder_query")
    if not ceq:
        raise _err("cross_encoder_query", "must be a non-empty string")

    # domain ---------------------------------------------------------------
    dom = _check_type(_require(d, "domain", ""), dict, "domain")
    _check_type(_require(dom, "in_domain_terms", "domain"), list, "domain.in_domain_terms")
    _check_type(_require(dom, "out_of_domain_terms", "domain"), list, "domain.out_of_domain_terms")
    _check_type(_require(dom, "in_domain_regex", "domain"), str, "domain.in_domain_regex")
    _check_type(_require(dom, "out_of_domain_regex", "domain"), str, "domain.out_of_domain_regex")

    # relevant_skill_regex -------------------------------------------------
    rsr = _check_type(_require(d, "relevant_skill_regex", ""),
                      str, "relevant_skill_regex")
    if not rsr:
        raise _err("relevant_skill_regex", "must be a non-empty string")

    # red_flags ------------------------------------------------------------
    rf = _check_type(_require(d, "red_flags", ""), dict, "red_flags")
    for name, body in rf.items():
        p = f"red_flags.{name}"
        _check_type(body, dict, p)
        _check_type(_require(body, "enabled", p), bool, f"{p}.enabled")


# ---------------------------------------------------------------------------
# Regex compilation (compile once, surface a precise field on failure)
# ---------------------------------------------------------------------------
def _compile(pattern: str, field_path: str) -> "re.Pattern":
    try:
        return re.compile(pattern, re.I)
    except re.error as exc:
        raise ValueError(
            f"invalid regex in field '{field_path}': {exc}") from exc


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _build_profile(d: dict) -> Profile:
    role = d["role"]
    ie = role["ideal_experience"]
    role_spec = RoleSpec(
        title=role["title"],
        company=role.get("company", "") or "",
        domain=role["domain"],
        summary=role.get("summary", "") or "",
        min_years=ie.get("min_years"),
        max_years=ie.get("max_years"),
        peak_years=float(ie["peak_years"]),
        sigma_years=float(ie["sigma_years"]),
        notice_preference_days=int(role["notice_preference_days"]),
    )

    loc = d["locations"]
    loc_spec = LocSpec(
        preferred=[str(x).lower() for x in loc["preferred"]],
        acceptable=[str(x).lower() for x in loc["acceptable"]],
        relocation_acceptable=bool(loc["relocation_acceptable"]),
        remote_acceptable=bool(loc["remote_acceptable"]),
    )

    signals = []
    for i, s in enumerate(d["signals"]):
        er = s.get("evidence_regex", None)
        compiled = _compile(er, f"signals[{i}].evidence_regex") if er else None
        signals.append(Signal(
            id=s["id"], label=s["label"], query=s["query"],
            evidence_regex=er, dense_weight=float(s["dense_weight"]),
            evidence_re=compiled,
        ))

    dom = d["domain"]
    domain_spec = DomainSpec(
        in_domain_terms=[str(x).lower() for x in dom["in_domain_terms"]],
        out_of_domain_terms=[str(x).lower() for x in dom["out_of_domain_terms"]],
        in_domain_regex=dom["in_domain_regex"],
        out_of_domain_regex=dom["out_of_domain_regex"],
        in_domain_re=_compile(dom["in_domain_regex"], "domain.in_domain_regex"),
        out_of_domain_re=_compile(dom["out_of_domain_regex"], "domain.out_of_domain_regex"),
    )

    red_flags = {name: RedFlag(name=name, enabled=bool(body["enabled"]))
                 for name, body in d["red_flags"].items()}

    return Profile(
        schema_version=int(d["schema_version"]),
        role=role_spec,
        locations=loc_spec,
        signals=signals,
        dense_extras=dict(d.get("dense_extras", {}) or {}),
        cross_encoder_query=d["cross_encoder_query"],
        domain=domain_spec,
        relevant_skill_regex=d["relevant_skill_regex"],
        red_flags=red_flags,
        relevant_skill_re=_compile(d["relevant_skill_regex"], "relevant_skill_regex"),
    )


def _mreq(d: dict, key: str, path: str) -> Any:
    if not isinstance(d, dict) or key not in d:
        raise ValueError(f"method_config.yaml: field '{path}' is required but missing")
    return d[key]


def _build_method(m: dict) -> Method:
    if not isinstance(m, dict):
        raise ValueError("method_config.yaml: top level must be a mapping/object")
    runtime = _mreq(m, "runtime", "runtime")
    ref_date = _mreq(runtime, "ref_date", "runtime.ref_date")
    context_regex = _mreq(m, "context_regex", "context_regex")
    years_regex = _mreq(m, "years_regex", "years_regex")

    context_re = {}
    for k in ("internal", "owner", "scale"):
        if k not in context_regex:
            raise ValueError(f"method_config.yaml: field 'context_regex.{k}' missing")
        context_re[k] = _compile(context_regex[k], f"context_regex.{k}")

    return Method(
        schema_version=int(m.get("schema_version", 1)),
        ref_date=ref_date,
        recency=_mreq(m, "recency", "recency"),
        thresholds=_mreq(m, "thresholds", "thresholds"),
        additive_weights=_mreq(m, "additive_weights", "additive_weights"),
        evidence_gate=_mreq(m, "evidence_gate", "evidence_gate"),
        claim_consistency=_mreq(m, "claim_consistency", "claim_consistency"),
        assessment_bonus=_mreq(m, "assessment_bonus", "assessment_bonus"),
        recency_ladder=_mreq(m, "recency_ladder", "recency_ladder"),
        experience_band=_mreq(m, "experience_band", "experience_band"),
        damps=_mreq(m, "damps", "damps"),
        hopper_def=_mreq(m, "hopper_def", "hopper_def"),
        hop_rate_tenure_months=float(_mreq(m, "hop_rate_tenure_months",
                                           "hop_rate_tenure_months")),
        integrity=_mreq(m, "integrity", "integrity"),
        world_facts=m.get("world_facts", {}) or {},
        availability=_mreq(m, "availability", "availability"),
        notice_tiers=_mreq(m, "notice_tiers", "notice_tiers"),
        assessment=_mreq(m, "assessment", "assessment"),
        alpha=float(_mreq(m, "alpha", "alpha")),
        years_regex=years_regex,
        context_regex=context_regex,
        evidence_context=_mreq(m, "evidence_context", "evidence_context"),
        lexicons=_mreq(m, "lexicons", "lexicons"),
        location_ladder=_mreq(m, "location_ladder", "location_ladder"),
        models=m.get("models", {}) or {},
        years_re=_compile(years_regex, "years_regex"),
        context_re=context_re,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def load(jd_path: str = _DEFAULT_JD,
         method_path: str = _DEFAULT_METHOD) -> "tuple[Profile, Method]":
    """Load + validate both config seams; compile every regex once.

    Raises ValueError (naming the offending field) on any parse / schema /
    regex problem. Uses jsonschema for jd_profile validation if importable,
    else a hand-rolled equivalent so it works without the optional dep.
    """
    # -- parse yaml --------------------------------------------------------
    try:
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_raw = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise ValueError(f"jd_profile.yaml not found at {jd_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"jd_profile.yaml is not valid YAML: {exc}") from exc
    if jd_raw is None:
        raise ValueError("jd_profile.yaml is empty")

    try:
        with open(method_path, "r", encoding="utf-8") as f:
            method_raw = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise ValueError(f"method_config.yaml not found at {method_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"method_config.yaml is not valid YAML: {exc}") from exc
    if method_raw is None:
        raise ValueError("method_config.yaml is empty")

    # -- validate jd_profile (jsonschema if present, else hand-rolled) ------
    try:
        _validate_with_jsonschema(jd_raw)
    except ImportError:
        _validate_hand_rolled(jd_raw)

    # -- build typed objects (also compiles regexes) -----------------------
    profile = _build_profile(jd_raw)
    method = _build_method(method_raw)
    return profile, method


# ---------------------------------------------------------------------------
# CLI: python -m redrob_ranker.profile --check <jd_path>
# ---------------------------------------------------------------------------
def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Validate a jd_profile.yaml.")
    ap.add_argument("--check", default=_DEFAULT_JD,
                    help="path to jd_profile.yaml to validate")
    ap.add_argument("--method", default=_DEFAULT_METHOD,
                    help="path to method_config.yaml")
    args = ap.parse_args(argv)
    try:
        prof, _ = load(args.check, args.method)
    except ValueError as exc:
        print(f"INVALID: {exc}")
        return 1
    print(f"OK: {args.check} validates; "
          f"{len(prof.signals)} signals "
          f"({len(prof.evidence_signals())} with evidence_regex): "
          f"{prof.signal_ids()}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
