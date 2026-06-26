"""redrob_ranker — v7 JD-seam pure core.

Public surface:
    load(jd_path, method_path) -> (Profile, Method)   [profile.py]
    Profile, Method, Signal, RoleSpec, LocSpec, DomainSpec
    compute_rules(features, profile, method) -> RuleResult   [rules.py]
    RuleResult, mm
    extract_intrinsic(records) -> DataFrame, INTRINSIC_COLUMNS  [intrinsic.py]
"""
from .profile import (load, Profile, Method, Signal, RoleSpec, LocSpec,
                      DomainSpec, RedFlag)
from .rules import compute_rules, RuleResult, mm, required_columns
from .intrinsic import extract_intrinsic, INTRINSIC_COLUMNS

__all__ = [
    "load", "Profile", "Method", "Signal", "RoleSpec", "LocSpec",
    "DomainSpec", "RedFlag",
    "compute_rules", "RuleResult", "mm", "required_columns",
    "extract_intrinsic", "INTRINSIC_COLUMNS",
]
