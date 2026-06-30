"""Single source of truth for the GATE / output columns.

These columns are applied as MULTIPLIERS *outside* the learned blend (integrity,
availability, notice, location, ...) or are rule-engine outputs; they must NEVER
be fed to the LightGBM student as features, or the student could learn to
reproduce a gate and the "gates apply outside the blend" guarantee would break.

`train.py` builds its `feat_cols` by EXCLUDING this set; `rank.py` loads the
persisted `feature_cols.json` and asserts it stays disjoint from this set. Both
import GATES from here so the two sides can never drift and silently leak a gate
into the student matrix.
"""

GATES = {"availability_mult", "integrity", "notice_pen", "loc2_v4",
         "fit_rules", "final_rules", "dormant", "low_rr", "anach",
         "la_lt_signup", "concurrent_deg", "remote_pref", "no_reloc",
         "city_ok", "notice_days"}
