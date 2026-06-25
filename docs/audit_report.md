# Audit Report — five audit-and-refine rounds

Structured record of the audit loop that produced versions v1 -> v4
(sourced from the project report, sections 3 and 6). The cycle for every round:
**rank -> audit (LLM agent reading the JD + profiles) -> verify every finding
against pool base rates -> encode only what survives -> re-rank with probe
tracking.**

---

## Round 1 — top-30 audit of v1 (field subset)

**Scope:** top-30 of the v1 baseline ranking, audited on a subset of profile fields.

**Verdict: 13 of 30 WEAK or RED-FLAG.**

Findings:

- **Cosine similarity is blurry** — churn-model and dashboard profiles scored
  0.58-0.68 vs genuine retrieval engineers at 0.6-0.8; the composite could not
  separate them.
- **CAND_0000031, a known plain-language Tier-5 (strong fit, zero buzzwords),
  sat at rank 91** — punished by v1's own anti-stuffer penalty, since it had no
  skill tags to corroborate.
- **CAND_0039983, a CV-primary disqualifier, sat at rank 10** (computer-vision
  career without NLP/IR — an explicit JD disqualifier).
- **YoE contradictions at ranks 14 and 25** — stated 16.6 / 16.9 years vs ~7.1
  years of actual career history; the 1.9x integrity bound was too loose.

**Outcome -> v2 rules:** career-text evidence layer (ownership x context x
recency, max-pooled per job), CV-primary damp x0.60, tightened yoe-vs-career
integrity threshold, evidence coverage added to the corroboration max (rescuing
plain-language profiles).

---

## Round 2 — top-50 audit of v1/v2 transition

**Scope:** top-50, all text fields.

Findings:

- **"Internal knowledge base" template trap** — internal-tool demo projects
  scored like production search systems -> context multiplier x0.4.
- **BLEU/ROUGE counted as ranking evaluation** — chatbot eval language
  qualified for the rankeval facet -> rankeval regex now requires ranking
  metrics (NDCG/MRR/MAP/recall@k) or offline-online / A-B language.
- **Summary-only keyword claims** — the rank-10 case claimed FAISS only in its
  boilerplate summary, never in any job description -> evidence scored on
  career descriptions only; summaries excluded.
- **Ownership insensitivity** — "led the migration... 30M corpus" scored the
  same as "deployment was handled by the platform team" -> ownership verbs x1.0
  vs participation x0.7.
- **Honeypot at rank 9 escaping a 1.9x threshold** — 15.2 stated years vs 86
  months of history -> threshold tightened (`yoe*12 > career_m*1.6 + 18`) plus
  a summary-stated-years cross-check.

**Outcome -> v2/v3 tightening.** Post-v2 result: 1 WEAK left in top-30; the
rank-9 honeypot fell to ~99,952.

---

## Round 3 — assessment diff-audit (v2 -> v3)

**Scope:** every candidate entering or exiting the top-50 when the
skill-assessment signal was added.

Findings:

- **Single-test, coverage-0.15 jumpers** — ungated assessment strength lifted
  profiles with ~0.15 evidence coverage roughly 200 ranks on the strength of
  one proctored test score.

**Outcome -> evidence gating:** assessment corroboration scaled by
`min(1, coverage/0.25)` (full credit only at coverage >= 0.25) and
confidence-discounted by test count (`n/(n+0.5)`). Top-50 churn dropped from 8
to 3 entrants, all legitimate tie-breaks.

---

## Round 4 — full-profile audit of the blend (complete JD + every field)

**Scope:** top-30 with the auditor reading entire candidate records and the
complete JD for the first time.

**Verdict: MAJOR.**

Findings:

- **The JD's literal "dormant + unresponsive" exclusion example at rank 17**
  (a month-arithmetic bug had let the 191-day case escape; fixed with
  day-precision).
- **Certification anachronisms** — e.g. a LangChain certification dated 2018
  (the library didn't exist).
- **Education impossibilities** — concurrent same-window degrees.
- **Signup-vs-activity contradictions** — last_active earlier than signup_date.

**Base-rate verification of the auditor's 5 proposed removals** (the pool is
the arbiter — an LLM auditor's claim is a hypothesis, not a fact):

| Pattern | Pool count / rate | Ruling |
|---|---|---|
| last_active < signup | 7,496 (7.5%) | pervasive generator noise -> soft x0.97 |
| career-before-education | 19,499 (19.5%) | pervasive noise -> no gate |
| concurrent degrees | 728 (0.73%) | soft x0.93 |
| cert anachronisms | 45 (0.045%) | rare fingerprint -> strong rule x0.30 |
| dormant + unresponsive | 3,710 (3.7%) | the JD's own example -> strong rule x0.5 |

Three of five demanded removals were demoted to soft penalties; the two rare
patterns became strong rules. **Rare -> strong rules; pervasive -> soft
penalties.**

**Outcome -> v4 rules** (dormant x0.5, low-response x0.8, anachronism x0.30,
activity-before-signup x0.97, concurrent degrees x0.93, remote-pref location
caps, notice tiers 1.0/0.93/0.90/0.85).

---

## Round 5 — final top-50 verification (post-tuning, post-v4)

**Scope:** top-50 of the final blended ranking.

**Verdict: MINOR.**

Findings and fixes:

- The reasoning label **"ranking-evaluation rigor (NDCG/MRR/A-B)" overstated
  the evidence for 2 candidates** whose profiles don't name those metrics ->
  softened to "ranking-evaluation work" (no hallucinated specifics).
- **Evidence-coverage tie-break** added before candidate_id in the top-100 sort.

**Final state: zero disqualifiers, zero hallucinations, zero
impossible/dormant profiles in the top-100.** Reproduction path = development
path, 100/100 identical rows.

---

## Methodology note

- **Every audit claim was verified against pool base rates before being
  encoded** — this caught both failure directions: over-gating (a
  skill-duration gate that would have zeroed 4.7% of the pool including good
  candidates) and under-gating (the 45-profile anachronism fingerprint).
- **14 tracked probe candidates** are printed on every re-rank — the elite #1,
  the Noida/Pune logistics-clean pair, the plain-language Tier-5, the honeypot,
  the hopper, the dormant case, the London no-relocate, etc. Regressions
  surface immediately (e.g. raising the evidence weight in v2 accidentally
  promoted the London candidate 33 -> 12; caught the same run, fixed with a
  targeted damp). Final probe state: honeypot at 99,9xx, dormant at 359,
  elite #1 at 1.
