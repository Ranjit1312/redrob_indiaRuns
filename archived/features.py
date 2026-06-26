"""
features.py — Redrob ranker feature extraction (Steps 1-3 + gates)

Builds a FIXED-WIDTH feature matrix from candidate profiles, regardless of how
many jobs / skills each candidate has. Pipeline:

    Step 1  Dense bi-encoder facet similarities  -> recency-weighted pooling
    Step 2  BM25 lexical facet scores
    Step 3  Structured fit features (YoE, product-vs-services, IC-recency,
            domain, corroboration, location)
    +       Honeypot / consistency gate   (force-to-zero flag)
    +       Behavioral availability multiplier (from redrob_signals)

The embedding backend tries sentence-transformers (the production choice) and
transparently falls back to a TF-IDF backend so the notebook runs anywhere,
offline, with no model download. Swap EMBED_MODEL for the real run.

Usage (script or notebook):
    from features import build_feature_frame, load_candidates
    cands = load_candidates("sample_candidates.json")
    df, meta = build_feature_frame(cands)
    df.head()
"""
from __future__ import annotations
import json, math, re
from datetime import date
from collections import defaultdict
import numpy as np
import pandas as pd

# Reference "today" — keep deterministic for reproducibility (matches env date).
REF_DATE = date(2026, 6, 6)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # used if sentence-transformers is present

# --------------------------------------------------------------------------- #
# JD facets — each JD "must-have" is its own query (Step 1 design).            #
# --------------------------------------------------------------------------- #
FACETS = {
    "retrieval":  "production embeddings based retrieval semantic search sentence "
                  "transformers BGE E5 vector embeddings deployed to real users, "
                  "embedding drift index refresh retrieval quality",
    "vectordb":   "vector database hybrid search FAISS Pinecone Weaviate Qdrant "
                  "Milvus OpenSearch Elasticsearch approximate nearest neighbour index",
    "ranking":    "built and shipped end to end ranking search or recommendation "
                  "system to real users at scale, recommender learning to rank, production",
    "evaluation": "evaluation framework for ranking systems NDCG MRR MAP offline "
                  "online A/B testing relevance metrics",
    "applied_ml": "applied machine learning engineer at a product company, deployed "
                  "ML models to production, feature engineering, model serving",
    "llm_ft":     "LLM fine tuning LoRA QLoRA PEFT large language models prompt",
}
FACET_ORDER = list(FACETS.keys())

# Keyword lexicons (no-LLM signals) ----------------------------------------- #
CONSULTING = {"tcs", "tata consultancy", "infosys", "wipro", "accenture",
              "cognizant", "capgemini", "tech mahindra", "hcl", "mindtree",
              "ltimindtree", "deloitte"}
PRODUCT_INDUSTRIES = {"software", "fintech", "food delivery", "ai/ml", "e-commerce",
                      "ecommerce", "internet", "saas", "transportation",
                      "social media", "gaming", "edtech", "healthtech"}
IC_TOKENS   = {"engineer", "developer", "scientist", "programmer", "sde"}
MGMT_TOKENS = {"manager", "lead", "architect", "director", "vp", "head",
               "principal", "consultant"}
NLP_IR_TERMS = {"nlp", "natural language", "retrieval", "search", "ranking",
                "recommendation", "recommender", "embedding", "information retrieval",
                "semantic", "transformer", "llm", "text", "bm25"}
CV_SPEECH_TERMS = {"computer vision", "image", "yolo", "opencv", "detection",
                   "segmentation", "speech", "asr", "audio", "robotics", "slam"}
INDIA_TIER1 = {"pune", "noida", "hyderabad", "bangalore", "bengaluru", "mumbai",
               "delhi", "gurgaon", "gurugram", "chennai", "ncr", "kolkata"}
PREFERRED_CITIES = {"pune", "noida"}

# --------------------------------------------------------------------------- #
# I/O                                                                         #
# --------------------------------------------------------------------------- #
def load_candidates(path: str):
    """Load either a JSON array (sample_candidates.json) or JSONL."""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]

# --------------------------------------------------------------------------- #
# Date helpers (Step 3 / honeypot) — no LLM, pure arithmetic                  #
# --------------------------------------------------------------------------- #
def _parse(d):
    if not d:
        return None
    try:
        y, m, day = (int(x) for x in d[:10].split("-"))
        return date(y, m, day)
    except Exception:
        return None

def _months_between(a: date, b: date) -> float:
    return (b.year - a.year) * 12 + (b.month - a.month) + (b.day - a.day) / 30.0

def months_since_end(job) -> float:
    """How long ago this role ended (0 if current)."""
    if job.get("is_current") or not job.get("end_date"):
        return 0.0
    e = _parse(job["end_date"]) or REF_DATE
    return max(0.0, _months_between(e, REF_DATE))

def recency_weight(msince: float, halflife: float = 18.0) -> float:
    """JD 18-month preference -> a role 18mo old counts half."""
    return 0.5 ** (msince / halflife)

# --------------------------------------------------------------------------- #
# Evidence text — deliberately built from descriptions, NOT the skills array  #
# (the skills array is where keyword-stuffers hide).                          #
# --------------------------------------------------------------------------- #
def career_text(job) -> str:
    return f"{job.get('title','')}. {job.get('description','')}"

def summary_text(c) -> str:
    p = c["profile"]
    return f"{p.get('headline','')}. {p.get('summary','')}"

# --------------------------------------------------------------------------- #
# Embedding backend: sentence-transformers, with TF-IDF fallback              #
# --------------------------------------------------------------------------- #
class EmbeddingBackend:
    def __init__(self, corpus):
        self.kind = None
        try:
            from sentence_transformers import SentenceTransformer  # noqa
            self.model = SentenceTransformer(EMBED_MODEL)
            self.kind = "st"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                       min_df=1, max_features=8000)
            self.vec.fit(corpus if corpus else ["placeholder"])
            self.kind = "tfidf"

    def encode(self, texts):
        texts = [t if t else " " for t in texts]
        if self.kind == "st":
            v = self.model.encode(texts, normalize_embeddings=True,
                                  show_progress_bar=False)
            return np.asarray(v, dtype=np.float32)
        m = self.vec.transform(texts).toarray().astype(np.float32)
        n = np.linalg.norm(m, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return m / n

def _cos(a, b):  # both L2-normalised already
    return float(np.dot(a, b))

# --------------------------------------------------------------------------- #
# Step 1 — per-facet dense similarity with recency pooling                    #
# --------------------------------------------------------------------------- #
def dense_facet_features(c, job_vecs, summ_vec, facet_vecs):
    """job_vecs: (k,d) embeddings for this candidate's k jobs (recency aligned)."""
    jobs = c["career_history"]
    durs = np.array([max(1, j.get("duration_months", 1)) for j in jobs], dtype=float)
    msince = np.array([months_since_end(j) for j in jobs], dtype=float)
    rweight = np.array([recency_weight(m) for m in msince])
    most_recent = int(np.argmin(msince)) if len(jobs) else None

    feats = {}
    for fname in FACET_ORDER:
        fv = facet_vecs[fname]
        if len(jobs) == 0:
            sims = np.array([0.0])
            w = np.array([1.0])
        else:
            sims = np.array([_cos(jv, fv) for jv in job_vecs])
            w = rweight * np.sqrt(durs)
        wsum = w.sum() or 1.0
        feats[f"{fname}__recencywt"] = float(np.dot(sims, w) / wsum)   # PRIMARY
        feats[f"{fname}__peak"]      = float(sims.max())               # best-ever
        feats[f"{fname}__nhits"]     = int((sims > 0.45).sum())
        feats[f"{fname}__recent"]    = float(sims[most_recent]) if most_recent is not None else 0.0
        feats[f"{fname}__summary"]   = _cos(summ_vec, fv)              # self-described
    return feats

# --------------------------------------------------------------------------- #
# Step 2 — BM25 lexical facet scores                                          #
# --------------------------------------------------------------------------- #
_tok = re.compile(r"[a-z0-9\+\#\.]+")
def _tokens(s): return _tok.findall(s.lower())

def bm25_facet_scores(docs):
    """docs: list[str] one evidence doc per candidate. Returns dict facet->scores."""
    try:
        from rank_bm25 import BM25Okapi
    except Exception:
        return None
    tokenized = [_tokens(d) for d in docs]
    bm25 = BM25Okapi(tokenized)
    out = {}
    for fname, q in FACETS.items():
        out[fname] = np.asarray(bm25.get_scores(_tokens(q)), dtype=float)
    # min-max normalise each facet to [0,1] for comparability
    for fname in out:
        s = out[fname]
        rng = s.max() - s.min()
        out[fname] = (s - s.min()) / rng if rng > 0 else np.zeros_like(s)
    return out

# --------------------------------------------------------------------------- #
# Step 3 — structured fit features                                            #
# --------------------------------------------------------------------------- #
def _is_ic(title):
    t = title.lower()
    return any(k in t for k in IC_TOKENS) and not any(k in t for k in MGMT_TOKENS)

def _is_mgmt(title):
    t = title.lower()
    return any(k in t for k in MGMT_TOKENS)

def structured_features(c, dense):
    p = c["profile"]; jobs = c["career_history"]
    yoe = float(p.get("years_of_experience", 0))

    # YoE band fit — gaussian peak at 7, the JD "ideal" centre of 6-8.
    yoe_fit = math.exp(-((yoe - 7.0) ** 2) / (2 * 2.5 ** 2))

    # tenure / job-hop
    durs = [j.get("duration_months", 0) for j in jobs]
    avg_tenure = float(np.mean(durs)) if durs else 0.0
    hop_rate = float(np.mean([d < 18 for d in durs])) if durs else 0.0
    n_jobs = len(jobs)

    # product vs services / consulting
    comps = [(j.get("company", "") or "").lower() for j in jobs]
    inds  = [(j.get("industry", "") or "").lower() for j in jobs]
    consult_hits = sum(any(k in comp for k in CONSULTING) for comp in comps)
    only_consulting = (consult_hits == n_jobs and n_jobs > 0)
    product_frac = float(np.mean([any(k in ind for k in PRODUCT_INDUSTRIES)
                                  for ind in inds])) if inds else 0.0

    # IC-recency (the 18-month "still writes code" preference)
    ic_gaps = [months_since_end(j) for j in jobs if _is_ic(j.get("title", ""))]
    months_since_ic = float(min(ic_gaps)) if ic_gaps else 999.0
    recent_is_mgmt = int(_is_mgmt(jobs[0].get("title", "")) if jobs else 0)

    # domain: NLP/IR vs CV/speech, from evidence text + skills names
    blob = (summary_text(c) + " " +
            " ".join(career_text(j) for j in jobs) + " " +
            " ".join(s.get("name", "") for s in c.get("skills", []))).lower()
    nlp_ir = sum(blob.count(t) for t in NLP_IR_TERMS)
    cv_sp  = sum(blob.count(t) for t in CV_SPEECH_TERMS)
    domain_ratio = (nlp_ir + 1) / (nlp_ir + cv_sp + 2)   # ->1 NLP/IR, ->0 CV/speech

    # corroboration: AI skills CLAIMED in the (easily-stuffed) skills array vs
    # SUPPORTED by narrative evidence (summary + job descriptions). Prefix match
    # so "Embeddings"->"embed", "Information Retrieval"->"infor/retri" still fire.
    ai_skill_terms = {"nlp", "llm", "fine-tun", "embed", "machine learning",
                      "deep learning", "recommend", "retriev", "pytorch", "mlops",
                      "tensorflow", "transformer", "ranking", "search", "faiss",
                      "vector", "feature engineering", "ml"}
    evidence = (summary_text(c) + " " +
                " ".join(career_text(j) for j in jobs)).lower()
    claimed = [s.get("name", "").lower() for s in c.get("skills", [])]
    ai_claimed = [s for s in claimed if any(t in s for t in ai_skill_terms)]
    def _supported(skill):
        toks = [w for w in re.split(r"[^a-z]+", skill) if len(w) >= 4]
        return any(w[:5] in evidence for w in toks)
    ai_supported = [s for s in ai_claimed if _supported(s)]
    corroboration = (len(ai_supported) + 0.0) / (len(ai_claimed) + 1.0)  # low => stuffer

    # location fit
    loc = (p.get("location", "") + " " + p.get("country", "")).lower()
    sig = c["redrob_signals"]
    in_india_t1 = any(ci in loc for ci in INDIA_TIER1)
    preferred   = any(ci in loc for ci in PREFERRED_CITIES)
    loc_fit = 1.0 if preferred else (0.8 if in_india_t1 else
              (0.6 if sig.get("willing_to_relocate") else 0.2))

    # soft data-consistency signals (noisy -> features, NOT honeypot gates)
    sal = sig.get("expected_salary_range_inr_lpa", {})
    salary_inconsistent = int(sal.get("min", 0) > sal.get("max", 1e9))
    end_years = [e.get("end_year") for e in c.get("education", []) if e.get("end_year")]
    yoe_vs_grad_gap = (yoe - (REF_DATE.year - min(end_years))) if end_years else 0.0

    return {
        "yoe": yoe, "yoe_fit": yoe_fit, "n_jobs": n_jobs,
        "avg_tenure_months": avg_tenure, "job_hop_rate": hop_rate,
        "consulting_frac": consult_hits / n_jobs if n_jobs else 0.0,
        "only_consulting": int(only_consulting),
        "product_frac": product_frac,
        "months_since_ic_role": months_since_ic,
        "recent_role_is_mgmt": recent_is_mgmt,
        "domain_nlp_ratio": domain_ratio,
        "ai_skill_corroboration": corroboration,
        "ai_skills_claimed": len(ai_claimed),
        "location_fit": loc_fit,
        "github_activity": max(0.0, sig.get("github_activity_score", -1)),
        "salary_inconsistent": salary_inconsistent,
        "yoe_vs_grad_gap": yoe_vs_grad_gap,
    }

# --------------------------------------------------------------------------- #
# Honeypot / consistency gate — impossible profiles -> force to zero          #
# --------------------------------------------------------------------------- #
def honeypot_flags(c):
    """Crisp CAREER/SKILL-logic impossibilities only — the spec's honeypot type
    ('8 yrs at a 3-yr-old company', 'expert in 10 skills with 0 years used').

    NOTE: salary min>max and last_active<signup are pervasive synthetic-data
    noise (~26% of the pool), NOT honeypots — they are exposed as SOFT features
    in structured_features(), never as a disqualifying gate.
    """
    p = c["profile"]; jobs = c["career_history"]
    yoe = float(p.get("years_of_experience", 0))
    flags = []

    # sum of (sequential) tenures far exceeding the stated YoE
    career_years = sum(j.get("duration_months", 0) for j in jobs) / 12.0
    if career_years > yoe + 3.5:
        flags.append("career_sum_exceeds_yoe")

    # any single role longer than the whole career claim
    if any(j.get("duration_months", 0) / 12.0 > yoe + 1.5 for j in jobs):
        flags.append("single_role_exceeds_yoe")

    # tenure at a company exceeding its plausible age, if start predates company...
    # (no founding date in schema -> approximated by the two checks above)

    # "expert" proficiency with zero usage, or expert in implausibly many skills
    skills = c.get("skills", [])
    if any(s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
           for s in skills):
        flags.append("expert_zero_duration")
    if sum(s.get("proficiency") == "expert" for s in skills) >= 8:
        flags.append("too_many_expert")

    return flags

# --------------------------------------------------------------------------- #
# Behavioral availability multiplier (from redrob_signals) ~ [0.5, 1.1]       #
# --------------------------------------------------------------------------- #
def availability_multiplier(c):
    s = c["redrob_signals"]
    la = _parse(s.get("last_active_date"))
    months_inactive = _months_between(la, REF_DATE) if la else 12.0
    recency   = math.exp(-max(0, months_inactive) / 6.0)          # 6-mo half-life-ish
    response  = s.get("recruiter_response_rate", 0.0)
    otw       = 1.0 if s.get("open_to_work_flag") else 0.0
    interview = s.get("interview_completion_rate", 0.5)
    complete  = s.get("profile_completeness_score", 50) / 100.0
    raw = (0.35 * recency + 0.25 * response + 0.15 * otw +
           0.15 * interview + 0.10 * complete)
    return 0.5 + 0.6 * raw            # in ~[0.5, 1.1]

# --------------------------------------------------------------------------- #
# Orchestration — build the FIXED-WIDTH feature frame                         #
# --------------------------------------------------------------------------- #
def build_feature_frame(cands):
    # ---- assemble embedding corpus (all jobs + summaries + facet queries) ----
    job_offsets = []      # (start, end) row range into job_matrix per candidate
    all_job_texts, all_summaries = [], []
    for c in cands:
        start = len(all_job_texts)
        for j in c["career_history"]:
            all_job_texts.append(career_text(j))
        job_offsets.append((start, len(all_job_texts)))
        all_summaries.append(summary_text(c))

    corpus = all_job_texts + all_summaries + list(FACETS.values())
    backend = EmbeddingBackend(corpus)
    job_matrix  = backend.encode(all_job_texts) if all_job_texts else np.zeros((0, 1), np.float32)
    summ_matrix = backend.encode(all_summaries)
    facet_vecs  = {f: v for f, v in zip(FACET_ORDER, backend.encode(list(FACETS.values())))}

    # ---- BM25 over per-candidate evidence docs ----
    evidence_docs = [summary_text(c) + " " + " ".join(career_text(j)
                     for j in c["career_history"]) for c in cands]
    bm25 = bm25_facet_scores(evidence_docs)

    rows = []
    meta = []
    for i, c in enumerate(cands):
        s, e = job_offsets[i]
        jvecs = job_matrix[s:e]
        feats = {"candidate_id": c["candidate_id"]}
        feats.update(dense_facet_features(c, jvecs, summ_matrix[i], facet_vecs))
        if bm25 is not None:
            for fname in FACET_ORDER:
                feats[f"{fname}__bm25"] = float(bm25[fname][i])
        feats.update(structured_features(c, feats))

        flags = honeypot_flags(c)
        feats["honeypot_flag"] = int(len(flags) > 0)
        feats["availability_mult"] = availability_multiplier(c)

        rows.append(feats)
        meta.append({"candidate_id": c["candidate_id"],
                     "title": c["profile"]["current_title"],
                     "yoe": c["profile"]["years_of_experience"],
                     "honeypot_reasons": ";".join(flags)})

    df = pd.DataFrame(rows).set_index("candidate_id")
    df = df.fillna(0.0)                      # (real run: leave NaN for LightGBM)
    return df, pd.DataFrame(meta).set_index("candidate_id")
