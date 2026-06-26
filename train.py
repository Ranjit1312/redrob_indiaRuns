"""
train.py — v7 component 3: teacher -> tuned student, in one script (GPU).

Chains the two training stages, reading its features from the live v7 feature
pass (`python rank.py --candidates <jsonl> --features-only` ->
artifacts_v7/features_v7.parquet):

  PHASE A  TEACHER
    - shortlist = top --shortlist by the rules-only score + --negatives random
      others (negatives teach what irrelevance looks like)
    - the cross-encoder teacher (method.models.teacher) scores
      (cross_encoder_query, evidence) pairs on CUDA; the query comes from
      profile.cross_encoder_query so the teacher re-targets automatically when
      the JD changes
    - bias check: CE scores for the corpus's most common job chunks printed
      BEFORE the labels are trusted
    - pseudo-labels gated by anti-stuffer corroboration x integrity. The
      integrity gate is taken from redrob_ranker.rules.compute_rules (the SINGLE
      rules engine) rather than re-deriving the integrity ladder here — this is
      the v7 dedup payoff.

  PHASE B  STUDENT
    - stratified 20% holdout + 4 stratified CV folds over 16 label bins
    - coordinate descent around the proven incumbent config, early stopping
    - final refit -> artifacts_v7/model.txt + feature_cols.json

    python rank.py --candidates ./candidates.jsonl --features-only
    python train.py
"""
import argparse, collections, json, os, time

import numpy as np
import pandas as pd

from redrob_ranker import profile as rprofile
from redrob_ranker import rules as rrules

BASE = os.environ.get("RANKER_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(BASE, "artifacts_v7")
HERE = os.path.dirname(os.path.abspath(__file__))
JD_DEFAULT = os.path.join(HERE, "jd", "jd_profile.yaml")
METHOD_DEFAULT = os.path.join(HERE, "jd", "method_config.yaml")
SEP = "\x1f"

# gate / output columns never fed to the student (mirror of rank.py GATES)
GATES = {"availability_mult", "integrity", "notice_pen", "loc2_v4",
         "fit_rules", "final_rules", "dormant", "low_rr", "anach",
         "la_lt_signup", "concurrent_deg", "remote_pref", "no_reloc",
         "city_ok", "notice_days"}

INCUMBENT = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=40,
                 feature_fraction=0.8, bagging_fraction=0.8,
                 lambda_l2=0.0, lambdarank_truncation_level=100)
NEIGHBORHOOD = dict(learning_rate=[0.03, 0.05, 0.08],
                    num_leaves=[15, 31, 63],
                    min_data_in_leaf=[20, 40, 80],
                    feature_fraction=[0.6, 0.8, 1.0],
                    bagging_fraction=[0.7, 0.8, 0.9],
                    lambda_l2=[0.0, 1.0, 5.0],
                    lambdarank_truncation_level=[50, 100, 200])


def groups_of(total, size=500):
    g = [size] * (total // size)
    if total % size:
        g.append(total % size)
    return g


def ndcg_at(y_true, y_score, k):
    o = np.argsort(-y_score)[:k]
    g = (2.0 ** (4 * y_true[o]) - 1) / np.log2(np.arange(2, len(o) + 2))
    i = np.sort(y_true)[::-1][:k]
    nm = ((2.0 ** (4 * i) - 1) / np.log2(np.arange(2, len(i) + 2))).sum()
    return g.sum() / nm if nm > 0 else 0.0


def stratified_indices(y_bins, frac, seed):
    rng = np.random.default_rng(seed)
    take = []
    for b in np.unique(y_bins):
        rows = np.where(y_bins == b)[0]
        take.append(rng.choice(rows, size=max(1, int(round(frac * len(rows)))),
                               replace=False))
    return np.concatenate(take)


def stratified_folds(y_bins, k, seed):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y_bins), dtype=int)
    for b in np.unique(y_bins):
        rows = rng.permutation(np.where(y_bins == b)[0])
        for i, r in enumerate(np.array_split(rows, k)):
            fold[r] = i
    return fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortlist", type=int, default=None)
    ap.add_argument("--negatives", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--jd", default=JD_DEFAULT)
    ap.add_argument("--method", default=METHOD_DEFAULT)
    args = ap.parse_args()
    t0 = time.time()
    import torch

    # the JD seam (cross_encoder_query, teacher model, stuffer gate, ...)
    art_jd = os.path.join(ART, "jd_profile.yaml")
    jd_path = art_jd if os.path.exists(art_jd) else args.jd
    profile, method = rprofile.load(jd_path, args.method)
    TC = method.models["teacher"]
    shortlist_n = args.shortlist or TC["shortlist"]
    negatives_n = args.negatives or TC["negatives"]
    batch = args.batch or TC["batch"]

    feats_path = os.path.join(ART, "features_v7.parquet")
    if not os.path.exists(feats_path):
        raise SystemExit("[train] artifacts_v7/features_v7.parquet missing — run "
                         "`python rank.py --candidates <jsonl> --features-only` first")
    F = pd.read_parquet(feats_path)
    evid_df = pd.read_parquet(os.path.join(ART, "evidence_texts.parquet"))
    n = len(F)
    print(f"[train] features {F.shape}; evidence texts {len(evid_df)}")

    # the single rules engine over the full frame: gives the rules-only ordering
    # (final_rules) AND the integrity gate the pseudo-labels reuse.
    rr = rrules.compute_rules(F, profile, method)
    final_rules = pd.Series(rr.final_rules, index=F.index)
    integrity = pd.Series(rr.integrity, index=F.index)

    # =================== PHASE A — cross-encoder teacher ===================
    tA = time.time()
    order = final_rules.sort_values(ascending=False)
    short_n = min(shortlist_n, n)
    short_ids = list(order.index[:short_n])
    rest = order.index[short_n:]
    rng = np.random.default_rng(TC["seed"])
    neg_n = min(negatives_n, len(rest))
    neg_ids = (list(rest[rng.choice(len(rest), size=neg_n, replace=False)])
               if neg_n else [])
    todo = short_ids + neg_ids
    print(f"[train] CE will score {len(todo)} candidates "
          f"({short_n} shortlist + {neg_n} random negatives)")

    def ce_text(cid):
        r = evid_df.loc[cid]
        jobs = r["jobs_text"].split(SEP) if r["jobs_text"] else []
        return "\n".join([r["headline_summary"]] + jobs)

    from sentence_transformers import CrossEncoder
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ce = CrossEncoder(TC["name"], device=dev, max_length=TC["max_length"],
                      model_kwargs={"torch_dtype": torch.float16}
                      if dev == "cuda" else {})
    print(f"[train] {TC['name']} loaded on device={dev} "
          f"(cuda_available={torch.cuda.is_available()})")
    JD_QUERY = profile.cross_encoder_query

    # bias validation BEFORE trusting labels: most common job-description chunks
    chunk_counter = collections.Counter()
    for jt in evid_df["jobs_text"].values:
        for ch in jt.split(SEP):
            if ch.strip():
                chunk_counter[ch] += 1
    templates = chunk_counter.most_common(14)
    if templates:
        tpl = ce.predict([(JD_QUERY, c) for c, _ in templates], batch_size=batch)
        tpl = 1 / (1 + np.exp(-np.asarray(tpl, dtype=np.float64)))
        print("\n[bias-check] CE scores for the corpus's most common chunks:")
        for (c, cnt), sc in sorted(zip(templates, tpl), key=lambda x: -x[1]):
            print(f"  {sc:.3f} (x{cnt:>6}) {c[:110]}")

    t1 = time.time()
    raw = ce.predict([(JD_QUERY, ce_text(cid)) for cid in todo],
                     batch_size=batch, show_progress_bar=True)
    raw = np.asarray(raw, dtype=np.float64)
    ce_sig = 1 / (1 + np.exp(-raw))
    print(f"[train] scored {len(todo)} pairs in {time.time()-t1:.0f}s "
          f"({len(todo)/max(1e-9, time.time()-t1):.1f}/s)")

    # gate the labels: anti-stuffer corroboration x integrity.
    # Mirrors the rank composite's evidence/claim/assessment gating, collapsed
    # into one "is this actually corroborated?" multiplier: floor + span * max(
    # ai_skill_corroboration, evidence_coverage, coverage-gated assessment). The
    # constants are the SAME reworked method keys rank.py uses — evidence_gate
    # (floor/span) and assessment_bonus.full_credit_cov — so there is no separate
    # stuffer_gate config to drift. Integrity is taken straight from the rules
    # engine (rr.integrity); no integrity ladder is re-derived here.
    sub = F.loc[todo]
    EG = method.evidence_gate
    cov0 = method.assessment_bonus["full_credit_cov"]
    assess_corr = (sub["assess_strength"].values
                   * np.minimum(1.0, sub["evid_coverage"].values / cov0))
    stuffer_gate = EG["floor"] + EG["span"] * np.maximum.reduce(
        [sub["ai_skill_corroboration"].values,
         sub["evid_coverage"].values, assess_corr])
    pseudo = ce_sig * stuffer_gate * integrity.loc[todo].values
    lab = pd.DataFrame({"candidate_id": todo, "ce_raw": raw,
                        "ce_sigmoid": ce_sig, "pseudo_label": pseudo,
                        "is_negative_sample": [0] * len(short_ids) + [1] * len(neg_ids)}
                       ).set_index("candidate_id")
    lab.to_parquet(os.path.join(ART, "pseudo_labels.parquet"))
    print(f"[train] PHASE A (teacher) wall={time.time()-tA:.0f}s; label dist: "
          f"p10={np.percentile(pseudo, 10):.3f} p50={np.percentile(pseudo, 50):.3f} "
          f"p90={np.percentile(pseudo, 90):.3f} max={pseudo.max():.3f}")
    if dev == "cuda":
        print(f"[train] gpu_peak={torch.cuda.max_memory_allocated()/2**30:.2f}GB")

    # =================== PHASE B — tuned LambdaMART student ================
    tB = time.time()
    import lightgbm as lgb
    feat_cols = [c for c in F.columns if c not in GATES and F[c].dtype != object]
    X = F.loc[lab.index, feat_cols].astype("float32").values
    y_cont = lab["pseudo_label"].values
    y = pd.qcut(pd.Series(y_cont).rank(method="first"), q=16,
                labels=False, duplicates="drop").values.astype(int)

    ho_idx = stratified_indices(y, 0.2, seed=7)
    hold = np.zeros(len(X), dtype=bool); hold[ho_idx] = True
    Xtr, ytr, yc_tr = X[~hold], y[~hold], y_cont[~hold]
    Xho, yho_bins, yho_cont = X[hold], y[hold], y_cont[hold]
    print(f"[tune] split: train={len(Xtr)} holdout={len(Xho)} "
          f"neg-frac train={lab['is_negative_sample'].values[~hold].mean():.3f} "
          f"holdout={lab['is_negative_sample'].values[hold].mean():.3f} "
          f"features={len(feat_cols)}")

    fold = stratified_folds(ytr, 4, seed=5)

    def cv_score(cfg):
        scores, iters = [], []
        for vi in range(4):
            tr, va = fold != vi, fold == vi
            dtr = lgb.Dataset(Xtr[tr], label=ytr[tr], group=groups_of(tr.sum()))
            dva = lgb.Dataset(Xtr[va], label=ytr[va], group=groups_of(va.sum()),
                              reference=dtr)
            params = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[50],
                          label_gain=[2**i - 1 for i in range(16)], verbosity=-1,
                          seed=7, bagging_freq=1, **cfg)
            bst = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
            p = bst.predict(Xtr[va], num_iteration=bst.best_iteration)
            scores.append(0.625 * ndcg_at(yc_tr[va], p, 10) +
                          0.375 * ndcg_at(yc_tr[va], p, 50))
            iters.append(bst.best_iteration)
        return float(np.mean(scores)), int(np.mean(iters))

    best, cache = dict(INCUMBENT), {}
    best_score, _ = cv_score(best); cache[tuple(sorted(best.items()))] = best_score
    print(f"[tune] incumbent cv={best_score:.4f} {best}")
    for p_ in range(2):                                  # 2 coordinate passes
        for knob, values in NEIGHBORHOOD.items():
            for v in values:
                if v == best[knob]:
                    continue
                cand = dict(best); cand[knob] = v
                key = tuple(sorted(cand.items()))
                if key in cache:
                    continue
                s, _ = cv_score(cand); cache[key] = s
                mark = ""
                if s > best_score:
                    best, best_score, mark = cand, s, "  <-- new best"
                print(f"[tune] pass{p_+1} {knob}={v}: cv={s:.4f}{mark}")
    print(f"\n[tune] BEST cv={best_score:.4f} {best} ({len(cache)} configs tried)")

    evals = {}
    dtr = lgb.Dataset(Xtr, label=ytr, group=groups_of(len(Xtr)))
    dho = lgb.Dataset(Xho, label=yho_bins, group=groups_of(len(Xho)), reference=dtr)
    params = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[10, 50],
                  label_gain=[2**i - 1 for i in range(16)], verbosity=-1,
                  seed=7, bagging_freq=1, **best)
    bst = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dho],
                    valid_names=["holdout"],
                    callbacks=[lgb.early_stopping(120, verbose=False),
                               lgb.record_evaluation(evals)])
    p = bst.predict(Xho, num_iteration=bst.best_iteration)
    n10, n50 = ndcg_at(yho_cont, p, 10), ndcg_at(yho_cont, p, 50)
    rho = pd.Series(p).corr(pd.Series(yho_cont), method="spearman")
    print(f"[tune] final: iter={bst.best_iteration} holdout NDCG@10={n10:.4f} "
          f"NDCG@50={n50:.4f} Spearman={rho:.4f}")

    bst.save_model(os.path.join(ART, "model.txt"),
                   num_iteration=bst.best_iteration)
    json.dump(feat_cols, open(os.path.join(ART, "feature_cols.json"), "w"))
    json.dump({"best_config": {k: float(v) for k, v in best.items()},
               "cv_score": best_score, "best_iteration": bst.best_iteration,
               "holdout_ndcg10": n10, "holdout_ndcg50": n50,
               "spearman": float(rho),
               "curve_ndcg10": evals["holdout"]["ndcg@10"],
               "curve_ndcg50": evals["holdout"]["ndcg@50"]},
              open(os.path.join(ART, "train_eval.json"), "w"), indent=2)

    print(f"[train] PHASE B (student) wall={time.time()-tB:.0f}s")
    print(f"[train] TOTAL wall={time.time()-t0:.0f}s device={dev}; saved "
          f"model.txt + feature_cols.json + pseudo_labels.parquet + train_eval.json "
          f"to {ART}")


if __name__ == "__main__":
    main()
