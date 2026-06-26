"""
train_student.py — tuned LambdaMART student.
  - STRATIFIED splits: 20% holdout and 4 CV folds preserve the 16 label bins
    (labeled pool is deliberately 2:1 shortlist:negatives)
  - COORDINATE-DESCENT search around the proven incumbent config (exploit the
    good initial picks instead of random exploration): one knob at a time,
    2 passes, 4-fold stratified CV, early stopping per fit
  - final refit on 80% with holdout early stop; eval curve persisted

    python train_student.py
"""
import json, os, time
import numpy as np
import pandas as pd
import lightgbm as lgb

BASE = os.environ.get("RANKER_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "artifacts_full")
GATES = {"honeypot_flag", "availability_mult", "integrity",
         "integrity_reasons", "notice_pen"}

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
    if total % size: g.append(total % size)
    return g

def ndcg_at(y_true, y_score, k):
    o = np.argsort(-y_score)[:k]
    g = (2.0 ** (4 * y_true[o]) - 1) / np.log2(np.arange(2, len(o) + 2))
    i = np.sort(y_true)[::-1][:k]
    nm = ((2.0 ** (4 * i) - 1) / np.log2(np.arange(2, len(i) + 2))).sum()
    return g.sum() / nm if nm > 0 else 0.0

def stratified_indices(y_bins, frac, seed):
    """indices of a stratified sample with `frac` of every label bin."""
    rng = np.random.default_rng(seed)
    take = []
    for b in np.unique(y_bins):
        rows = np.where(y_bins == b)[0]
        take.append(rng.choice(rows, size=max(1, int(round(frac * len(rows)))),
                               replace=False))
    return np.concatenate(take)

def stratified_folds(y_bins, k, seed):
    """fold id per row, stratified by label bin."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y_bins), dtype=int)
    for b in np.unique(y_bins):
        rows = rng.permutation(np.where(y_bins == b)[0])
        for i, r in enumerate(np.array_split(rows, k)):
            fold[r] = i
    return fold

def main():
    t0 = time.time()
    feats = pd.read_parquet(os.path.join(ART, "features.parquet"))
    ref = pd.read_parquet(os.path.join(ART, "features_refined_v3.parquet"))
    sig = pd.read_parquet(os.path.join(ART, "signals_features.parquet"))
    lab = pd.read_parquet(os.path.join(ART, "pseudo_labels_v2.parquet"))
    X_all = feats.join(ref).join(sig)
    feat_cols = [c for c in X_all.columns if c not in GATES and X_all[c].dtype != object]
    X = X_all.loc[lab.index, feat_cols].astype("float32").values
    y_cont = lab["pseudo_label"].values
    y = pd.qcut(pd.Series(y_cont).rank(method="first"), q=16,
                labels=False, duplicates="drop").values.astype(int)

    # stratified 20% holdout (by label bin)
    ho_idx = stratified_indices(y, 0.2, seed=7)
    hold = np.zeros(len(X), dtype=bool); hold[ho_idx] = True
    Xtr, ytr, yc_tr = X[~hold], y[~hold], y_cont[~hold]
    Xho, yho_bins, yho_cont = X[hold], y[hold], y_cont[hold]
    neg_frac_tr = lab["is_negative_sample"].values[~hold].mean()
    neg_frac_ho = lab["is_negative_sample"].values[hold].mean()
    print(f"[tune] split: train={len(Xtr)} holdout={len(Xho)} "
          f"neg-frac train={neg_frac_tr:.3f} holdout={neg_frac_ho:.3f}")

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
    for p in range(2):                                   # 2 coordinate passes
        for knob, values in NEIGHBORHOOD.items():
            for v in values:
                if v == best[knob]: continue
                cand = dict(best); cand[knob] = v
                key = tuple(sorted(cand.items()))
                if key in cache: continue
                s, _ = cv_score(cand); cache[key] = s
                mark = ""
                if s > best_score:
                    best, best_score, mark = cand, s, "  <-- new best"
                print(f"[tune] pass{p+1} {knob}={v}: cv={s:.4f}{mark}")
    print(f"\n[tune] BEST cv={best_score:.4f} {best} ({len(cache)} configs tried)")

    # final refit with eval curve
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
          f"NDCG@50={n50:.4f} Spearman={rho:.4f} (old: 0.8018/0.8841/0.9587)")

    json.dump({"best_config": {k: (float(v)) for k, v in best.items()},
               "cv_score": best_score, "best_iteration": bst.best_iteration,
               "holdout_ndcg10": n10, "holdout_ndcg50": n50, "spearman": float(rho),
               "curve_ndcg10": evals["holdout"]["ndcg@10"],
               "curve_ndcg50": evals["holdout"]["ndcg@50"]},
              open(os.path.join(ART, "lgbm_eval_curve.json"), "w"), indent=2)
    bst.save_model(os.path.join(ART, "model_v3.txt"), num_iteration=bst.best_iteration)
    pred = bst.predict(X_all[feat_cols].astype("float32").values,
                       num_iteration=bst.best_iteration)
    pd.DataFrame({"candidate_id": X_all.index, "lgbm_score": pred}
                 ).set_index("candidate_id").to_parquet(
                 os.path.join(ART, "lgbm_scores_v3.parquet"))
    print(f"[tune] saved model_v3.txt + lgbm_scores_v3.parquet "
          f"+ lgbm_eval_curve.json ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
