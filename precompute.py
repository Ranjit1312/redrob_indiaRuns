"""
precompute.py — one-time OFFLINE pipeline orchestrator (UNCONSTRAINED: may use
GPU/network; this is NOT the graded ranking step). Runs the five precompute
stages in dependency order, each as its own process, with a banner + wall time:

    1. run_step35.py          bi-encoder embeddings + BM25 + structured features
                              (GPU auto-used by sentence-transformers if present)
    2. build_rule_features.py JD-rule/evidence/integrity features
                              -> artifacts_full/features_refined_v3.parquet
    3. precompute_teacher.py  cross-encoder teacher (bge-reranker-v2-m3) ->
                              pseudo labels + signals_features.parquet
    4. train_student.py       tuned LightGBM LambdaMART student
                              -> model_v3.txt + feature_cols_v2.json
    5. build_final_features.py full-profile-audit flags + blend
                              -> artifacts_full/features_v4.parquet

After this completes the constrained ranking step is just:

    python rank.py --candidates ../../candidates.jsonl --out ./submission.csv

------------------------------------------------------------------------------
INCREMENTAL / NEW-DATASET BEHAVIOUR  (the candidate_id-vs-parquet check)
------------------------------------------------------------------------------
On startup precompute.py diffs the candidate_ids in the supplied
candidates.jsonl against those already covered by artifacts_full/features.parquet:

  * ALL covered + artifacts present  -> NO-OP. Nothing is recomputed; it tells you
    to run rank.py directly. (This is the case for the shipped official pool.)

  * Some ids MISSING (organisers ran us on a NEW candidates file) -> the ONLY
    expensive, GPU-bound stage (step 1, the bi-encoder embeddings) is run for
    JUST the missing candidates and merged into features.parquet. The remaining
    CPU stages (2-5) then re-derive the rule/teacher/student/audit artifacts over
    the full pool so every derived table stays consistent. Because only step 1 is
    embedding-bound, "precompute only the new candidates" is exactly what happens.

Flags:
  --candidates PATH     candidates.jsonl to cover (default: <root>/candidates.jsonl)
  --skip-embeddings     skip step 1 even if ids are missing (use shipped embeddings)
  --full-recompute      re-embed the ENTIRE pool instead of just the missing ids
"""
import argparse, os, subprocess, sys, time, re

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("RANKER_ROOT") or os.path.dirname(os.path.dirname(HERE))
ART = os.path.join(BASE, "artifacts_full")

_CID_RE = re.compile(rb'"candidate_id"\s*:\s*"(CAND_\d+)"')


def scan_candidate_ids(path):
    ids = set()
    with open(path, "rb") as f:
        for line in f:
            m = _CID_RE.search(line)
            if m:
                ids.add(m.group(1).decode())
    return ids


def covered_ids():
    """candidate_ids already present in artifacts_full/features.parquet (index-only read)."""
    fp = os.path.join(ART, "features.parquet")
    if not os.path.isfile(fp):
        return set()
    import pandas as pd
    return set(pd.read_parquet(fp, columns=[]).index.astype(str))


def write_subset(candidates, ids, out_path):
    """Stream candidates.jsonl and write only the lines whose candidate_id is in `ids`."""
    n = 0
    with open(candidates, "rb") as fin, open(out_path, "wb") as fout:
        for line in fin:
            m = _CID_RE.search(line)
            if m and m.group(1).decode() in ids:
                fout.write(line)
                n += 1
    return n


def merge_parquet(new_dir, names):
    """Concat freshly-computed subset parquets into the existing ART parquets (new rows win)."""
    import pandas as pd
    for name in names:
        new_fp = os.path.join(new_dir, name)
        old_fp = os.path.join(ART, name)
        if not os.path.isfile(new_fp):
            continue
        new = pd.read_parquet(new_fp)
        if os.path.isfile(old_fp):
            old = pd.read_parquet(old_fp)
            merged = pd.concat([old[~old.index.isin(new.index)], new])
        else:
            merged = new
        merged.to_parquet(old_fp)
        print(f"[precompute]   merged {name}: +{len(new)} new rows -> {len(merged)} total")


def gpu_banner():
    try:
        import torch
        avail = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if avail else "none (CPU fallback)"
        print(f"[gpu] torch.cuda.is_available() = {avail}; device = {dev}")
    except ImportError:
        print("[gpu] torch not installed — embedding/teacher steps will run on "
              "CPU (or fail if sentence-transformers is missing)")


def run_step(n, total, script, extra_args=(), cwd=HERE):
    print("\n" + "=" * 78)
    print(f"[precompute] step {n}/{total}: {script} {' '.join(extra_args)}")
    print("=" * 78, flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, os.path.join(HERE, script), *extra_args],
                   cwd=cwd, check=True)
    print(f"[precompute] step {n}/{total} ({script}) done in {time.time() - t0:.0f}s",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default=os.path.join(BASE, "candidates.jsonl"),
                    help="candidates.jsonl to cover")
    ap.add_argument("--skip-embeddings", action="store_true",
                    help="skip step 1 even if ids are missing")
    ap.add_argument("--full-recompute", action="store_true",
                    help="re-embed the entire pool instead of only the missing ids")
    args = ap.parse_args()

    t_all = time.time()
    os.makedirs(ART, exist_ok=True)
    gpu_banner()

    # ---- coverage diff: which candidate_ids do we still need to embed? -------------
    wanted = scan_candidate_ids(args.candidates)
    have = covered_ids()
    missing = wanted - have
    print(f"[precompute] coverage: {len(wanted)} candidates supplied; "
          f"{len(wanted & have)} already in features.parquet; {len(missing)} missing.")

    have_artifacts = all(os.path.isfile(os.path.join(ART, a)) for a in
                         ("features.parquet", "features_refined_v3.parquet",
                          "features_v4.parquet", "signals_features.parquet",
                          "model_v3.txt", "feature_cols_v2.json"))

    if not missing and have_artifacts and not args.full_recompute:
        print("[precompute] ✓ all candidates already covered and every artifact present "
              "— nothing to precompute.")
        print("[precompute] run:  python rank.py --candidates "
              f"{args.candidates} --out ./submission.csv")
        return

    # ---- step 1: embeddings (the only GPU/expensive stage), incremental if possible ----
    incr_dir = os.path.join(ART, "_incremental")
    if args.skip_embeddings:
        print("[precompute] --skip-embeddings: step 1 skipped — using shipped embeddings in", ART)
    elif missing and have_artifacts and not args.full_recompute:
        os.makedirs(incr_dir, exist_ok=True)
        subset = os.path.join(incr_dir, "candidates_missing.jsonl")
        n = write_subset(args.candidates, missing, subset)
        print(f"[precompute] INCREMENTAL embeddings for {n} NEW candidates only "
              f"(full pool stays untouched).")
        run_step(1, 5, "run_step35.py",
                 ("--candidates", subset, "--out-dir", incr_dir,
                  "--topk", str(min(100, n))))
        merge_parquet(incr_dir, ("features.parquet", "meta.parquet",
                                 "scores_step35.parquet"))
    else:
        print("[precompute] FULL embeddings over the entire supplied pool.")
        run_step(1, 5, "run_step35.py",
                 ("--candidates", args.candidates, "--out-dir", ART))

    # ---- steps 2-5: CPU stages re-derive over the FULL pool for consistency -----------
    # (they stream the repo-root candidates.jsonl / read the merged features.parquet)
    for n, (script, extra) in enumerate(
            [("build_rule_features.py", ()),
             ("precompute_teacher.py", ()),
             ("train_student.py", ()),
             ("build_final_features.py", ())], start=2):
        run_step(n, 5, script, extra)

    print("\n" + "=" * 78)
    print(f"[precompute] ALL steps done in {(time.time() - t_all)/60:.1f} min. Artifacts in {ART}")
    print("[precompute] now run:  python rank.py --candidates "
          f"{args.candidates} --out ./submission.csv")


if __name__ == "__main__":
    main()
