"""
validate_submission.py — offline pre-flight validator for the Redrob
submission CSV (mirrors the spec's Stage-1 auto-validator checks).

    python validate_submission.py --submission submission.csv
    python validate_submission.py --submission submission.csv --candidates ../../candidates.jsonl
"""
import argparse, csv, os, re, sys

CAND_ID_RE = re.compile(r"^CAND_\d+$")

def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok

def warn(name, detail=""):
    print(f"[WARN] {name}" + (f" — {detail}" if detail else ""))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True, help="path to submission CSV")
    ap.add_argument("--candidates", default=None,
                    help="optional path to candidates.jsonl to verify ids exist")
    args = ap.parse_args()

    ok = True

    # ---- file exists, UTF-8 ----------------------------------------------------
    if not check("file exists", os.path.isfile(args.submission), args.submission):
        sys.exit(1)
    try:
        with open(args.submission, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        check("file is valid UTF-8", True)
    except UnicodeDecodeError as e:
        check("file is valid UTF-8", False, str(e))
        sys.exit(1)

    # ---- header -----------------------------------------------------------------
    expected = ["candidate_id", "rank", "score", "reasoning"]
    header = rows[0] if rows else []
    ok &= check("header is exactly candidate_id,rank,score,reasoning",
                header == expected, f"got {header}")

    data = rows[1:]
    ok &= check("exactly 100 data rows", len(data) == 100, f"got {len(data)}")

    # ---- per-row parsing ----------------------------------------------------------
    ids, ranks, scores, n_empty_reason = [], [], [], 0
    parse_ok = True
    for i, row in enumerate(data, start=2):
        if len(row) != 4:
            parse_ok = False
            print(f"       line {i}: expected 4 columns, got {len(row)}")
            continue
        cid, rank_s, score_s, reasoning = row
        ids.append(cid)
        try:
            ranks.append(int(rank_s))
        except ValueError:
            parse_ok = False
            print(f"       line {i}: rank not an int: {rank_s!r}")
        try:
            scores.append(float(score_s))
        except ValueError:
            parse_ok = False
            print(f"       line {i}: score not a float: {score_s!r}")
        if not reasoning.strip():
            n_empty_reason += 1
    ok &= check("all rows parse (4 cols, int rank, float score)", parse_ok)

    # ---- ranks ---------------------------------------------------------------------
    ok &= check("ranks are 1..100, each exactly once",
                sorted(ranks) == list(range(1, 101)),
                f"got {len(set(ranks))} unique of {len(ranks)}")

    # ---- candidate ids ---------------------------------------------------------------
    ok &= check("candidate_ids unique", len(set(ids)) == len(ids),
                f"{len(ids) - len(set(ids))} duplicates")
    bad = [c for c in ids if not CAND_ID_RE.match(c)]
    ok &= check("candidate_ids match CAND_\\d+", not bad,
                f"bad: {bad[:5]}" if bad else "")

    # ---- scores -----------------------------------------------------------------------
    if scores and len(scores) == len(ranks) == 100:
        by_rank = [s for _, s in sorted(zip(ranks, scores))]
        non_inc = all(by_rank[i] >= by_rank[i + 1] - 1e-12 for i in range(len(by_rank) - 1))
        ok &= check("scores monotonically non-increasing with rank", non_inc)
        ok &= check("scores not all identical", len(set(scores)) > 1)
    else:
        ok &= check("scores monotonically non-increasing with rank", False,
                    "could not evaluate (row parse errors)")

    # ---- reasoning (warn only) -----------------------------------------------------------
    if n_empty_reason:
        warn("empty reasoning strings", f"{n_empty_reason} rows (penalized at Stage 4, not auto-rejected)")
    else:
        print("[PASS] no empty reasoning strings")

    # ---- ids exist in candidates.jsonl ----------------------------------------------------
    if args.candidates:
        if not os.path.isfile(args.candidates):
            ok &= check("candidates.jsonl exists", False, args.candidates)
        else:
            missing = set(ids)
            needles = {c: f'"{c}"'.encode() for c in missing}
            with open(args.candidates, "rb") as f:
                for line in f:
                    found = [c for c in missing if needles[c] in line]
                    for c in found:
                        missing.discard(c)
                    if not missing:
                        break
            ok &= check("every candidate_id exists in candidates.jsonl",
                        not missing, f"missing: {sorted(missing)[:5]}" if missing else "")

    print()
    if ok:
        print("RESULT: PASS — submission is spec-compliant.")
        sys.exit(0)
    print("RESULT: FAIL — fix the failures above before submitting.")
    sys.exit(1)

if __name__ == "__main__":
    main()
