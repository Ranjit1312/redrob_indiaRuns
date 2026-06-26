"""
embed_candidates.py — v7 component 1: the JD-INDEPENDENT candidate pass (GPU).

Streams --candidates jsonl ONCE and saves to <RANKER_ROOT|repo>/artifacts_v7/:

    job_embeddings.npy       (n_jobs_total, 384)  embedder, L2-normed
    summary_embeddings.npy   (n_candidates, 384)  headline+summary per candidate
    job_offsets.npy          (n_candidates, 2)    [start, end) rows per candidate
    evidence_texts.parquet   candidate_id, headline_summary, jobs_text (per-job
                             "title. description" chunks joined with \\x1f),
                             jobs_meta (json per-job title/company/industry/
                             duration_months/end_date/is_current)
    intrinsic.parquet        the typed JD-independent candidate facts — built by
                             redrob_ranker.intrinsic.extract_intrinsic (the
                             SINGLE source of that logic; not re-implemented here)

Deliberately JD-INDEPENDENT: no facet similarities, no evidence regexes, no
rule features. Everything JD-flavored happens in jd_compile.py (queries) and
rank.py / features.py (live feature computation). When the JD changes, this
step's outputs survive untouched.

Reads the embedder id from method_config.yaml (models.embedder). Prints the
torch device.

    python embed_candidates.py --candidates ./candidates.jsonl
"""
import argparse, os, time

import numpy as np
import pandas as pd

from redrob_ranker import profile as rprofile
from redrob_ranker.intrinsic import extract_intrinsic

BASE = os.environ.get("RANKER_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(BASE, "artifacts_v7")
HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_DEFAULT = os.path.join(HERE, "jd", "method_config.yaml")
JD_DEFAULT = os.path.join(HERE, "jd", "jd_profile.yaml")

SEP = "\x1f"   # unit separator: joins per-job chunks inside jobs_text


def career_text(job):
    return f"{job.get('title', '')}. {job.get('description', '')}"


def summary_text(c):
    p = c["profile"]
    return f"{p.get('headline', '')}. {p.get('summary') or ''}"


def _scan_shard(task):
    """Worker: parse one [start, end) byte range of the candidates file and
    return compact payloads. Raw candidate dicts never cross the process
    boundary — extract_intrinsic runs IN-WORKER and only its (small) DataFrame
    comes back. Byte boundaries are realigned to whole lines so every candidate
    record is processed exactly once regardless of where the split byte lands.

    Returns (job_texts, summaries, evid_rows, chunk_counts, intrinsic_df), all
    in file order; the parent concatenates shards in task order (== file order).
    """
    path, start, end = task
    import orjson
    job_texts, summaries, evid_rows, chunk_counts, records = [], [], [], [], []
    with open(path, "rb") as f:
        if start:
            # Realign to a line boundary. If byte[start-1] is '\n', `start` is a
            # line beginning -> keep it; else we're mid-line -> drop the partial
            # (the previous shard reads past its end to finish that line).
            f.seek(start - 1)
            if f.read(1) != b"\n":
                f.readline()
        while f.tell() < end:           # own every line whose START is < end
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            c = orjson.loads(line)
            jobs = c.get("career_history") or []
            chunks = [career_text(j) for j in jobs]
            metas = [{"t": j.get("title") or "", "c": j.get("company") or "",
                      "i": j.get("industry") or "", "d": j.get("duration_months") or 0,
                      "e": j.get("end_date") or "", "cur": bool(j.get("is_current"))}
                     for j in jobs]
            job_texts.extend(chunks)
            chunk_counts.append(len(chunks))
            summaries.append(summary_text(c))
            evid_rows.append({"candidate_id": c["candidate_id"],
                              "headline_summary": summary_text(c),
                              "jobs_text": SEP.join(chunks),
                              "jobs_meta": orjson.dumps(metas).decode()})
            records.append(c)
    return job_texts, summaries, evid_rows, chunk_counts, extract_intrinsic(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=os.path.join(BASE, "candidates.jsonl"))
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel JSONL-scan processes (default: os.cpu_count(); "
                         "use 1 to force the serial path)")
    ap.add_argument("--method", default=METHOD_DEFAULT)
    ap.add_argument("--jd", default=JD_DEFAULT)
    args = ap.parse_args()
    os.makedirs(ART, exist_ok=True)
    t0 = time.time()

    # model id from the JD-stable method config
    _, method = rprofile.load(args.jd, args.method)
    EMBED_MODEL = method.models["embedder"]["name"]
    BATCH = args.batch_size or int(method.models["embedder"].get("batch_size", 256))

    # ---- 1. parallel streaming scan ---------------------------------------
    # The scan (orjson parse + per-record intrinsic extraction + evidence-text
    # build) is the single-threaded bottleneck of the candidate pass, so split
    # the file into ~equal byte ranges and parse them across processes. Each
    # worker runs extract_intrinsic on its own shard (the single source of
    # intrinsic logic) and returns only compact results; the parent stitches
    # them back IN FILE ORDER, so job_offsets / embeddings / intrinsic all stay
    # aligned and the artifacts are byte-identical to the serial path.
    file_size = os.path.getsize(args.candidates)
    workers = max(1, args.workers or (os.cpu_count() or 1))
    if workers > 1 and file_size > (8 << 20):
        bounds = [file_size * i // workers for i in range(workers + 1)]
        tasks = [(args.candidates, bounds[i], bounds[i + 1]) for i in range(workers)]
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            shards = pool.map(_scan_shard, tasks)
        print(f"[embed] scanned with {workers} workers")
    else:
        shards = [_scan_shard((args.candidates, 0, file_size))]

    all_job_texts, all_summaries, evid_rows, chunk_counts, intr_parts = \
        [], [], [], [], []
    for jt, sm, ev, cc, idf in shards:
        all_job_texts.extend(jt)
        all_summaries.extend(sm)
        evid_rows.extend(ev)
        chunk_counts.extend(cc)
        intr_parts.append(idf)

    # rebuild per-candidate [start, end) job-chunk offsets from the chunk counts
    counts = np.asarray(chunk_counts, dtype=np.int64)
    ends = np.cumsum(counts)
    starts = ends - counts
    job_offsets = list(zip(starts.tolist(), ends.tolist()))

    # ---- intrinsic facts (concatenated shards, file order) ----------------
    intr_df = pd.concat(intr_parts) if intr_parts else extract_intrinsic([])
    n = len(intr_df)
    print(f"[embed] scanned {n} candidates, {len(all_job_texts)} job chunks "
          f"({time.time()-t0:.0f}s)")

    # ---- 2. parquet artifacts ---------------------------------------------
    pd.DataFrame(evid_rows).set_index("candidate_id").to_parquet(
        os.path.join(ART, "evidence_texts.parquet"))
    intr_df.to_parquet(os.path.join(ART, "intrinsic.parquet"))
    np.save(os.path.join(ART, "job_offsets.npy"),
            np.asarray(job_offsets, dtype=np.int64))
    print(f"[embed] evidence_texts.parquet + intrinsic.parquet + job_offsets.npy "
          f"written ({time.time()-t0:.0f}s)")

    # ---- 3. GPU embedding pass --------------------------------------------
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL, device=dev)
    print(f"[embed] {EMBED_MODEL} loaded on device={model.device} "
          f"(cuda_available={torch.cuda.is_available()})")

    t1 = time.time()
    job_matrix = (model.encode(
        [t if t else " " for t in all_job_texts], normalize_embeddings=True,
        batch_size=BATCH, show_progress_bar=True).astype(np.float32)
        if all_job_texts else np.zeros((0, 384), np.float32))
    print(f"[embed] job chunks embedded in {time.time()-t1:.0f}s")
    t2 = time.time()
    summ_matrix = model.encode(
        [t if t else " " for t in all_summaries], normalize_embeddings=True,
        batch_size=BATCH, show_progress_bar=True).astype(np.float32)
    print(f"[embed] summaries embedded in {time.time()-t2:.0f}s")

    np.save(os.path.join(ART, "job_embeddings.npy"), job_matrix)
    np.save(os.path.join(ART, "summary_embeddings.npy"), summ_matrix)

    print(f"[embed] DONE wall={time.time()-t0:.0f}s device={dev} "
          f"job_embeddings={job_matrix.shape} summary_embeddings={summ_matrix.shape}")
    print(f"[embed] artifacts in {ART} (all JD-independent)")


if __name__ == "__main__":
    main()
