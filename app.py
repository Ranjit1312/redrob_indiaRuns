"""
app.py — RedRob v7 ranker sandbox (Gradio). Satisfies submission_spec §10.5
(a working demo) and doubles as the HuggingFace Space entrypoint (SERVE mode in
the Dockerfile / entrypoint.sh).

HOW IT WORKS — honest about the v7 lifecycle:
  The constrained rank step (`rank.py`) ranks the *precomputed* candidate pool in
  `artifacts_v7/` (embeddings, evidence, intrinsic facts, JD vectors, BM25 facets,
  trained student) and returns the top 100. Producing those artifacts is the GPU
  precompute phase (embed_candidates -> jd_compile -> train), which a hosted Space
  can't run. So this sandbox ranks the *bundled demo pool* — the candidates the
  shipped `artifacts_v7/` were built from — CPU-only, offline, in seconds, exactly
  as the judged container does.

  RANKER_ROOT is pinned to this folder so rank.py finds the shipped artifacts_v7/.
  The bundled pool jsonl (POOL_JSONL env, default: sample_candidates.json
  normalized to JSONL) is passed as --candidates so the top-k reasoning can stream
  full profiles. No GPU, no network.

To deploy / refresh the Space: run the precompute phase over your demo pool, then
commit the resulting artifacts_v7/ and the matching pool jsonl alongside this app.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts_v7")
SAMPLE_JSON = os.path.join(HERE, "sample_candidates.json")
TOPK = 100


def _ensure_pool_jsonl():
    """Resolve the demo pool the shipped artifacts_v7/ were built from.

    Prefer $POOL_JSONL, then a bundled pool.jsonl, else normalize the bundled
    sample_candidates.json (a JSON array) into pool.jsonl once.
    """
    env = os.environ.get("POOL_JSONL")
    if env and os.path.isfile(env):
        return env
    bundled = os.path.join(HERE, "pool.jsonl")
    if os.path.isfile(bundled):
        return bundled
    if os.path.isfile(SAMPLE_JSON):
        with open(SAMPLE_JSON, "r", encoding="utf-8") as f:
            records = json.load(f)
        with open(bundled, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return bundled
    return None


POOL_JSONL = _ensure_pool_jsonl()


def _jd_summary():
    art_jd = os.path.join(ART, "jd_profile.yaml")
    src_jd = os.path.join(HERE, "jd", "jd_profile.yaml")
    path = art_jd if os.path.isfile(art_jd) else src_jd
    if not os.path.isfile(path):
        return "_(no jd_profile.yaml found)_"
    try:
        import yaml
        jd = yaml.safe_load(open(path, encoding="utf-8"))
        role = jd.get("role", {})
        sigs = ", ".join(s.get("id", "?") for s in jd.get("signals", []))
        return (f"**Target role:** {role.get('title','?')} "
                f"@ {role.get('company','?')}  \n"
                f"**Signals:** {sigs}")
    except Exception:  # noqa: BLE001
        return "_(jd_profile.yaml present)_"


def rank_pool():
    if not POOL_JSONL:
        return None, None, "No demo pool found (expected artifacts_v7/ + a pool jsonl)."
    if not os.path.isdir(ART):
        return None, None, ("artifacts_v7/ is missing — run the precompute phase and "
                            "commit it into the Space (see README).")

    workdir = tempfile.mkdtemp(prefix="redrob_")
    out = os.path.join(workdir, "submission.csv")
    env = dict(os.environ, RANKER_ROOT=HERE)
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "rank.py"),
         "--candidates", POOL_JSONL, "--out", out, "--topk", str(TOPK)],
        cwd=HERE, env=env, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 or not os.path.isfile(out):
        tail = (proc.stdout[-1500:] + "\n" + proc.stderr[-1500:]).strip()
        return None, None, f"Ranking failed:\n```\n{tail}\n```"

    rows = list(csv.reader(open(out, encoding="utf-8", newline="")))

    # surface the rank.py telemetry (wall / peak RAM / budget headroom)
    tele_path = os.path.join(HERE, "telemetry.json")
    status = f"Ranked {len(rows) - 1} candidates on CPU."
    if os.path.isfile(tele_path):
        t = json.load(open(tele_path, encoding="utf-8"))
        hr = t.get("headroom", {})
        status = (f"Ranked **{t.get('n_candidates','?')}** candidates -> top "
                  f"**{len(rows) - 1}**, CPU-only.  \n"
                  f"wall **{t.get('total_wall_s','?')}s** "
                  f"({hr.get('wall_pct_used','?')}% of the 5-min budget) · "
                  f"peak RAM **{t.get('peak_memory_gb','?')}GB** "
                  f"({hr.get('ram_pct_used','?')}% of 16GB) · "
                  f"artifacts **{t.get('artifact_total_mb','?')}MB**")
    return rows, out, status


with gr.Blocks(title="RedRob v7 Candidate Ranker") as demo:
    gr.Markdown(
        "# RedRob Candidate Ranker — v7 sandbox\n"
        "Ranks the **precomputed demo pool** for the target JD and returns the "
        "spec-compliant top-100 `submission.csv` — CPU-only, offline, in seconds, "
        "exactly as the judged container runs `rank.py`.")
    gr.Markdown(_jd_summary())
    run_btn = gr.Button("Rank candidates", variant="primary")
    status = gr.Markdown()
    table = gr.Dataframe(headers=["candidate_id", "rank", "score", "reasoning"],
                         label="Ranking (top 100)", wrap=True)
    download = gr.File(label="Download submission.csv")

    run_btn.click(rank_pool, outputs=[table, download, status])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
