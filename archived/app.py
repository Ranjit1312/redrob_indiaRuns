"""
app.py — RedRob Ranker sandbox (Gradio). Satisfies submission_spec.docx §10.5:
upload up to 100 candidate profiles, rank them CPU-only end-to-end, download the
ranked submission CSV. Doubles as the HuggingFace Space entrypoint (SERVE mode in
the Dockerfile / entrypoint.sh).

How it works: the upload is normalised to JSONL and handed to the *unmodified*
constrained ranking step — `rank.py --candidates <upload> --out <csv>` — with
RANKER_ROOT pinned to this folder so it finds the shipped artifacts_full/. No GPU,
no network. Candidates must be drawn from the precomputed pool (their features
live in artifacts_full/); brand-new candidates would first need precompute.py
(see README). The bundled sample_candidates.json is a one-click valid demo.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "sample_candidates.json")
MAX_CANDIDATES = 100


def _to_jsonl(src_path, dst_path):
    """Accept a JSON array OR JSONL upload; write at most MAX_CANDIDATES JSONL lines."""
    with open(src_path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        records = []
        if head == "[":
            records = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    records = records[:MAX_CANDIDATES]
    with open(dst_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return len(records)


def rank(file_obj):
    if file_obj is None:
        return None, None, "Upload a candidates file (.json array or .jsonl) first."
    workdir = tempfile.mkdtemp(prefix="redrob_")
    cand = os.path.join(workdir, "candidates.jsonl")
    out = os.path.join(workdir, "submission.csv")
    try:
        n = _to_jsonl(file_obj.name, cand)
    except Exception as e:                                   # noqa: BLE001
        return None, None, f"Could not parse upload: {e}"
    if n == 0:
        return None, None, "No candidate records found in the upload."

    env = dict(os.environ, RANKER_ROOT=HERE)
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "rank.py"),
         "--candidates", cand, "--out", out, "--topk", str(min(MAX_CANDIDATES, n))],
        cwd=HERE, env=env, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 or not os.path.isfile(out):
        return None, None, f"Ranking failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"

    rows = list(csv.reader(open(out, encoding="utf-8", newline="")))
    table = rows  # header + data, rendered as a Dataframe
    status = (f"Ranked {n} candidate(s) on CPU. "
              f"{proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ''}")
    return table, out, status


def load_sample():
    return SAMPLE if os.path.isfile(SAMPLE) else None


with gr.Blocks(title="RedRob Candidate Ranker") as demo:
    gr.Markdown(
        "# RedRob Candidate Ranker — sandbox\n"
        "Upload up to **100** candidate profiles (a JSON array or JSONL, matching "
        "`candidate_schema.json`) and get the ranked `submission.csv`. Runs CPU-only, "
        "offline, in seconds. Click **Load sample** to try the bundled candidates.")
    with gr.Row():
        upload = gr.File(label="candidates (.json / .jsonl)", file_types=[".json", ".jsonl"])
        with gr.Column():
            sample_btn = gr.Button("Load sample")
            run_btn = gr.Button("Rank candidates", variant="primary")
    status = gr.Markdown()
    table = gr.Dataframe(headers=["candidate_id", "rank", "score", "reasoning"],
                         label="Ranking", wrap=True)
    download = gr.File(label="Download submission.csv")

    sample_btn.click(load_sample, outputs=upload)
    run_btn.click(rank, inputs=upload, outputs=[table, download, status])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
