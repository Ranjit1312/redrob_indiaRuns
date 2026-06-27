# Docker runbook — RedRob v7 (build, run, capture telemetry)

All commands are **Windows PowerShell**, run from the repo root. `Tee-Object`
writes a log AND prints to screen — keep the `*.log` files; they're the source
for the README telemetry section.

The image has three modes (one Dockerfile, selected by `--build-arg ENV_MODE`):

| Mode | Deps | Hardware | What it does |
|------|------|----------|--------------|
| `RANK` | CPU only (no torch) | CPU, <5 min, <=16 GB | the judged step: `rank.py` -> `submission.csv` -> `validate` |
| `PRECOMPUTE` | full ML + GPU | GPU, minutes | builds `artifacts_v7/` (embed -> jd_compile -> features -> train) |
| `SERVE` | CPU + gradio | CPU | the HuggingFace Space sandbox (`app.py` on :7860) |

Ordering matters: **PRECOMPUTE writes `./artifacts_v7/`, then the RANK/SERVE
images bake it in** (`COPY . .`). So precompute first, build rank second.

---

## 0. One-time: the bracketed path

The folder name `[PUB] ...` contains PowerShell wildcard chars. Docker takes
`${PWD}` as a literal string, so the volume mounts below are fine — but if any
`-v` mount ever errors on the brackets, `cd` out and use the absolute path
quoted, e.g. `-v "C:\Users\jadha\Downloads\[PUB] ...\candidates.jsonl:/app/candidates.jsonl:ro"`.

---

## 1. PRECOMPUTE — build `artifacts_v7/` (GPU, one-off per pool/JD)

```powershell
# build the precompute image (full ML stack). REBUILD whenever any *.py changes.
docker build -t redrob:precompute --build-arg ENV_MODE=PRECOMPUTE .
```

### 1a. Verify the GPU FIRST (it silently falls back to CPU otherwise)

None of the scripts error when the GPU is invisible — they just run on CPU and
crawl. So confirm before a full run:

```powershell
# (i) does OUR image's torch see the GPU? must print: ... True
docker run --rm --gpus all --entrypoint python redrob:precompute `
  -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# (ii) if that prints False, does ANY container see the GPU? (driver-only test)
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
#   prints the GPU table -> plumbing OK; torch wheel vs driver mismatch (see note below)
#   errors              -> Docker can't pass the GPU into WSL2: run
#                          `wsl --update; wsl --shutdown`, restart Docker Desktop
#                          (Settings: WSL2 backend + GPU enabled), retry (i)
```

> Wheel/driver mismatch: the default `torch` wheel may bundle a newer CUDA than
> your driver supports. The driver runs any *older* CUDA runtime, so pin a
> matching wheel in `requirements-precompute.txt`, e.g.
> `--extra-index-url https://download.pytorch.org/whl/cu128` + `torch==2.12.1`,
> then rebuild. Check your driver's max CUDA with `nvidia-smi`.

### 1b. Run precompute end-to-end

```powershell
# mount the candidate pool (read-only) + an artifacts_v7 output dir.
# the hf_cache volume avoids re-downloading models on re-runs.
# HF_HUB_OFFLINE=1 skips the hub (no network, no "set HF_TOKEN" warning) —
#   FIRST run: DROP it so the models can download; add it on every run after.
New-Item -ItemType Directory -Force artifacts_v7 | Out-Null
docker run --rm --gpus all `
  -e ENV_MODE=PRECOMPUTE -e HF_HUB_OFFLINE=1 `
  -v "${PWD}\candidates.jsonl:/app/candidates.jsonl:ro" `
  -v "${PWD}\artifacts_v7:/app/artifacts_v7" `
  -v "redrob_hf_cache:/root/.cache/huggingface" `
  redrob:precompute 2>&1 | Tee-Object -FilePath precompute.log

# CONFIRM it actually used the GPU (every stage should say device=cuda):
Select-String -Path precompute.log -Pattern "device=|cuda_available|scanned with"
```

> The JSONL scan now parses in parallel (`embed_candidates.py` auto-uses
> `os.cpu_count()` processes; the log prints `scanned with N workers`). To
> compare against the old serial path, run that step with `--workers 1`.

After this, `./artifacts_v7/` holds: `job_embeddings.npy`, `summary_embeddings.npy`,
`job_offsets.npy`, `evidence_texts.parquet`, `intrinsic.parquet`, `jd_vectors.npy`,
`bm25_facets.parquet`, `jd_profile.yaml`, `model.txt`, `feature_cols.json`,
`features_v7.parquet`, `pseudo_labels.parquet`, `train_eval.json`.

> Local alternative (no Docker, if you have the precompute venv with torch):
> ```powershell
> $env:RANKER_ROOT = $PWD.Path
> python embed_candidates.py --candidates .\candidates.jsonl
> python jd_compile.py
> python rank.py --candidates .\candidates.jsonl --features-only
> python train.py
> ```

---

## 2. RANK — the judged container (CPU, <5 min) + headline telemetry

```powershell
# build AFTER precompute, so COPY . . bakes artifacts_v7/ into the image
docker build -t redrob:rank --build-arg ENV_MODE=RANK .

# run the constrained step. --network none + --memory 16g + --cpus prove the
# budget. writes submission.csv (mounted out) and telemetry.json inside /app.
docker run --rm `
  --network none --memory 16g --cpus 8 `
  -e ENV_MODE=RANK `
  -v "${PWD}\candidates.jsonl:/app/candidates.jsonl:ro" `
  -v "${PWD}\out_docker:/app/out_docker" `
  -e OUT=/app/out_docker/submission.csv `
  redrob:rank 2>&1 | Tee-Object -FilePath rank.docker.log

# copy the telemetry the container wrote (rank.py writes it next to rank.py = /app)
docker create --name redrob_tmp redrob:rank | Out-Null
docker cp redrob_tmp:/app/telemetry.json .\telemetry.docker.json 2>$null
docker rm redrob_tmp | Out-Null
```

`rank.docker.log` ends with the `[rank] TOTAL wall=… peak_ram=… artifacts=…`
line — that, plus `telemetry.json`, is what I'll fold into the README.

> Local alternative (uses the CPU rank `.venv`, no torch needed):
> ```powershell
> $env:RANKER_ROOT = $PWD.Path
> .\.venv\Scripts\python rank.py --candidates .\candidates.jsonl --out .\submission.csv 2>&1 | Tee-Object -FilePath rank.local.log
> .\.venv\Scripts\python validate_submission.py --submission .\submission.csv
> ```
> `rank.py` writes `telemetry.json` next to itself either way.

---

## 3. SERVE — the HuggingFace Space sandbox

Test locally first:

```powershell
docker build -t redrob:serve --build-arg ENV_MODE=SERVE .
docker run --rm -p 7860:7860 -e ENV_MODE=SERVE redrob:serve
# open http://localhost:7860  -> click "Rank candidates"
```

The Space ranks the **bundled demo pool** (`app.py` normalizes
`sample_candidates.json` -> `pool.jsonl`, or set `POOL_JSONL`). For the hosted
Space, the shipped `artifacts_v7/` must have been precomputed over that same pool.

Deploy to a **Docker** Space (HF auto-detects the `Dockerfile`):

```powershell
# the Space needs: Dockerfile, entrypoint.sh, app.py, rank.py, validate_submission.py,
# redrob_ranker/, jd/, sample_candidates.json (or pool.jsonl), and artifacts_v7/.
# add to the Space's README.md front-matter:  sdk: docker   app_port: 7860
# then push to the Space remote (artifacts_v7 may need git-lfs for the .npy files):
git remote add space https://huggingface.co/spaces/<user>/redrob-v7
git push space main
```

---

## Telemetry to capture for the README

Save these and hand them back to me:

- `rank.docker.log` (or `rank.local.log`) — the judged run's stage timings
- `telemetry.docker.json` (or `telemetry.json`) — wall / peak RAM / artifact MB / headroom
- `precompute.log` — embed + train wall times and device lines
- `artifacts_v7/train_eval.json` — holdout NDCG@10/@50 + Spearman (model quality)
- the `validate_submission.py` PASS block from the rank log

I'll turn those into the README's "Reproducibility & budget" section before the push.
```
