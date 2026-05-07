# cloud_runner — Modal launcher prototype

Kicks off an ActivitySim run on Modal: clones a branch, runs the model,
zips outputs, and stores the zip on a Modal Volume for download. The
Modal container shuts down automatically when the run finishes, so
billing stops the moment the zip is written.

This is a CLI prototype for the eventual SaaS service. It is **not**
shipped as part of the OSS ActivitySim package — it lives in this
private working branch only.

## One-time setup

```bash
pip install modal
modal token new        # browser-based auth to your Modal workspace
```

If you'll run against private repos, register a GitHub token as a Modal
secret named `github-token` (the launcher injects it as `GITHUB_TOKEN`
inside the container):

```bash
modal secret create github-token GITHUB_TOKEN=ghp_xxx
```

For public repos no secret is needed.

## Smoke test (matches the local run we already validated)

30-household sample on the SANDAG 1-zone example. ~2 min wall-clock,
costs cents.

```bash
modal run cloud_runner/launch.py \
  --branch claude/model-run-management-tools-QvSyd \
  --sample-size 30 \
  --cpu 4 --memory-gb 8 --timeout-hours 0.5
```

When it finishes you'll see:

```
Download: modal volume get activitysim-runs <run-id>/output.zip ./<run-id>.zip
```

Run that command locally to pull the zip. The container is already gone.

## Full run (scale up)

```bash
modal run cloud_runner/launch.py \
  --branch main \
  --sample-size 0 \
  --settings-file settings_mp.yaml \
  --cpu 24 --memory-gb 64 --timeout-hours 24
```

`--sample-size 0` means "no override" — uses whatever the settings file
declares (production size). `settings_mp.yaml` enables multiprocessing.

## Parameters

| Flag | Default | Notes |
|---|---|---|
| `--repo-url` | this repo | Any clonable git URL |
| `--branch` | dev branch | Branch or tag |
| `--sample-size` | `30` | `0` = no CLI override |
| `--config-dirs` | SANDAG 1-zone + MTC | Comma-separated, applied left-to-right |
| `--data-dir` | placeholder SANDAG `data_1` | Path inside the cloned repo |
| `--settings-file` | empty | e.g. `settings_mp.yaml` for multiprocessing |
| `--chunk-size` | empty | passes to `-g` |
| `--cpu` | `4` | cores |
| `--memory-gb` | `8` | RAM |
| `--timeout-hours` | `1.0` | hard cap; container is killed past this |

## What it produces

Inside the zip:
- All `final_*.csv` tables
- `trips_*.omx` matrices
- `manifest.json` with run id, commit SHA, branch, timing, parameters
- `pipeline.parquetpipeline/` checkpoints
- Logs

## Cost estimate

Modal CPU pricing is roughly $0.20/CPU-hour (US region). Approximate
per-run costs at typical sizes:

| Run | Resources | Wall time | Cost |
|---|---|---|---|
| Smoke (30 hh) | 4 cores, 8 GB | ~2 min | ~$0.03 |
| Mid (50k hh, MP) | 24 cores, 64 GB | ~30 min | ~$2.40 |
| Full SANDAG (3.5M hh, MP+sharrow) | 24 cores, 64 GB | 3–12 hr | ~$15–60 |

The first $30/mo of compute is free on Modal's starter tier.

## How auto-shutdown works

The Modal `@app.function` decorator turns the function body into the
container's entire lifecycle. Modal spins up a container, runs the
function, and tears the container down the moment it returns. There's
no idle period to pay for. The `timeout` parameter is a hard ceiling
that kills the container if the run hangs.

## Next steps (out of scope for this prototype)

1. GitHub App + web UI for non-CLI users
2. Presigned S3/R2 download URLs (so users don't need the Modal CLI)
3. Webhook on `workflow_dispatch` from a customer's repo to trigger runs
4. Tenant isolation (one Modal workspace per customer, or scoped volumes)
5. Run history, logs UI, billing
