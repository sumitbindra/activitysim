# asim-harness

A minimal harness for running ActivitySim's shipped example (`prototype_mtc`)
in a reproducible, agent-operable way. Every run gets an ID, a manifest, a
summary, a scorecard against targets (or a structured error), and a row in a
ledger. A small CLI (`asim`) and an MCP server (`asim mcp`) expose the same
operations. No LLM code lives in this repo.

`PLAN.md` is the spec, `NOTES.md` is the build log (read it first in a new
session), `CLAUDE.md` is the playbook for agents operating the harness.

## Setup

```bash
uv sync --locked           # Python 3.11 venv with ActivitySim 1.4.0 and the harness (editable)
source .venv/bin/activate
asim init                  # recreates example/data from the ActivitySim package (data is gitignored)
asim run --label "baseline full"
```

The example lives in `example/` (configs committed, data and output ignored)
and is never edited: every variation goes through a per-run override config
directory.

## Commands

```
asim run --label "smoke" --sample-size 500                          # households_sample_size
asim run --label "mc only" --resume-from <run_id> --resume-after trip_scheduling
asim run --label "x" --models initialize_landuse,initialize_households,...
asim run --label "y" --override-file my/trip_mode_choice_coefficients.csv   # shadows the example file
asim list                    # ledger, newest first
asim show <run_id>           # manifest (unique id prefixes are accepted everywhere)
asim summarize <run_id>      # metrics -> runs/<id>/summary.json
asim check <run_id>          # scorecard vs targets/prototype_mtc.yaml -> runs/<id>/scorecard.json
asim compare <run_a> <run_b> # per-metric deltas, b minus a
asim error <run_id>          # what failed: step, exception, expression context, traceback tail
asim targets bootstrap <run_id>   # write a run's share metrics as targets (refuses to overwrite)
asim reindex                 # rebuild runs/index.jsonl from manifests
asim mcp                     # serve the same operations as MCP tools over stdio
```

A run directory `runs/<run_id>/` holds `overrides/configs/` (the
`settings.yaml` with `inherit_settings: True` plus any shadowing files),
`output/` (ActivitySim's output, logs under `output/log/`), `manifest.json`,
`stdout.log`, `stderr.log`, and after the run `summary.json` and
`scorecard.json`, or `error.json` when it failed.

Timing on 4 cores: the full 5000-household example takes about 2 minutes, a
500-household sample about 1.5 minutes, a resume after `trip_scheduling`
(mode choice and the writers only) about 15 seconds.

## MCP server

`.mcp.json` registers `asim mcp` for Claude Code (project scope; approve it on
first use). Tools: `list_runs`, `run_model` (use `wait=false` and poll
`get_run` for anything longer than a couple of minutes), `get_run`,
`summarize_run`, `check_targets`, `compare_runs`, `get_error`,
`get_log_tail`, `read_config`, `list_configs`. Nothing writes under `example/`.

## Tests

```bash
pytest                      # unit tests, ~1 s, all in temp directories
pytest -m slow              # one integration test: a 500-household run, ~1.5 min
```

## Layout

```
src/asim_harness/   cli, runner, manifest, ledger, summarize, targets, compare, errors, mcp_server
example/            the packaged prototype_mtc example (configs committed; data, output ignored)
runs/               one directory per run (ignored) + index.jsonl
targets/            prototype_mtc.yaml, bootstrapped from the baseline run
tests/              unit tests and fixtures (output tables, failure overrides, real failure logs)
```

Environment variables `ASIM_HARNESS_ROOT`, `ASIM_EXAMPLE_DIR` and
`ASIM_RUNS_DIR` relocate the pieces without code changes.

---

## About the prototype_mtc example (upstream README)

The primary ActivitySim example model.

The `prototype_mtc` example is based on (but has evolved away from) the
[Bay Area Metro Travel Model One](https://github.com/BayAreaMetro/travel-model-one), 
also known as "TM1". TM1 has its roots in a wide array of analytical approaches, 
including discrete choice forms (multinomial and nested logit models), activity 
duration models, time-use models, models of individual micro-simulation with 
constraints, entropy-maximization models, etc. These tools are combined in the 
model design to realistically represent travel behavior, adequately replicate 
observed activity-travel patterns, and ensure model sensitivity to infrastructure
and policies. The model is implemented in a micro-simulation framework. Microsimulation
methods capture aggregate outcomes through the representation of the behavior of
individual decision-makers.

There are two model structures in the `prototype_mtc` example: a simpler model that is
relatively close to the TM1 model, and a more complex model that is incorporates
new model components that have been added by the ActivitySim consortium over the
past few years.

See https://activitysim.github.io for more information.

# Installation

The following short Python script will download and prepare the example data 
for the `prototype_mtc_extended` example model.

```python
from pathlib import Path
from activitysim.examples.external import download_external_example

example_dir = download_external_example(
  name="prototype_mtc_extended", 
  working_dir=Path.cwd(),
  url="https://github.com/ActivitySim/activitysim-prototype-mtc/archive/refs/heads/extended.tar.gz",
  assets={
    "data_full.tar.zst": {
      "url": "https://github.com/ActivitySim/activitysim-prototype-mtc/releases/download/v1.3.4/data_full.tar.zst",
      "sha256": "b402506a61055e2d38621416dd9a5c7e3cf7517c0a9ae5869f6d760c03284ef3",
      "unpack": "data_full",
    },
    "test/prototype_mtc_reference_pipeline.zip": {
      "url": "https://github.com/ActivitySim/activitysim-prototype-mtc/releases/download/v1.3.2/prototype_mtc_extended_reference_pipeline.zip",
      "sha256": "4d94b6a8a83225dda17e9ca19c9110bc1df2df5b4b362effa153d1c8d31524f5",
    }
  }
)
```

# Benchmarking

The `prototype_mtc` example model is run using the `activitysim` command line tool.
A quick and easy way to run the model for benchmarking is to use the following command:

```shell
cd activitysim-prototype-mtc-extended
activitysim workflow performance-benchmarking
```