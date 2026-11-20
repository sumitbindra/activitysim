# asim_core additions to the 1.x example configs

Files here are read before the example's own configs folder (listed first
in `configs_dirs`), the way 1.x layers configs folders. The 1.x examples
themselves are not changed.

- `<example>/input_tables.yaml`: the region's input columns (meaning, unit,
  rules), on top of asim_core's standard tables
  (`src/asim_core/inputs/standard_tables.yaml`). See `docs/input_checks.md`.

Check an example's inputs:

    python -m asim_core.inputs -c asim_core/configs/prototype_mtc \
        -c activitysim/examples/prototype_mtc/configs -d activitysim/examples/prototype_mtc/data
