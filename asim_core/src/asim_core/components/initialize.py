"""
The first two steps of a 1.x model: reading the land use, households and
persons inputs, and annotating them.

``initialize_landuse`` (1.x: ``abm/models/initialize.py``, ``abm/tables/landuse.py``,
``abm/tables/accessibility.py``):

1. read the land use table as 1.x's ``read_input_table`` does (renames, kept
   columns, recodes; asim_core's reader, tested to give 1.x's tables), sorted
   by zone if it isn't;
2. annotate it (``initialize_landuse.yaml``'s ``annotate_tables``);
3. make the empty ``accessibility`` table (its index is the land use's).

``initialize_households`` (``abm/tables/households.py``, ``persons.py``,
``time_windows.py``, ``shadow_pricing.add_size_tables``):

1. read households and persons; sample households as 1.x does
   (``households_sample_size``: NumPy's RandomState seeded with the base seed
   and ``hash32('sample_households')``, ``choice`` without replacement, in the
   order drawn), add ``sample_rate``, keep the persons of the sampled
   households (in file order) and check that every person has a household
   and every household a person;
2. ``households_skipped``: no rows, the households' columns;
3. annotate, in the order of ``initialize_households.yaml`` (1.x: persons,
   households, persons again);
4. ``person_windows``: every person's time windows, all free (one column per
   hour of the departure/duration alternatives, plus one at each end);
5. the size tables of the location models (``<selector>_destination_size``,
   from ``shadow_pricing.yaml``): each zone's size per segment, scaled to the
   sample only with shadow pricing on a full run, as 1.x.

Numbers in CSV files are read as 1.x reads them (pandas' parser) by
default, for results identical to 1.x's; ``csv_reader='exact'`` reads them
correctly rounded instead (1.x findings, #11).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..component import Component, StepContext
from ..inputs.reader import InputTable, input_tables_from_settings, read_table, table_columns, to_pandas
from ..random import hash32
from ..spec.legacy import find_config_file, load_assignment_spec, load_global_constants, load_top_settings, read_yaml
from ..store import Table
from . import annotate
from .annotate import Annotation


class InitializeError(ValueError):
    pass


CSV_READERS = {"asim1": True, "exact": False}  # -> read_table's like_asim1


def annotations_from_settings(configs_dirs, settings_file: str, step: str) -> list[Annotation]:
    """``annotate_tables`` of an initialize settings file, in its order."""
    settings = read_yaml(find_config_file(configs_dirs, settings_file))
    top = load_top_settings(configs_dirs)
    constants = dict(load_global_constants(configs_dirs))
    out = []
    for i, item in enumerate(settings.get("annotate_tables") or []):
        if item.get("column_map"):
            raise NotImplementedError(f"{settings_file}: column_map (renaming in annotate_tables)")
        table, a = item["tablename"], item.get("annotate")
        if not a:
            continue
        spec_file = str(a["SPEC"])
        spec_file = spec_file if spec_file.endswith(".csv") else spec_file + ".csv"
        tables = list(a.get("TABLES") or [])
        spec = load_assignment_spec(configs_dirs, spec_file, constants, skim_accessors=[annotate.SKIM_ACCESSOR],
                                    name=f"{step}_{i}_{table}", tables=tables, settings=top,
                                    df_names=[a.get("DF", table)])
        out.append(Annotation(table, spec, tables))
    return out


def _spec(specs: Sequence[InputTable], name: str) -> InputTable:
    for s in specs:
        if s.tablename == name:
            return s
    raise InitializeError(f"settings.yaml's input_table_list has no '{name}' table")


def _read(specs, names: set[str], data_dirs, csv_reader: str, recode: bool) -> dict[str, pd.DataFrame]:
    """Input tables as 1.x's read_input_table gives them (recodes applied, in table order).
    A table recoded from another (households' zones from land use) reads that one too."""
    wanted = [s for s in specs if s.tablename in names]
    if recode:
        for s in list(wanted):
            for how in (s.recode_columns or {}).values():
                parent = how.split(".")[0] if how != "zero-based" else None
                if parent and parent not in {w.tablename for w in wanted}:
                    wanted.insert(0, _spec(specs, parent))
    like_asim1 = CSV_READERS[csv_reader]
    tables = {s.tablename: read_table(s, data_dirs, like_asim1)[0] for s in wanted}
    return to_pandas(tables, wanted, recode=recode)


def _targets(annotations: Sequence[Annotation], table: str) -> list[str]:
    out: list[str] = []
    for a in annotations:
        if a.table == table:
            out += [t for t in a.spec.targets if t not in out]
    return out


def _columns(base: list[str], annotations, table: str) -> list[str]:
    """A table's columns after annotating: its own, then new annotation targets in order."""
    return list(base) + [c for c in _targets(annotations, table) if c not in base]


def _final(df: pd.DataFrame, name: str, made: dict) -> Table:
    """The table with its annotations: replaced columns in place, new ones appended in order."""
    out = df.copy()
    for col, values in made.items():
        out[col] = values
    return Table.from_pandas(name, out)


def _bind(annotations, own: dict[str, list[str]], available):
    """Bind annotations of tables the step makes (``own``: their columns): what they read
    of other tables, and the bound annotations."""
    bound, needs, _ = annotate.bind(annotations, {**available, **{t: set(c) for t, c in own.items()}}, {})
    return bound, {t: sorted(c) for t, c in needs.items() if t not in own}


def initialize_landuse(configs_dirs: Sequence[str | Path], data_dirs: Sequence[str | Path],
                       settings_file: str = "initialize_landuse.yaml", csv_reader: str = "asim1") -> Component:
    configs_dirs = [Path(d) for d in configs_dirs]
    data_dirs = [Path(d) for d in data_dirs]
    step = "initialize_landuse"
    top = load_top_settings(configs_dirs)
    recode = bool(top.get("recode_pipeline_columns", False))
    specs = input_tables_from_settings(configs_dirs)
    columns = table_columns(_spec(specs, "land_use"), data_dirs, recode=recode)
    annotations = annotations_from_settings(configs_dirs, settings_file, step)
    if any(a.table != "land_use" for a in annotations):
        raise NotImplementedError(f"{settings_file}: annotating tables other than land_use")
    bound, needs = _bind(annotations, {"land_use": columns}, {})
    if needs:
        raise NotImplementedError(f"{settings_file}: annotations read other tables {sorted(needs)}")

    def run(ctx: StepContext):
        lu = _read(specs, {"land_use"}, data_dirs, csv_reader, recode)["land_use"]
        if not lu.index.is_monotonic_increasing:  # 1.x sorts land use by zone
            lu = lu.sort_index()
        ctx.create_table(Table.from_pandas("land_use", lu))
        made: dict = {}
        annotate.run(ctx, bound, made)
        return {"land_use": _final(lu, "land_use", made.get("land_use", {})),
                "accessibility": Table("accessibility", lu.index.to_numpy(), lu.index.name or "zone_id")}

    return Component(
        step, run, needs={}, provides={"land_use": _columns(columns, annotations, "land_use"), "accessibility": []},
        description="read and annotate the land use table; make the empty accessibility table",
        info={"annotations": annotations, "generated_source": "\n".join(b.bound.source for b in bound)},
    )


def initialize_households(configs_dirs: Sequence[str | Path], data_dirs: Sequence[str | Path],
                          settings_file: str = "initialize_households.yaml",
                          households_sample_size: int | None = None, trace_hh_id: int | None = None,
                          csv_reader: str = "asim1") -> Component:
    """
    ``households_sample_size`` and ``trace_hh_id`` default to settings.yaml's
    (1.x's command line can override the sample size: pass the same here).
    """
    configs_dirs = [Path(d) for d in configs_dirs]
    data_dirs = [Path(d) for d in data_dirs]
    step = "initialize_households"
    top = load_top_settings(configs_dirs)
    if top.get("override_hh_ids"):
        raise NotImplementedError("settings.yaml: override_hh_ids")
    sample_size = int(top.get("households_sample_size") or 0) if households_sample_size is None \
        else int(households_sample_size)
    trace = top.get("trace_hh_id") if trace_hh_id is None else trace_hh_id
    base_seed = int(top.get("rng_base_seed") or 0)
    recode = bool(top.get("recode_pipeline_columns", False))
    specs = input_tables_from_settings(configs_dirs)
    hh_cols = table_columns(_spec(specs, "households"), data_dirs, recode=recode)
    if "sample_rate" not in hh_cols:
        hh_cols = hh_cols + ["sample_rate"]
    per_cols = table_columns(_spec(specs, "persons"), data_dirs, recode=recode)
    own = {"households": hh_cols, "persons": per_cols}
    annotations = annotations_from_settings(configs_dirs, settings_file, step)
    if any(a.table not in own for a in annotations):
        raise NotImplementedError(f"{settings_file}: annotating tables other than households and persons")
    windows = window_columns(configs_dirs)
    size_tables = size_table_settings(configs_dirs)
    # 1.x scales size tables to the choosers only with shadow pricing on a full run
    scaled = bool(top.get("use_shadow_pricing")) and sample_size == 0 and any(st["scale"] for st in size_tables)
    if scaled:
        persons_after = set(_columns(per_cols, annotations, "persons"))
        for st in size_tables:
            wanted = {st["segment_col"], st["filter_col"]} - {None}
            if st["chooser_table"] not in ("persons", "persons_merged") or not wanted <= persons_after:
                raise NotImplementedError(f"{st['name']}: scaling to choosers in '{st['chooser_table']}' "
                                          f"by {sorted(wanted)} (asim_core reads them from persons)")
    bound_now: dict = {}

    def resolve_needs(available):
        bound, needs = _bind(annotations, own, available)
        for st in size_tables:
            needs.setdefault("land_use", [])
            needs["land_use"] = sorted(set(needs["land_use"]) | set(st["columns"]))
        bound_now["bound"] = bound
        return needs

    def run(ctx: StepContext):
        dfs = _read(specs, {"households", "persons"}, data_dirs, csv_reader, recode)
        hh, persons = sample_households(dfs["households"], dfs["persons"], sample_size, trace, base_seed)
        ctx.create_table(Table.from_pandas("households", hh))
        ctx.create_table(Table.from_pandas("persons", persons))
        made: dict = {}
        annotate.run(ctx, bound_now["bound"], made)
        out = {
            "households": _final(hh, "households", made.get("households", {})),
            "persons": _final(persons, "persons", made.get("persons", {})),
            "households_skipped": Table.from_pandas("households_skipped", hh.iloc[0:0]),
            "person_windows": Table("person_windows", persons.index.to_numpy(), persons.index.name or "person_id",
                                    {w: np.zeros(len(persons), dtype=np.int8) for w in windows}),
        }
        for st in size_tables:
            out[st["name"]] = size_table(ctx, st, out["persons"] if scaled else None, sample_size, top)
        return out

    provides = {"households": _columns(hh_cols, annotations, "households"),
                "persons": _columns(per_cols, annotations, "persons"),
                "households_skipped": hh_cols, "person_windows": windows}
    for st in size_tables:
        provides[st["name"]] = list(st["segments"])
    return Component(
        step, run, provides=provides, resolve_needs=resolve_needs,
        description="read, sample and annotate households and persons; time windows; size tables",
        info={"annotations": annotations, "households_sample_size": sample_size},
    )


def sample_households(hh_full: pd.DataFrame, persons_full: pd.DataFrame, sample_size: int, trace=None,
                      base_seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1.x's households and persons tables: the sample, sample_rate, the sampled households' persons."""
    total = len(hh_full)
    sliced = True
    if trace and sample_size == 1:
        hh = hh_full[hh_full.index.isin([trace])]
    elif total > sample_size > 0:
        # 1.x's external rng for the pseudo-step 'sample_households'
        prng = np.random.RandomState([base_seed, hash32("sample_households")])
        hh = hh_full.take(prng.choice(total, size=sample_size, replace=False))
        if trace and trace not in hh.index and trace in hh_full.index:
            hh = pd.concat([hh_full.loc[[trace]], hh[1:]])  # 1.x: replaces the first one
    else:
        hh, sliced = hh_full, False
    if "sample_rate" not in hh.columns:
        hh = hh.copy()
        hh["sample_rate"] = 1 if sample_size == 0 else sample_size / total
    persons = persons_full[persons_full.household_id.isin(hh.index)] if sliced else persons_full
    if hh.index.duplicated().any() or persons.index.duplicated().any():
        raise InitializeError("household or person IDs used more than once")
    without_hh = ~persons.household_id.isin(hh.index)
    if without_hh.any():
        raise InitializeError(f"{int(without_hh.sum())} persons with bad household_id")
    without_persons = persons.groupby("household_id").size().reindex(hh.index).isnull()
    if without_persons.any():
        raise InitializeError(f"{int(without_persons.sum())} households with no persons")
    return hh, persons


def window_columns(configs_dirs) -> list[str]:
    """1.x's create_timetable_windows: one column per hour of the alternatives, plus one at each end."""
    alts = pd.read_csv(find_config_file(configs_dirs, "tour_departure_and_duration_alternatives.csv"))
    return [str(w) for w in range(int(alts.start.min()) - 1, int(alts.end.max()) + 2)]


def size_table_settings(configs_dirs) -> list[dict]:
    """The location models whose size tables 1.x makes (shadow_pricing.yaml's shadow_pricing_models)."""
    try:
        sp = read_yaml(find_config_file(configs_dirs, "shadow_pricing.yaml"))
    except FileNotFoundError:
        return []
    models = sp.get("shadow_pricing_models")
    if not models:
        return []
    terms = pd.read_csv(find_config_file(configs_dirs, "destination_choice_size_terms.csv"), comment="#",
                        index_col="segment")
    out = []
    for selector, model_name in models.items():
        settings = read_yaml(find_config_file(configs_dirs, f"{model_name}.yaml"))
        if settings.get("MODEL_SELECTOR") != selector:
            raise InitializeError(f"shadow_pricing.yaml: {model_name}'s MODEL_SELECTOR is not '{selector}'")
        st = terms[terms.model_selector == selector].drop(columns="model_selector")
        segment_ids = settings.get("SEGMENT_IDS") or {}
        if set(st.index) != set(segment_ids):
            raise InitializeError(f"{model_name}: size term segments {sorted(st.index)} differ from SEGMENT_IDS "
                                  f"{sorted(segment_ids)}")
        out.append({"name": f"{selector}_destination_size", "selector": selector, "terms": st,
                    "segments": list(st.index), "columns": list(st.columns), "segment_ids": segment_ids,
                    "chooser_table": settings.get("CHOOSER_TABLE_NAME"),
                    "segment_col": settings.get("CHOOSER_SEGMENT_COLUMN_NAME"),
                    "filter_col": settings.get("CHOOSER_FILTER_COLUMN_NAME"),
                    "scale": bool(sp.get("SCALE_SIZE_TABLE", False))})
    return out


def size_table(ctx: StepContext, st: dict, choosers: Table, sample_size: int, top) -> Table:
    """Each zone's size per segment (1.x's tour_destination_size_terms), land use sorted by zone;
    scaled to the choosers only with shadow pricing on a full run (1.x's add_size_tables)."""
    zones = ctx.ids("land_use")
    order = np.argsort(zones, kind="stable")
    land_use = pd.DataFrame({c: np.asarray(ctx.column("land_use", c))[order] for c in st["columns"]},
                            index=pd.Index(zones[order], name="zone_id"))
    sizes = pd.DataFrame({seg: land_use[row.index].dot(row) for seg, row in st["terms"].iterrows()},
                         index=land_use.index)
    if bool(top.get("use_shadow_pricing")) and sample_size == 0 and st["scale"]:
        segment = np.asarray(choosers[st["segment_col"]])
        if st["filter_col"]:
            segment = segment[np.asarray(choosers[st["filter_col"]]) != 0]
        factors = {}
        for c in sizes:
            desired = sizes[c].astype(np.float64).sum()
            factor = (segment == st["segment_ids"][c]).sum() / np.maximum(desired, 1)
            factors[c] = factor if factor > 0 else 1
        sizes = sizes * factors
    return Table.from_pandas(st["name"], sizes)
