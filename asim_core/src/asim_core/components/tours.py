"""
Making tours, and their IDs, as 1.x does (``abm/models/util/tour_frequency.py``
and ``canonical_ids.py``).

**Tour IDs.** 1.x numbers every tour a person could make: it lists all
possible tours of every tour model ("flavors": ``work1``, ``work2``,
``school1``, ``escort1`` ... ``j_shopping2``, ``eat1_1`` ...) from the
models' settings and alternatives files, sorts the list, and gives a tour
``person_id * len(list) + its place in the list``. So a tour's ID is stable
(the same person's same tour gets the same ID whatever else happens), and
it names the tour's random number stream. ``canonical_tour_channels``
builds the same list from the same files, with 1.x's rules and defaults.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..spec.legacy import find_config_file, read_yaml

TOUR_CATEGORIES = ["mandatory", "joint", "non_mandatory", "atwork"]


def _settings(configs_dirs, name) -> dict:
    try:
        return read_yaml(find_config_file(configs_dirs, name)) or {}
    except FileNotFoundError:
        return {}


def _csv(configs_dirs, name) -> pd.DataFrame:
    try:
        return pd.read_csv(find_config_file(configs_dirs, name), comment="#")
    except FileNotFoundError:
        return pd.DataFrame()


def _numbers_in(columns, flavor: str) -> int:
    """1.x's parse_tour_flavor_from_columns: the largest number in the column names containing ``flavor``."""
    found = [int(x) for col in columns if flavor in col for x in re.findall(r"\d+", col)]
    return max(found) if found else -1


def _flavors_from_alts(alts: pd.DataFrame, provided, default, max_extension=0) -> dict:
    """1.x's determine_flavors_from_alts_file."""
    try:
        flavors = {c: int(alts[c].max() + max_extension) for c in alts.columns
                   if all(alts[c].astype(str).str.isnumeric()) and c != "tot_tours"}
        valid = all(isinstance(f, str) and n >= 0 for f, n in flavors.items()) and len(flavors) > 0
    except (ValueError, AttributeError):
        valid = False
    if provided is not None:
        return provided
    return flavors if valid else default


def _enumerate(flavors: Mapping[str, int]) -> list[str]:
    return [f"{t}{n}" for t, most in flavors.items() for n in range(1, most + 1)]


def canonical_tour_channels(configs_dirs: Sequence[str | Path], models: Sequence[str] | None = None) -> list[str]:
    """1.x's canonical_tours: every possible tour's label, sorted. ``models``: the run's
    step list (settings.yaml's ``models``), which decides whether school escort tours count."""
    configs_dirs = [Path(d) for d in configs_dirs]
    # non-mandatory
    nm_settings = _settings(configs_dirs, "non_mandatory_tour_frequency.yaml")
    nm_alts = _csv(configs_dirs, "non_mandatory_tour_frequency_alternatives.csv")
    ext = _csv(configs_dirs, "non_mandatory_tour_frequency_extension_probs.csv")
    provided_ext = nm_settings.get("MAX_EXTENSION")
    found_ext = _numbers_in(ext.columns, "tour")
    max_ext = provided_ext if provided_ext is not None else (found_ext if found_ext >= 0 else 2)
    nm_default = {"escort": 2 + max_ext, "shopping": 1 + max_ext, "othmaint": 1 + max_ext,
                  "othdiscr": 1 + max_ext, "eatout": 1 + max_ext, "social": 1 + max_ext}
    nm = _enumerate(_flavors_from_alts(nm_alts, nm_settings.get("NON_MANDATORY_TOUR_FLAVORS"), nm_default, max_ext))
    # mandatory
    mtf_settings = _settings(configs_dirs, "mandatory_tour_frequency.yaml")
    mtf_spec = _csv(configs_dirs, str(mtf_settings.get("SPEC", "mandatory_tour_frequency.csv")))
    parsed = {"work": _numbers_in(mtf_spec.columns, "work"), "school": _numbers_in(mtf_spec.columns, "school")}
    provided = mtf_settings.get("MANDATORY_TOUR_FLAVORS")
    if provided is not None:
        mandatory_flavors = provided
    elif parsed["work"] >= 1 and parsed["school"] >= 1:
        mandatory_flavors = parsed
    else:
        mandatory_flavors = {"work": 2, "school": 2}
    mandatory = _enumerate(mandatory_flavors)
    # at-work subtours, per parent work tour (eat1_1: eat subtour of work tour 1)
    aw_settings = _settings(configs_dirs, "atwork_subtour_frequency.yaml")
    aw_alts = _csv(configs_dirs, "atwork_subtour_frequency_alternatives.csv")
    atwork = _enumerate(_flavors_from_alts(aw_alts, aw_settings.get("ATWORK_SUBTOUR_FLAVORS"),
                                           {"eat": 1, "business": 2, "maint": 1}))
    atwork = [f"{c}_{i + 1}" for c in atwork for i in range(mandatory_flavors["work"])]
    # joint
    jt_settings = _settings(configs_dirs, "joint_tour_frequency.yaml")
    jt_alts = _csv(configs_dirs, "joint_tour_frequency_alternatives.csv")
    joint = _enumerate(_flavors_from_alts(jt_alts, jt_settings.get("JOINT_TOUR_FLAVORS"),
                                          {"shopping": 2, "othmaint": 2, "othdiscr": 2, "eatout": 2, "social": 2}))
    channels = nm + mandatory + atwork + [f"j_{c}" for c in joint]
    if models is not None and "school_escorting" in models:
        n = _settings(configs_dirs, "school_escorting.yaml").get("NUM_ESCORTEES", 3)
        channels += [f"se_{c}" for c in _enumerate({"escort": 2 * n})]
    return sorted(channels)


def create_tours(tour_counts: pd.DataFrame, tour_category: str, parent_col: str = "person_id") -> pd.DataFrame:
    """1.x's create_tours: one row per tour from counts per parent (rows) and tour type (columns),
    numbered within type and within parent."""
    tours = tour_counts.stack().reset_index()
    tours.columns = [parent_col, "tour_type", "tour_type_count"]
    tours = tours.take(np.repeat(tours.index.values, tours.tour_type_count.values))
    grouped = tours.groupby([parent_col, "tour_type"])
    tours["tour_type_num"] = grouped.cumcount() + 1
    tours["tour_type_count"] = tours["tour_type_num"] + grouped.cumcount(ascending=False)
    grouped = tours.groupby(parent_col)
    tours["tour_num"] = grouped.cumcount() + 1
    tours["tour_count"] = tours["tour_num"] + grouped.cumcount(ascending=False)
    tours["tour_category"] = pd.Series(tour_category, index=tours.index).astype(
        pd.CategoricalDtype(TOUR_CATEGORIES, ordered=False))
    tours["number_of_participants"] = 1
    tours.reset_index(drop=True, inplace=True)
    for c in ("tour_count", "tour_num", "tour_type_num", "tour_type_count", "number_of_participants"):
        tours[c] = tours[c].astype("int8")
    return tours


def set_tour_index(tours: pd.DataFrame, channels: Sequence[str], parent_tour_num_col: str | None = None,
                   is_joint: bool = False, is_school_escorting: bool = False) -> pd.DataFrame:
    """1.x's set_tour_index: ``person_id * len(channels) + place of '<tour_type><tour_type_num>'``."""
    label = tours.tour_type.astype(str) + tours["tour_type_num"].map(str)
    if parent_tour_num_col:
        label = label + "_" + tours[parent_tour_num_col].astype(np.int64).map(str)
    if is_joint:
        label = "j_" + label
    if is_school_escorting:
        label = "se_" + label
    place = label.map({c: i for i, c in enumerate(channels)})
    if place.isnull().any():
        raise ValueError(f"tours not among the possible tours (check the tour models' alternatives files): "
                         f"{sorted(label[place.isnull()].unique())[:10]}")
    tours["tour_id"] = tours.person_id.astype(np.int64) * len(channels) + place.astype(np.int64)
    if tours.tour_id.duplicated().any():
        raise ValueError("tour IDs are not unique")
    tours.set_index("tour_id", inplace=True, verify_integrity=True)
    return tours


def extend_table(table: pd.DataFrame | None, rows: pd.DataFrame) -> pd.DataFrame:
    """
    1.x's ``state.extend_table`` (adding rows): categories of columns in both
    are united (sorted), the new rows go after the old ones, columns only in
    the old table become missing for the new rows (so integers become
    floats), and text columns, or categories that include '', missing in the
    new rows get ''.
    """
    if table is None:
        return rows
    from pandas.api.types import union_categoricals

    if len(table.index.intersection(rows.index)):
        raise ValueError("new rows reuse existing IDs")
    table = table.copy()
    rows = rows.copy()
    blank = [c for c in table.columns if c not in rows.columns and table[c].dtype == "O"]
    for c in table.columns:
        if c in rows.columns:
            if isinstance(table[c].dtype, pd.CategoricalDtype) and isinstance(rows[c].dtype, pd.CategoricalDtype):
                uc = union_categoricals([table[c], rows[c]], sort_categories=True)
                table[c] = pd.Categorical(table[c], categories=uc.categories)
                rows[c] = pd.Categorical(rows[c], categories=uc.categories)
        elif isinstance(table[c].dtype, pd.CategoricalDtype) and "" in table[c].cat.categories:
            blank.append(c)
    out = pd.concat([table, rows], sort=False, axis=0)
    for c in blank:
        out[c] = out[c].fillna("")
    return out


def assign_in_place(df: pd.DataFrame, df2: pd.DataFrame) -> None:
    """
    1.x's util.assign_in_place (without its optional downcasting): update
    ``df``'s rows from ``df2`` (categories united), then put back an integer
    type that pandas' update turned into floats; new columns are added.
    """
    from pandas.api.types import union_categoricals

    if len(df2.index.difference(df.index)):
        raise ValueError("rows to update that the table doesn't have")
    common = df2.columns.intersection(df.columns)
    if len(common):
        old_dtypes = [df[c].dtype for c in common]
        for c in common:
            if isinstance(df[c].dtype, pd.CategoricalDtype):
                if not isinstance(df2[c].dtype, pd.CategoricalDtype):
                    df2[c] = df2[c].astype("category")
                uc = union_categoricals([df[c], df2[c]], sort_categories=True)
                df[c] = pd.Categorical(df[c], categories=uc.categories)
                df2[c] = pd.Categorical(df2[c], categories=uc.categories)
        df.update(df2)
        for c, old in zip(common, old_dtypes):
            if old == df2[c].dtype and df[c].dtype != old:
                df[c] = df[c].astype(old)
            if isinstance(old, pd.CategoricalDtype):
                continue
            if (np.issubdtype(old, np.integer) and np.issubdtype(df2[c].dtype, np.integer)
                    and not np.issubdtype(df[c].dtype, np.integer)):
                df[c] = df[c].astype(old)
    new = [c for c in df2.columns if c not in df.columns]
    df[new] = df2[new]
    for c in new:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].astype("category")
