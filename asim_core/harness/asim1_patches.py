"""
Optional changes to ActivitySim 1.x, applied at run time by the harness
(``run_asim1.py --patch NAME``), so improvements to 1.x can be tested against
an unpatched 1.x run with the same comparison and profiling tools. Nothing
in ActivitySim's own source files is changed.

Patches:

``logsum_dedupe``
    Tour scheduling computes mode choice logsums for every tour and
    departure-duration option. Without a ``tour_departure_and_duration_segments``
    file, 1.x computes one logsum per (tour, departure period, return period,
    duration). The logsum only depends on duration if a mode choice formula
    reads it. When none does, this patch computes one logsum per (tour,
    departure period, return period) instead, and uses it for every
    duration. The logsums are the same numbers, computed fewer times.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _logsum_files_mention(state, model_settings, word: str) -> bool:
    """True if any file of the logsum (mode choice) model mentions ``word``. Conservative: plain text search."""
    import yaml

    logsum_yaml = model_settings.LOGSUM_SETTINGS
    paths = [state.filesystem.get_config_file_path(logsum_yaml)]
    settings = yaml.safe_load(open(paths[0])) or {}
    names = [settings.get("SPEC"), settings.get("COEFFICIENTS"), settings.get("COEFFICIENT_TEMPLATE")]
    for key in ("preprocessor", "nontour_preprocessor"):
        pre = settings.get(key) or {}
        if isinstance(pre, dict) and pre.get("SPEC"):
            names.append(pre["SPEC"])
    for name in filter(None, names):
        name = str(name)
        if not name.endswith(".csv"):
            name += ".csv"
        p = state.filesystem.get_config_file_path(name, mandatory=False)
        if p:
            paths.append(p)
    return any(word in open(p).read() for p in paths)


def logsum_dedupe() -> None:
    import activitysim.abm.models.util.vectorize_tour_scheduling as vts

    original_dedupe = vts.dedupe_alt_tdd
    original_logsums = vts.compute_tour_scheduling_logsums
    current = {}

    def compute_tour_scheduling_logsums(state, alt_tdd, tours_merged, tour_purpose, model_settings,
                                        trace_label, *, chunk_sizer):
        current["model_settings"] = model_settings  # dedupe_alt_tdd isn't given the settings
        return original_logsums(state, alt_tdd, tours_merged, tour_purpose, model_settings,
                                trace_label, chunk_sizer=chunk_sizer)

    def dedupe_alt_tdd(state, alt_tdd, tour_purpose, trace_label):
        settings = current.get("model_settings")
        if (
            state.get_injectable("tdd_alt_segments", None) is not None
            or settings is None
            or _logsum_files_mention(state, settings, "duration")
        ):
            return original_dedupe(state, alt_tdd, tour_purpose, trace_label)
        key = alt_tdd.index.name
        periods = (
            alt_tdd[["out_period", "in_period", "duration"]]
            .reset_index()
            .drop_duplicates(subset=[key, "out_period", "in_period"])
            .set_index(key)
        )
        logger.info(f"{trace_label}: logsum_dedupe: {len(alt_tdd)} options -> {len(periods)} logsums")
        return periods, ["out_period", "in_period"]

    vts.compute_tour_scheduling_logsums = compute_tour_scheduling_logsums
    vts.dedupe_alt_tdd = dedupe_alt_tdd


def no_numexpr() -> None:
    """
    pandas evaluates Series arithmetic with numexpr above 1M elements, with
    other number rules than NumPy (asim1 findings, #3). Off: NumPy at any size.

    Only for arithmetic: 1.x's formulas evaluated with ``pd.eval`` keep using
    numexpr, as they do in 1.x by default. (Turning numexpr off with pandas'
    ``compute.use_numexpr`` option, as this patch first did, also switches
    ``pd.eval`` to plain pandas, which changes integer arithmetic: 8-bit
    columns overflow, e.g. 7 * 22 = -102 (asim1 findings, #14).)
    """
    from pandas.core.computation import eval as pandas_eval
    from pandas.core.computation import expressions

    expressions.set_use_numexpr(False)  # arithmetic: NumPy at any size
    original = pandas_eval._check_engine
    if getattr(original, "_asim_core_patched", False):
        return

    def check_engine(engine):  # pd.eval: numexpr unless asked otherwise, as by default
        return original("numexpr" if engine is None else engine)

    check_engine._asim_core_patched = True
    pandas_eval._check_engine = check_engine


PATCHES = {"logsum_dedupe": logsum_dedupe, "no_numexpr": no_numexpr}


def apply(names) -> None:
    for name in names or []:
        if name not in PATCHES:
            raise KeyError(f"unknown patch '{name}'; available: {sorted(PATCHES)}")
        PATCHES[name]()
