"""
An experiment: formulas and utilities compiled with JAX (XLA), on the CPU.

Registers ``utilities: jax`` and ``interaction_utilities: jax`` when JAX is
installed (it is optional; nothing else in asim_core needs it). The default
implementations stay the reference: these are alternatives to measure.

How it works. A model's formulas are generated as plain NumPy code
(spec/translate.py), passed to ``utilities`` as ``FormulaValues``. Here the
same source is executed with ``jax.numpy`` in place of NumPy and compiled,
together with the utility sum, into one function (``jax.jit``): XLA fuses the
chain of element-wise operations and the sum into few passes over memory.

* **Skims stay in NumPy.** The skim values the formulas read are gathered
  first (through the step's skim lookups, timed as ``skim_lookup``) and passed
  in; copying the skims into JAX would cost gigabytes.
* **Array sizes.** XLA compiles once per input shape, so rows are padded to
  the next power of two (at least 1,024) and the result cut back.
* **Timetables stay in NumPy too.** Scheduling formulas ask the timetable
  about each row (``tt.previous_tour_ends(c['person_id'], c['start'])``);
  those calls are worked out first and passed in, like the skims.
* **Compiles are shared and can be kept.** Formulas with the same source
  share their compiled code across steps. With ``ASIM_JAX_CACHE_DIR`` set
  (or ``use_compilation_cache``), compiled code is also kept on disk and
  reused by later runs, as a model in production would.
* **Text columns become codes.** Formulas compare text columns with words
  (``c['tour_type'] == 'escort'``); the words are replaced by numbers and the
  column's values by the same numbers (0 for words the formulas don't name).
* **Checked against NumPy.** With ``ASIM_JAX_VERIFY=1``, the first call of
  each compiled formula set is also computed with NumPy and the largest
  difference kept (``report``).
* **Not everything compiles.** Formulas that use pandas-only functions (``_map``, ``_cut``, ``_shift``, ``_astype``) are left
  to the NumPy code, as are any that fail to compile; ``STATS`` counts calls
  each way and the time spent compiling.
* **Close, not identical.** XLA may add up and fuse multiplications and
  additions differently from NumPy, so results agree to rounding
  ("approximate"); 64-bit floats are turned on for that reason.
"""

from __future__ import annotations

import inspect
import os
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # optional
    jax = None

from .logit import interaction_utilities as _np_interaction_utilities
from .logit import utilities as _np_utilities
from .seams import register

MIN_ROWS = 1024
PAD_STEPS = 8  # sizes per doubling: at most 1/8 more rows than needed, 8 shapes to compile per doubling


def padded_rows(n: int) -> int:
    """Rows padded to one of a few sizes, so a compiled function is reused across batches."""
    if n <= MIN_ROWS:
        return MIN_ROWS
    # between 2^(b-1) and 2^b, sizes 1/PAD_STEPS of that range apart
    step = max(1, (1 << (n - 1).bit_length()) // (2 * PAD_STEPS))
    return -(-n // step) * step
STATS: dict[str, float] = defaultdict(float)  # calls, fallbacks, compiles, compile_s

_UNSUPPORTED = re.compile(r"\btt\.|\b_map\(|\b_cut\(|\b_shift\(|\b_astype\(|\br\[")
# a word compared with a text column: c['tour_type'] == 'escort' (also inside pandas expressions)
_WORD = re.compile(r"""(==|!=)\s*(\\?['"])([A-Za-z_][\w ]*)\2""")
VERIFY = bool(os.environ.get("ASIM_JAX_VERIFY"))
# a timetable question about each row, from plain columns: worked out in NumPy first
_TT_CALL = re.compile(r"""\btt\.(\w+)\(((?:\s*c\[(['"])[^'"]+\3\]\s*,?)+)\)""")
_COLUMN = re.compile(r"""\bc\[(['"])([^'"]+)\1\]""")
_SKIM = re.compile(r"""\bs\((['"])([^'"]+)\1,\s*(['"])([^'"]+)\3(?:,\s*(['"])([^'"]+)\5)?\)""")


def _jax_namespace():
    def clip(x, lower=None, upper=None):
        if lower is None and upper is None:
            return x
        return jnp.clip(x, lower, upper)

    def full(x, n):
        return jnp.broadcast_to(jnp.asarray(x), (n,)) if jnp.ndim(x) == 0 else x

    def fillna(x, value):
        x = jnp.asarray(x)
        return jnp.where(jnp.isnan(x), value, x) if jnp.issubdtype(x.dtype, jnp.floating) else x

    def pandas_eval(expr, names):
        return eval(expr, {"__builtins__": {}}, dict(names))  # noqa: S307 (generated formulas)

    def as_int8(x):
        x = jnp.asarray(x)
        if jnp.issubdtype(x.dtype, jnp.floating):
            x = jnp.where(jnp.isnan(x), 0, x)
        return x.astype(jnp.int8)

    def lognormal(draws, mu, sigma, scale=False):
        if scale:
            x = 1 + ((sigma * sigma) / (mu * mu))
            mu = jnp.log(mu / (jnp.sqrt(x)))
            sigma = jnp.sqrt(jnp.log(x))
        return jnp.exp(draws * sigma + mu)

    return {"np": jnp, "_clip": clip, "_full": full, "_fillna": fillna, "_pandas_eval": pandas_eval,
            "_as_int8": as_int8, "_lognormal": lognormal}


class _Compiled:
    """One generated formula function, compiled for JAX (or marked unusable)."""

    def __init__(self, fn):
        self.ok = False
        self.name = getattr(fn, "_asim_name", getattr(fn, "__name__", "?"))
        self.reason = ""
        self.calls = {"jax": 0, "numpy": 0}
        self.max_diff = None  # against NumPy, first call (ASIM_JAX_VERIFY)
        self.compiles, self.compile_s = 0, 0.0
        source = getattr(fn, "_asim_source", None)
        if jax is None or source is None:
            self.reason = "no generated source"
            return
        # timetable calls -> values passed in (tt becomes a dict of them)
        self.tt_calls = {}

        def to_lookup(m):
            cols = [x.group(2) for x in _COLUMN.finditer(m.group(2))]
            key = "|".join([m.group(1)] + cols)
            self.tt_calls[key] = (m.group(1), cols)
            return f"tt[{key!r}]"
        source = "\n".join(line if line.lstrip().startswith("#") else _TT_CALL.sub(to_lookup, line)
                           for line in source.splitlines())
        # words compared with text columns -> numbers (the columns' values are coded the same way)
        words = sorted({m.group(3) for line in source.splitlines() if not line.lstrip().startswith("#")
                        for m in _WORD.finditer(line)})
        self.word_codes = {w: i + 1 for i, w in enumerate(words)}
        source = "\n".join(line if line.lstrip().startswith("#") else
                           _WORD.sub(lambda m: f"{m.group(1)} {self.word_codes[m.group(3)]}", line)
                           for line in source.splitlines())
        code_lines = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
        m = _UNSUPPORTED.search(code_lines)
        if m:
            self.reason = f"uses {m.group(0).strip('(')}"
            return
        namespace = _jax_namespace()
        try:
            exec(compile(source, f"<jax {fn._asim_name}>", "exec"), namespace)
        except Exception as err:
            self.reason = f"doesn't load: {err}"
            return
        self.fn = namespace[fn._asim_name]
        self.params = list(inspect.signature(fn).parameters)
        self.columns = sorted({m.group(2) for m in _COLUMN.finditer(code_lines)})
        self.skims = sorted({(m.group(2), m.group(4), m.group(6) or "get") for m in _SKIM.finditer(code_lines)})
        self.executables: dict = {}
        self.ok = True

    def call(self, args, coefficients, kind, n):
        a = dict(zip(self.params, args))
        c, s = a.get("c"), a.get("s")
        cols = {}
        for name in self.columns:
            v = c[name]
            if hasattr(v, "dtype") and str(v.dtype) == "category":
                v = np.asarray(v, dtype=object)
            v = np.asarray(v)
            if v.dtype.kind in "OUS" and self.word_codes:  # text: coded as the formulas' words are
                v = pd.Series(v).map(self.word_codes).fillna(0).to_numpy(dtype=np.int32)
            if v.dtype.kind not in "biuf" or (v.ndim and len(v) != n):
                self.last_reason = f"column {name} is {v.dtype}"
                return None  # not one number per row
            cols[name] = v
        skims = {}
        if self.skims:
            if s is None:
                return None
            for acc, key, how in self.skims:
                skims[f"{acc}|{key}|{how}"] = np.asarray(
                    s(acc, key) if how == "get" else s(acc, key, how))
        if self.tt_calls:
            tt = a.get("tt")
            if tt is None:
                self.last_reason = "no timetable given"
                return None
            for key, (method, names) in self.tt_calls.items():
                skims[f"tt|{key}"] = np.asarray(getattr(tt, method)(*[np.asarray(c[x]) for x in names]))
        k = a.get("k")
        kvals = {name: float(v) for name, v in (k or {}).items()}
        rows = padded_rows(n)
        pad = lambda v: np.pad(v, (0, rows - len(v))) if v.ndim else v  # noqa: E731
        cols = {name: pad(v) for name, v in cols.items()}
        skims = {name: pad(v) for name, v in skims.items()}
        coefs = (np.asarray(coefficients, dtype=np.float64) if kind == "utilities"
                 else [float(x) for x in coefficients])
        key = (kind, rows, tuple((n_, v.dtype.str, v.shape) for n_, v in cols.items()),
               tuple((n_, v.dtype.str, v.shape) for n_, v in skims.items()), tuple(sorted(kvals)),
               np.shape(coefs) if kind == "utilities" else len(coefs))
        exe = self.executables.get(key)
        if exe is None:
            t0 = time.perf_counter()
            try:
                exe = jax.jit(self._utilities_fn(kind, rows)).lower(cols, skims, kvals, coefs).compile()
            except Exception as err:
                self.ok = False
                self.reason = f"doesn't compile: {type(err).__name__}: {str(err).splitlines()[0][:120]}"
                return None
            dt = time.perf_counter() - t0
            STATS["compile_s"] += dt
            STATS["compiles"] += 1
            self.compile_s += dt
            self.compiles += 1
            self.executables[key] = exe
        out = np.asarray(exe(cols, skims, kvals, coefs))
        return out[:n]

    def _utilities_fn(self, kind, rows):
        fn, params = self.fn, self.params

        def run(cols, skims, kvals, coefs):
            def s(acc, key, how="get"):
                return skims[f"{acc}|{key}|{how}"]
            tt = {key[3:]: v for key, v in skims.items() if key.startswith("tt|")}
            given = {"c": cols, "s": s, "k": kvals, "n": rows, "tt": tt}
            terms = list(fn(**{p: given.get(p) for p in params}))
            if kind == "utilities":
                # term by term into (rows x alternatives): no (terms x rows) matrix in between,
                # so the compiler can fuse each term's formula with its use
                total = jnp.zeros((rows, coefs.shape[1]), dtype=jnp.float64)
                for i, t in enumerate(terms):
                    t = jnp.broadcast_to(jnp.asarray(t, dtype=jnp.float64), (rows,))
                    total = total + t[:, None] * coefs[i][None, :]
                return total
            total = jnp.zeros(rows, dtype=jnp.float64)
            for t, cf in zip(terms, coefs):  # each product in the term's own type, as the reference
                total = total + jnp.broadcast_to(jnp.asarray(t * cf, dtype=jnp.float64), (rows,))
            return total
        return run


_COMPILED: dict = {}


def _compiled(fn):
    """Compiled once per formula source: steps using the same formulas share it."""
    key = getattr(fn, "_asim_source", None) or fn
    c = _COMPILED.get(key)
    if c is None:
        c = _COMPILED[key] = _Compiled(fn)
    return c


def use_compilation_cache(path: str | os.PathLike) -> None:
    """Keep compiled code on disk in ``path`` and reuse it in later runs."""
    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


def _evaluate(terms, coefficients, n, kind, fallback):
    STATS["calls"] += 1
    fn, args = getattr(terms, "fn", None), getattr(terms, "args", None)
    comp = _compiled(fn) if fn is not None else None
    if comp is not None and comp.ok:
        out = comp.call(args, coefficients, kind, n)
        if out is not None:
            comp.calls["jax"] += 1
            if VERIFY and comp.max_diff is None:
                ref = fallback(terms, coefficients, n)
                with np.errstate(invalid="ignore"):
                    diff = np.abs(np.asarray(out, dtype=np.float64) - ref)
                same_missing = np.isnan(out) == np.isnan(ref)
                comp.max_diff = float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0
                if not same_missing.all():
                    comp.max_diff = float("inf")
            return out
        if comp.ok and getattr(comp, "last_reason", ""):
            comp.reason = comp.last_reason
    if comp is not None:
        comp.calls["numpy"] += 1
    STATS["fallbacks"] += 1
    return fallback(terms, coefficients, n)


def report() -> list[dict]:
    """Per formula function: calls through JAX and NumPy, compiles, and why NumPy."""
    return [{"function": c.name, "jax_calls": c.calls["jax"], "numpy_calls": c.calls["numpy"],
             "compiles": c.compiles, "compile_s": round(c.compile_s, 3), "why_numpy": c.reason,
             "max_diff_vs_numpy": c.max_diff}
            for c in _COMPILED.values()]


if jax is not None:
    jax.config.update("jax_enable_x64", True)
    if os.environ.get("ASIM_JAX_CACHE_DIR"):
        use_compilation_cache(os.environ["ASIM_JAX_CACHE_DIR"])

    @register("utilities", "jax", agreement="approximate",
              description="formulas and the utility sum compiled together with JAX (XLA, CPU); "
                          "skims gathered in NumPy first; falls back to NumPy where it can't compile")
    def utilities_jax(terms, coefficients, n):
        return _evaluate(terms, coefficients, n, "utilities", _np_utilities)

    @register("interaction_utilities", "jax", agreement="approximate",
              description="formulas and the utility sum compiled together with JAX (XLA, CPU); "
                          "skims gathered in NumPy first; falls back to NumPy where it can't compile")
    def interaction_utilities_jax(terms, coefficients, n):
        return _evaluate(terms, coefficients, n, "interaction", _np_interaction_utilities)
