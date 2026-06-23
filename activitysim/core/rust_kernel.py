# ActivitySim
# See full license in LICENSE.txt.
"""
Optional Rust kernel loader and dispatch helper.

The Rust kernel (``activitysim-kernel`` crate, importable as the
``activitysim_kernel`` module) is a *surgical* acceleration of the
destination/location choice hot path. It is strictly optional: if the compiled
extension is not installed, everything here degrades gracefully and ActivitySim
runs on its existing pure-Python code path.

Nothing in ActivitySim should import ``activitysim_kernel`` directly. Go through
this module so there is a single, tested place that handles availability and
fallback. See ``docs/rust_kernel_recon.md`` for the design.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import activitysim_kernel as _kernel  # type: ignore

    _IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover - exercised only when extension absent
    _kernel = None
    _IMPORT_ERROR = e


def is_available() -> bool:
    """Return True if the compiled Rust kernel extension is importable."""
    return _kernel is not None


def kernel():
    """Return the imported ``activitysim_kernel`` module.

    Raises
    ------
    RuntimeError
        If the extension is not available, with the original import error
        chained for diagnostics.
    """
    if _kernel is None:
        raise RuntimeError(
            "activitysim_kernel (Rust extension) is not available; "
            "build it with `maturin develop` in activitysim-kernel/ to enable "
            "the Rust choice kernel."
        ) from _IMPORT_ERROR
    return _kernel


def version() -> str | None:
    """Return the Rust kernel version string, or None if unavailable."""
    if _kernel is None:
        return None
    return getattr(_kernel, "__version__", None)


def use_rust_kernel(state) -> bool:
    """Decide whether the Rust kernel should be used for this run.

    Combines the ``use_rust_kernel`` settings flag (default off until parity and
    performance are proven, per the rollout plan) with actual availability of the
    compiled extension. Spec-level fallback (unsupported IR ops) is handled at the
    dispatch site, not here.
    """
    flag = bool(getattr(state.settings, "use_rust_kernel", False))
    if flag and not is_available():
        logger.warning(
            "use_rust_kernel is set but the activitysim_kernel extension is not "
            "installed; falling back to the Python choice kernel."
        )
        return False
    return flag
