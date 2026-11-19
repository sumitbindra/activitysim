"""
asim_core: a minimal, component-based core for ActivitySim.

The core knows how to hold data, run steps, save state, compare against
ActivitySim 1.x and measure runtime and memory. It knows nothing about travel
models: every model (auto ownership, mode choice, park-and-ride, ...) is a
component that plugs into it.
"""

__version__ = "0.1.0"
