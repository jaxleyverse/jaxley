# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
"""Performance benchmarks for a few common models. Local only, not run in CI.

Run with `pytest -m perf` (without `-n`), which prints a table of seconds per scenario:

- build: constructing the model (e.g. reading the SWC file, inserting channels). JAX's
  one-time start-up costs are excluded by `_warm_up()`; without it, the first
  `read_swc` in a process takes ~10 s longer.
- compile: tracing and compiling `jit(simulate)`.
- simulate: one run of the compiled simulation (fastest of 3).
- grad compile, grad: the same for `jit(grad(loss))`.

To compare against another git ref, run `python tests/perf_regression.py --base main`.
"""

import json
import os
import sys
import time
from functools import cache

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import jaxley as jx
from jaxley.channels import HH
from jaxley.synapses import IonotropicSynapse

CA1 = os.path.join(os.path.dirname(__file__), "swc_files", "morph_ca1_n120.swc")
COLUMNS = ["build", "compile", "simulate", "grad compile", "grad"]
RESULTS = {}  # {scenario: {column: seconds}}, printed by `conftest.py`.


def _make_simulate(module, t_max):
    """Stimulate and record the first compartment, make the radius trainable."""
    current = jx.step_current(1.0, 10.0, 0.2, 0.025, t_max)
    module.select(nodes=0).stimulate(current, verbose=False)
    module.select(nodes=0).record("v", verbose=False)
    module.make_trainable("radius", verbose=False)

    def simulate(params):
        return jx.integrate(module, params=params, t_max=t_max)

    return simulate, module.get_parameters()


def _ca1_cell():
    cell = jx.read_swc(CA1, ncomp=4)
    cell.insert(HH())
    return cell


def _small_cell():
    """A cell with 7 branches of 2 compartments each."""
    branch = jx.Branch(jx.Compartment(), 2)
    return jx.Cell(branch, parents=[-1, 0, 0, 1, 1, 2, 2])


# Scenarios: each builds a model and returns `(simulate, params)`.


def comp():
    """Single compartment with HH, 50 ms."""
    comp = jx.Compartment()
    comp.insert(HH())
    return _make_simulate(comp, t_max=50.0)


def cell():
    """CA1 cell from SWC (4 compartments per branch) with HH, 20 ms."""
    return _make_simulate(_ca1_cell(), t_max=20.0)


def cell_vmap():
    """The CA1 cell, vmapped over 8 parameter sets."""
    simulate, params = _make_simulate(_ca1_cell(), t_max=20.0)
    scale = jnp.linspace(0.9, 1.1, 8)
    batched = jax.tree.map(lambda x: scale[:, None] * x, params)
    return jax.vmap(simulate), batched


def net():
    """10 CA1 cells with sparse ionotropic synapses, 20 ms."""
    cell = _ca1_cell()
    net = jx.Network([cell for _ in range(10)])
    np.random.seed(0)
    jx.sparse_connect(net.cell("all"), net.cell("all"), IonotropicSynapse(), 0.1)
    return _make_simulate(net, t_max=20.0)


def net_large():
    """1000 small cells with HH and sparse ionotropic synapses, 10 ms."""
    cell = _small_cell()
    net = jx.Network([cell for _ in range(1000)])
    net.insert(HH())
    np.random.seed(0)
    jx.sparse_connect(net.cell("all"), net.cell("all"), IonotropicSynapse(), 0.001)
    return _make_simulate(net, t_max=10.0)


SCENARIOS = {f.__name__: f for f in [comp, cell, cell_vmap, net, net_large]}


@cache
def _warm_up():
    """Builds the models once (untimed), so that one-time JAX costs (compiling the small
    eager ops used while building) do not depend on which scenario runs first."""
    _ca1_cell()
    _small_cell()


def _compile_and_time(fn, params):
    """Returns (compile time, fastest of 3 runs)."""
    t0 = time.perf_counter()
    compiled = jax.jit(fn).lower(params).compile()
    compile_time = time.perf_counter() - t0
    jax.block_until_ready(compiled(params))  # Warm-up.
    runs = []
    for _ in range(3):
        t0 = time.perf_counter()
        jax.block_until_ready(compiled(params))
        runs.append(time.perf_counter() - t0)
    return compile_time, min(runs)


def measure(name: str) -> dict:
    """Runs one scenario and returns its timings in seconds."""
    _warm_up()
    t0 = time.perf_counter()
    simulate, params = SCENARIOS[name]()
    build = time.perf_counter() - t0

    def loss(params):
        return jnp.mean(simulate(params))

    compile_time, run_time = _compile_and_time(simulate, params)
    grad_compile_time, grad_time = _compile_and_time(jax.grad(loss), params)
    return {
        "build": build,
        "compile": compile_time,
        "simulate": run_time,
        "grad compile": grad_compile_time,
        "grad": grad_time,
    }


def format_table(results: dict, title: str, fmt: str = ".3f") -> str:
    """Formats `{scenario: {column: value}}` as a table."""
    width = max(len(name) for name in [title, *results])
    lines = [title.ljust(width) + "".join(f"{c:>14}" for c in COLUMNS)]
    for name, row in results.items():
        values = "".join(f"{row[c]:>14{fmt}}" for c in COLUMNS)
        lines.append(name.ljust(width) + values)
    return "\n".join(lines)


@pytest.mark.perf
@pytest.mark.parametrize("name", SCENARIOS)
def test_perf(name):
    RESULTS[name] = measure(name)


if __name__ == "__main__":
    # Used by `perf_regression.py`: prints the timings of the given scenarios as JSON.
    print(json.dumps({name: measure(name) for name in sys.argv[1:]}))
