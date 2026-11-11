# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
"""Performance benchmarks for a few common models. Local only, not run in CI.

Run with `pytest -m perf` (without `-n`), which prints a table of seconds per scenario:

- build: constructing the model (e.g. reading the SWC file, inserting channels). JAX's
  one-time start-up costs are excluded by `warm_up()`; without it, the first `read_swc`
  in a process takes ~10 s longer.
- compile: tracing and compiling `jit(simulate)`.
- simulate: one run of the compiled simulation (fastest of 3).
- grad compile, grad: the same for `jit(grad(loss))`.

To compare against another git ref, run `python tests/perf_regression.py --base main`.
"""

import os
import time

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


def warm_up():
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
    t0 = time.perf_counter()
    simulate, params = SCENARIOS[name]()
    build = time.perf_counter() - t0

    def loss(params):
        return jnp.mean(simulate(params))

    compile_time, run_time = _compile_and_time(simulate, params)
    grad_compile_time, grad_time = _compile_and_time(jax.grad(loss), params)
    return dict(
        zip(COLUMNS, [build, compile_time, run_time, grad_compile_time, grad_time])
    )


def format_table(results, title, fmt="{:.3f}", mark_above=None) -> str:
    """Formats `{scenario: {column: value}}` as a table.

    Missing values are shown as '-', values above `mark_above` get a '*'."""

    def cell(v):
        if v is None:
            return "- "
        return fmt.format(v) + ("*" if mark_above and v > mark_above else " ")

    rows = [[title] + [c + " " for c in COLUMNS]]
    for name, values in results.items():
        values = values or {}
        rows.append([name] + [cell(values.get(c)) for c in COLUMNS])
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for row in rows:
        cells = [row[0].ljust(widths[0])]
        cells += [c.rjust(w) for c, w in zip(row[1:], widths[1:])]
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)


@pytest.fixture(scope="module")
def warmed_up():
    warm_up()


@pytest.mark.perf
@pytest.mark.parametrize("name", SCENARIOS)
def test_perf(name, warmed_up):
    RESULTS[name] = measure(name)
