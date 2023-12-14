from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

import jax.numpy as jnp
import numpy as np

import jaxley as jx
from jaxley.channels import HH
from jaxley.synapses import GlutamateSynapse


def test_swc_cell():
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = jx.step_current(0.5, 1.0, 0.2, dt, t_max)

    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "../morph.swc")
    cell = jx.read_swc(fname, nseg=2, max_branch_len=300.0)
    cell.insert(HH())
    cell.branch(1).comp(0.0).record()
    cell.branch(1).comp(0.0).stimulate(current)

    voltages = jx.integrate(cell, delta_t=dt)

    voltages_081123 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703,
                -57.02065054,
                -49.74541341,
                -46.92144876,
                -27.85593412,
                25.00515973,
                4.52875333,
                -23.36842446,
                -46.25720697,
                -68.22233297,
            ]
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


def test_swc_net():
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = jx.step_current(0.5, 1.0, 0.2, dt, t_max)

    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "../morph.swc")
    cell1 = jx.read_swc(fname, nseg=2, max_branch_len=300.0)
    cell2 = jx.read_swc(fname, nseg=2, max_branch_len=300.0)

    connectivities = [
        jx.Connectivity(GlutamateSynapse(), [jx.Connection(0, 0, 0.0, 1, 0, 0.0)])
    ]
    network = jx.Network([cell1, cell2], connectivities)
    network.insert(HH())

    for cell_ind in range(2):
        network.cell(cell_ind).branch(1).comp(0.0).record()

    for stim_ind in range(2):
        network.cell(stim_ind).branch(1).comp(0.0).stimulate(current)

    voltages = jx.integrate(network, delta_t=dt)

    voltages_081123 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703,
                -57.02065054,
                -49.74541341,
                -46.92144876,
                -27.85593412,
                25.00515973,
                4.52875333,
                -23.36842446,
                -46.25720697,
                -68.22233297,
            ],
            [
                -70.0,
                -66.52400879,
                -57.01032453,
                -49.72868896,
                -46.87353462,
                -27.60185258,
                25.04026114,
                4.3565282,
                -23.50819992,
                -46.38431714,
                -68.28501838,
            ],
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"
