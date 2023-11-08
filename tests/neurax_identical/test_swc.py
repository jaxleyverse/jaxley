from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

import jax.numpy as jnp
import numpy as np

import neurax as nx
from neurax.channels import HHChannel
from neurax.synapses import GlutamateSynapse


def test_swc_cell():
    dt = 0.025  # ms
    t_max = 5.0  # ms
    time_vec = jnp.arange(0.0, t_max + dt, dt)
    current = nx.step_current(0.5, 1.0, 0.2, time_vec)

    cell = nx.read_swc("../morph.swc", nseg=2, max_branch_len=300.0)
    cell.insert(HHChannel())
    cell.branch(1).comp(0.0).record()
    cell.branch(1).comp(0.0).stimulate(current)

    voltages = nx.integrate(cell, delta_t=dt)

    voltages_081123 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703,
                -57.02065054,
                -49.74541341,
                -46.15576812,
                -23.71760359,
                25.19649297,
                1.99881676,
                -25.42530891,
                -48.23078669,
                -69.2479302,
            ]
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


def test_swc_net():
    dt = 0.025  # ms
    t_max = 5.0  # ms
    time_vec = jnp.arange(0.0, t_max + dt, dt)
    current = nx.step_current(0.5, 1.0, 0.2, time_vec)

    cell1 = nx.read_swc("../morph.swc", nseg=2, max_branch_len=300.0)
    cell2 = nx.read_swc("../morph.swc", nseg=2, max_branch_len=300.0)

    connectivities = [
        nx.Connectivity(GlutamateSynapse(), [nx.Connection(0, 0, 0.0, 1, 0, 0.0)])
    ]
    network = nx.Network([cell1, cell2], connectivities)
    network.insert(HHChannel())

    for cell_ind in range(2):
        network.cell(cell_ind).branch(1).comp(0.0).record()

    for stim_ind in range(2):
        network.cell(stim_ind).branch(1).comp(0.0).stimulate(current)

    voltages = nx.integrate(network, delta_t=dt)

    voltages_081123 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703,
                -57.02065054,
                -49.74541341,
                -46.15576812,
                -23.71760359,
                25.19649297,
                1.99881676,
                -25.42530891,
                -48.23078669,
                -69.2479302,
            ],
            [
                -70.0,
                -66.52400879,
                -57.01032453,
                -49.72868896,
                -46.10615669,
                -23.43965885,
                25.17580247,
                1.83985193,
                -25.55061179,
                -48.34838475,
                -69.29381344,
            ],
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"
