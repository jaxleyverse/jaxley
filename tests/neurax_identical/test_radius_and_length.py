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


def test_radius_and_length_compartment():
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = nx.step_current(0.5, 1.0, 0.02, dt, t_max)

    comp = nx.Compartment().initialize()

    np.random.seed(1)
    comp.set_params("length", 5 * np.random.rand(1))
    comp.set_params("radius", np.random.rand(1))

    comp.insert(HHChannel())
    comp.record()
    comp.stimulate(current)

    voltages = nx.integrate(comp, delta_t=dt)

    voltages_081123 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703,
                45.83306656,
                29.72581199,
                -15.44336119,
                -39.98282246,
                -66.77430474,
                -75.56725325,
                -75.75072145,
                -75.48894182,
                -75.15588746,
            ]
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


def test_radius_and_length_branch():
    nseg_per_branch = 2
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = nx.step_current(0.5, 1.0, 0.02, dt, t_max)

    comp = nx.Compartment().initialize()
    branch = nx.Branch([comp for _ in range(nseg_per_branch)]).initialize()

    np.random.seed(1)
    branch.set_params("length", 5 * np.random.rand(2))
    branch.set_params("radius", np.random.rand(2))

    branch.insert(HHChannel())
    branch.comp(0.0).record()
    branch.comp(0.0).stimulate(current)

    voltages = nx.integrate(branch, delta_t=dt)

    voltages_081123 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703,
                57.69711962,
                27.50364167,
                -20.46106389,
                -44.65846514,
                -70.81919426,
                -75.78039147,
                -75.74705134,
                -75.46858134,
                -75.13087565,
            ]
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


def test_radius_and_length_cell():
    nseg_per_branch = 2
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = nx.step_current(0.5, 1.0, 0.02, dt, t_max)

    depth = 2
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    num_branches = len(parents)

    comp = nx.Compartment().initialize()
    branch = nx.Branch([comp for _ in range(nseg_per_branch)]).initialize()
    cell = nx.Cell([branch for _ in range(len(parents))], parents=parents)

    np.random.seed(1)
    cell.set_params("length", 5 * np.random.rand(2 * num_branches))
    cell.set_params("radius", np.random.rand(2 * num_branches))

    cell.insert(HHChannel())
    cell.branch(1).comp(0.0).record()
    cell.branch(1).comp(0.0).stimulate(current)

    voltages = nx.integrate(cell, delta_t=dt)

    voltages_081123 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703,
                1.40098029,
                41.85849945,
                -3.41062602,
                -30.40531156,
                -54.43060925,
                -73.96911419,
                -75.74495833,
                -75.58187443,
                -75.27068799,
            ]
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


def test_radius_and_length_net():
    nseg_per_branch = 2
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = nx.step_current(0.5, 1.0, 0.02, dt, t_max)

    depth = 2
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    num_branches = len(parents)

    comp = nx.Compartment().initialize()
    branch = nx.Branch([comp for _ in range(nseg_per_branch)]).initialize()
    cell1 = nx.Cell([branch for _ in range(len(parents))], parents=parents)
    cell2 = nx.Cell([branch for _ in range(len(parents))], parents=parents)

    np.random.seed(1)
    cell1.set_params("length", 5 * np.random.rand(2 * num_branches))
    cell1.set_params("radius", np.random.rand(2 * num_branches))

    np.random.seed(2)
    cell2.set_params("length", 5 * np.random.rand(2 * num_branches))
    cell2.set_params("radius", np.random.rand(2 * num_branches))

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
                1.40098029,
                41.85849945,
                -3.41062602,
                -30.40531156,
                -54.43060925,
                -73.96911419,
                -75.74495833,
                -75.58187443,
                -75.27068799,
            ],
            [
                -70.0,
                -66.46899201,
                -14.64499375,
                43.92453203,
                3.25054262,
                -25.60587037,
                -49.03042036,
                -71.89299285,
                -75.57211264,
                -75.4917933,
                -75.16938855,
            ],
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"
