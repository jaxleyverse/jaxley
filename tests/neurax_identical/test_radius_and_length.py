from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

import jax.numpy as jnp
import numpy as np

import jaxley as jx
from jaxley.channels import HH
from jaxley.synapses import IonotropicSynapse


def test_radius_and_length_compartment():
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = jx.step_current(0.5, 1.0, 0.02, dt, t_max)

    comp = jx.Compartment().initialize()

    np.random.seed(1)
    comp.set("length", 5 * np.random.rand(1))
    comp.set("radius", np.random.rand(1))

    comp.insert(HH())
    comp.record()
    comp.stimulate(current)

    voltages = jx.integrate(comp, delta_t=dt)

    voltages_081123 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703,
                45.83306656,
                29.72581199,
                -15.28462737,
                -39.8274255,
                -66.59985023,
                -75.55397377,
                -75.74933676,
                -75.48829132,
                -75.15524717,
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
    current = jx.step_current(0.5, 1.0, 0.02, dt, t_max)

    comp = jx.Compartment().initialize()
    branch = jx.Branch([comp for _ in range(nseg_per_branch)]).initialize()

    np.random.seed(1)
    branch.set("length", 5 * np.random.rand(2))
    branch.set("radius", np.random.rand(2))

    branch.insert(HH())
    branch.loc(0.0).record()
    branch.loc(0.0).stimulate(current)

    voltages = jx.integrate(branch, delta_t=dt)

    voltages_081123 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703,
                57.69711962,
                27.50364167,
                -20.26260571,
                -44.421609,
                -70.6396421,
                -75.76980478,
                -75.74539948,
                -75.46741856,
                -75.12962902,
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
    current = jx.step_current(0.5, 1.0, 0.02, dt, t_max)

    depth = 2
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    num_branches = len(parents)

    comp = jx.Compartment().initialize()
    branch = jx.Branch([comp for _ in range(nseg_per_branch)]).initialize()
    cell = jx.Cell([branch for _ in range(len(parents))], parents=parents)

    np.random.seed(1)
    cell.set("length", 5 * np.random.rand(2 * num_branches))
    cell.set("radius", np.random.rand(2 * num_branches))

    cell.insert(HH())
    cell.branch(1).loc(0.0).record()
    cell.branch(1).loc(0.0).stimulate(current)

    voltages = jx.integrate(cell, delta_t=dt)

    voltages_081123 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703,
                1.40098029,
                41.85849945,
                -3.29312727,
                -30.32559832,
                -54.3463986,
                -73.94743834,
                -75.74328911,
                -75.58138665,
                -75.27029656,
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
    current = jx.step_current(0.5, 1.0, 0.02, dt, t_max)

    depth = 2
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    num_branches = len(parents)

    comp = jx.Compartment().initialize()
    branch = jx.Branch([comp for _ in range(nseg_per_branch)]).initialize()
    cell1 = jx.Cell([branch for _ in range(len(parents))], parents=parents)
    cell2 = jx.Cell([branch for _ in range(len(parents))], parents=parents)

    np.random.seed(1)
    cell1.set("length", 5 * np.random.rand(2 * num_branches))
    cell1.set("radius", np.random.rand(2 * num_branches))

    np.random.seed(2)
    cell2.set("length", 5 * np.random.rand(2 * num_branches))
    cell2.set("radius", np.random.rand(2 * num_branches))

    connectivities = [
        jx.Connectivity(IonotropicSynapse(), [jx.Connection(0, 0, 0.0, 1, 0, 0.0)])
    ]
    network = jx.Network([cell1, cell2], connectivities)
    network.insert(HH())

    for cell_ind in range(2):
        network.cell(cell_ind).branch(1).loc(0.0).record()

    for stim_ind in range(2):
        network.cell(stim_ind).branch(1).loc(0.0).stimulate(current)

    voltages = jx.integrate(network, delta_t=dt)

    voltages_040224 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703,
                1.40098029,
                41.85849945,
                -3.29312727,
                -30.32559832,
                -54.3463986,
                -73.94743834,
                -75.74328911,
                -75.58138665,
                -75.27029656,
            ],
            [
                -70.0,
                -66.46902476,
                -14.64500976,
                43.9244311,
                3.3488237,
                -25.54186113,
                -48.97168575,
                -71.86634877,
                -75.5702194,
                -75.49141107,
                -75.16913904,
            ],
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_040224))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"
