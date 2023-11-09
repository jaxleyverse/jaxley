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


def test_compartment():
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = nx.step_current(0.5, 1.0, 0.02, dt, t_max)

    comp = nx.Compartment().initialize()
    comp.insert(HHChannel())
    comp.record()
    comp.stimulate(current)

    voltages = nx.integrate(comp, delta_t=dt)

    voltages_081123 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703,
                -50.08343917,
                -20.4930498,
                32.59313718,
                2.43694172,
                -26.28659049,
                -50.60083575,
                -73.00785374,
                -75.70088187,
                -75.58850932,
            ]
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


def test_branch():
    nseg_per_branch = 2
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = nx.step_current(0.5, 1.0, 0.02, dt, t_max)

    comp = nx.Compartment().initialize()
    branch = nx.Branch([comp for _ in range(nseg_per_branch)]).initialize()
    branch.insert(HHChannel())
    branch.comp(0.0).record()
    branch.comp(0.0).stimulate(current)

    voltages = nx.integrate(branch, delta_t=dt)

    voltages_081123 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703,
                -56.84345977,
                -47.34841899,
                -31.44840463,
                31.42082062,
                7.06374191,
                -22.55693822,
                -46.57781433,
                -71.30357569,
                -75.67506729,
            ]
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


def test_cell():
    nseg_per_branch = 2
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = nx.step_current(0.5, 1.0, 0.02, dt, t_max)

    depth = 2
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]

    comp = nx.Compartment().initialize()
    branch = nx.Branch([comp for _ in range(nseg_per_branch)]).initialize()
    cell = nx.Cell([branch for _ in range(len(parents))], parents=parents)
    cell.insert(HHChannel())
    cell.branch(1).comp(0.0).record()
    cell.branch(1).comp(0.0).stimulate(current)

    voltages = nx.integrate(cell, delta_t=dt)

    voltages_081123 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703,
                -60.77514388,
                -57.04487857,
                -57.47607922,
                -56.17604462,
                -54.14448985,
                -50.28092319,
                -37.61368601,
                24.47352735,
                10.24274831,
            ]
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


def test_net():
    nseg_per_branch = 2
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = nx.step_current(0.5, 1.0, 0.02, dt, t_max)

    depth = 2
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]

    comp = nx.Compartment().initialize()
    branch = nx.Branch([comp for _ in range(nseg_per_branch)]).initialize()
    cell1 = nx.Cell([branch for _ in range(len(parents))], parents=parents)
    cell2 = nx.Cell([branch for _ in range(len(parents))], parents=parents)

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
                -60.77514388,
                -57.04487857,
                -57.47607922,
                -56.17604462,
                -54.14448985,
                -50.28092319,
                -37.61368601,
                24.47352735,
                10.24274831,
            ],
            [
                -70.0,
                -66.12980895,
                -59.94208128,
                -55.74082517,
                -55.34657106,
                -52.32113275,
                -44.76100591,
                -3.65687352,
                22.581919,
                -8.29885715,
                -33.70855517,
            ],
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"
