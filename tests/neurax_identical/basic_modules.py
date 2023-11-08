from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

import numpy as np
import jax.numpy as jnp

import neurax as nx
from neurax.channels import HHChannel
from neurax.synapses import GlutamateSynapse


def test_compartment():
    dt = 0.025  # ms
    t_max = 5.0  # ms

    time_vec = jnp.arange(0.0, t_max + dt, dt)
    current = nx.step_current(0.5, 1.0, 0.02, time_vec)

    comp = nx.Compartment().initialize()
    comp.insert(HHChannel())
    comp.record()
    comp.stimulate(current)

    voltages = nx.integrate(comp, delta_t=dt)

    voltages_081123 = None
    max_error = np.max(np.abs(voltages[:, ::10] - voltages_081123))
    tolerance = 0.0
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


def test_branch():
    nseg_per_branch = 2
    dt = 0.025  # ms
    t_max = 5.0  # ms

    time_vec = jnp.arange(0.0, t_max + dt, dt)
    current = nx.step_current(0.5, 1.0, 0.02, time_vec)

    comp = nx.Compartment().initialize()
    branch = nx.Branch([comp for _ in range(nseg_per_branch)]).initialize()
    branch.insert(HHChannel())
    branch.comp(0.0).record()
    branch.comp(0.0).stimulate(current)

    voltages = nx.integrate(branch, delta_t=dt)

    voltages_081123 = None
    max_error = np.max(np.abs(voltages[:, ::10] - voltages_081123))
    tolerance = 0.0
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


def test_cell():
    nseg_per_branch = 2
    dt = 0.025  # ms
    t_max = 5.0  # ms

    time_vec = jnp.arange(0.0, t_max + dt, dt)
    current = nx.step_current(0.5, 1.0, 0.02, time_vec)

    depth = 2
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]

    comp = nx.Compartment().initialize()
    branch = nx.Branch([comp for _ in range(nseg_per_branch)]).initialize()
    cell = nx.Cell([branch for _ in range(len(parents))], parents=parents)
    cell.insert(HHChannel())
    cell.branch(1).comp(0.0).record()
    cell.branch(1).comp(0.0).stimulate(current)

    voltages = nx.integrate(cell, delta_t=dt)

    voltages_081123 = None
    max_error = np.max(np.abs(voltages[:, ::10] - voltages_081123))
    tolerance = 0.0
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


def test_net():
    nseg_per_branch = 2
    dt = 0.025  # ms
    t_max = 5.0  # ms

    time_vec = jnp.arange(0.0, t_max + dt, dt)
    current = nx.step_current(0.5, 1.0, 0.02, time_vec)

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

    voltages_081123 = None
    max_error = np.max(np.abs(voltages[:, ::10] - voltages_081123))
    tolerance = 0.0
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"
