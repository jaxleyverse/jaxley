# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import connect
from jaxley.integrate import build_init_and_step_fn
from jaxley.synapses import IonotropicSynapse


def test_api_equivalence_morphology(SimpleComp):
    """Test the API for how one can build morphologies from scratch."""
    ncomp_per_branch = 2
    depth = 2
    dt = 0.025

    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    parents = jnp.asarray(parents)
    num_branches = len(parents)

    comp = SimpleComp()

    branch1 = jx.Branch([comp for _ in range(ncomp_per_branch)])
    cell1 = jx.Cell([branch1 for _ in range(num_branches)], parents=parents)

    branch2 = jx.Branch(comp, ncomp=ncomp_per_branch)
    cell2 = jx.Cell(branch2, parents=parents)

    cell1.branch(2).loc(0.4).record()
    cell2.branch(2).loc(0.4).record()

    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )
    cell1.branch(1).loc(1.0).stimulate(current)
    cell2.branch(1).loc(1.0).stimulate(current)

    voltages1 = jx.integrate(cell1, delta_t=dt)
    voltages2 = jx.integrate(cell2, delta_t=dt)
    assert (
        jnp.max(jnp.abs(voltages1 - voltages2)) < 1e-8
    ), "Voltages do not match between morphology APIs."


def test_solver_backends_comp(SimpleComp):
    """Test whether ways of adding synapses are equivalent."""
    comp = SimpleComp()

    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )
    comp.stimulate(current)
    comp.record()

    voltages_jx_cpu = jx.integrate(comp, voltage_solver="jaxley.dhs.cpu")
    voltages_jx_gpu = jx.integrate(comp, voltage_solver="jaxley.dhs.gpu")
    voltages_jx_sparse = jx.integrate(comp, voltage_solver="jax.sparse")

    message = "Voltages do not match between"
    max_error = np.max(np.abs(voltages_jx_cpu - voltages_jx_gpu))
    assert max_error < 1e-8, f"{message} cpu/gpu. Error={max_error}"
    max_error = np.max(np.abs(voltages_jx_cpu - voltages_jx_sparse))
    assert max_error < 1e-8, f"{message} cpu/sparse. Error={max_error}"


def test_solver_backends_branch(SimpleBranch):
    """Test whether ways of adding synapses are equivalent."""
    branch = SimpleBranch(4)

    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )
    branch.loc(0.0).stimulate(current)
    branch.loc(0.5).record()

    voltages_jx_cpu = jx.integrate(branch, voltage_solver="jaxley.dhs.cpu")
    voltages_jx_gpu = jx.integrate(branch, voltage_solver="jaxley.dhs.gpu")
    voltages_jx_sparse = jx.integrate(branch, voltage_solver="jax.sparse")

    message = "Voltages do not match between"
    max_error = np.max(np.abs(voltages_jx_cpu - voltages_jx_gpu))
    assert max_error < 1e-8, f"{message} cpu/gpu. Error={max_error}"
    max_error = np.max(np.abs(voltages_jx_cpu - voltages_jx_sparse))
    assert max_error < 1e-8, f"{message} cpu/sparse. Error={max_error}"


@pytest.mark.slow
def test_solver_backends_cell(SimpleCell):
    """Test whether ways of adding synapses are equivalent."""
    cell = SimpleCell(4, 4)

    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )
    cell.branch(0).loc(0.0).stimulate(current)
    cell.branch(0).loc(0.5).record()
    cell.branch(3).loc(0.5).record()

    voltages_jx_cpu = jx.integrate(cell, voltage_solver="jaxley.dhs.cpu")
    voltages_jx_gpu = jx.integrate(cell, voltage_solver="jaxley.dhs.gpu")
    voltages_jx_sparse = jx.integrate(cell, voltage_solver="jax.sparse")

    message = "Voltages do not match between"
    max_error = np.max(np.abs(voltages_jx_cpu - voltages_jx_gpu))
    assert max_error < 1e-8, f"{message} cpu/gpu. Error={max_error}"
    max_error = np.max(np.abs(voltages_jx_cpu - voltages_jx_sparse))
    assert max_error < 1e-8, f"{message} cpu/sparse. Error={max_error}"


def test_solver_backends_net(SimpleNet):
    """Test whether ways of adding synapses are equivalent."""
    net = SimpleNet(2, 4, 4)

    connect(
        net.cell(0).branch(0).loc(1.0),
        net.cell(1).branch(3).loc(1.0),
        IonotropicSynapse(),
    )
    connect(
        net.cell(1).branch(1).loc(0.8),
        net.cell(0).branch(3).loc(0.1),
        IonotropicSynapse(),
    )

    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )
    net.cell(0).branch(0).loc(0.0).stimulate(current)
    net.cell(0).branch(0).loc(0.5).record()
    net.cell(1).branch(3).loc(0.5).record()

    voltages_jx_cpu = jx.integrate(net, voltage_solver="jaxley.dhs.cpu")
    voltages_jx_gpu = jx.integrate(net, voltage_solver="jaxley.dhs.gpu")
    voltages_jx_sparse = jx.integrate(net, voltage_solver="jax.sparse")

    message = "Voltages do not match between"
    max_error = np.max(np.abs(voltages_jx_cpu - voltages_jx_gpu))
    assert max_error < 1e-8, f"{message} cpu/gpu. Error={max_error}"
    max_error = np.max(np.abs(voltages_jx_cpu - voltages_jx_sparse))
    assert max_error < 1e-8, f"{message} cpu/sparse. Error={max_error}"


def test_api_equivalence_synapses(SimpleNet):
    """Test whether ways of adding synapses are equivalent."""
    net1 = SimpleNet(2, 4, 4)

    connect(
        net1.cell(0).branch(0).loc(1.0),
        net1.cell(1).branch(3).loc(1.0),
        IonotropicSynapse(),
    )
    connect(
        net1.cell(1).branch(1).loc(0.8),
        net1.cell(0).branch(3).loc(0.1),
        IonotropicSynapse(),
    )

    net2 = SimpleNet(2, 4, 4)
    pre = net2.cell(0).branch(0).loc(1.0)
    post = net2.cell(1).branch(3).loc(1.0)
    connect(pre, post, IonotropicSynapse())

    pre = net2.cell(1).branch(1).loc(0.8)
    post = net2.cell(0).branch(3).loc(0.1)
    connect(pre, post, IonotropicSynapse())

    for net in [net1, net2]:
        current = jx.step_current(
            i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
        )
        net.cell(0).branch(0).loc(0.0).stimulate(current)
        net.cell(0).branch(0).loc(0.5).record()
        net.cell(1).branch(3).loc(0.5).record()

    voltages1 = jx.integrate(net1)
    voltages2 = jx.integrate(net2)

    assert (
        np.max(np.abs(voltages1 - voltages2)) < 1e-8
    ), "Voltages do not match between synapse APIs."


def test_api_equivalence_continued_simulation(SimpleCell):
    cell = SimpleCell(3, 2)
    cell.insert(HH())
    cell[0, 1].record()

    v1 = jx.integrate(cell, t_max=4.0)
    v21, states = jx.integrate(cell, return_states=True, t_max=2.0)
    v22 = jx.integrate(cell, all_states=states, t_max=2.0)

    v2 = jnp.concatenate([v21, v22[:, 1:]], axis=1)
    assert np.max(np.abs(v1 - v2)) < 1e-8


def test_api_equivalence_network_matches_cell(SimpleBranch):
    """Test whether a network with w=0 synapses equals the individual cells.

    This runs an unequal number of compartments per branch."""
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )

    branch1 = SimpleBranch(ncomp=1)
    branch2 = SimpleBranch(ncomp=2)
    branch3 = SimpleBranch(ncomp=3)
    cell1 = jx.Cell([branch1, branch2, branch3], parents=[-1, 0, 0])
    cell2 = jx.Cell([branch1, branch2], parents=[-1, 0])
    cell1.insert(HH())
    cell2.insert(HH())

    net = jx.Network([cell1, cell2])
    pre = net.cell(0).branch(2).comp(2)
    post = net.cell(1).branch(1).comp(1)
    connect(pre, post, IonotropicSynapse())
    net.IonotropicSynapse.edge("all").set("IonotropicSynapse_gS", 0.0)

    net.cell(0).branch(2).comp(2).stimulate(current)
    net.cell(0).branch(0).comp(0).record()

    net.cell(1).branch(1).comp(1).stimulate(current)
    net.cell(1).branch(0).comp(0).record()
    voltages_net = jx.integrate(net, delta_t=dt, voltage_solver="jaxley.dhs.cpu")

    cell1.branch(2).comp(2).stimulate(current)
    cell1.branch(0).comp(0).record()

    cell2.branch(1).comp(1).stimulate(current)
    cell2.branch(0).comp(0).record()
    voltages_cell1 = jx.integrate(cell1, delta_t=dt, voltage_solver="jaxley.dhs.cpu")
    voltages_cell2 = jx.integrate(cell2, delta_t=dt, voltage_solver="jaxley.dhs.cpu")
    voltages_cells = jnp.concatenate([voltages_cell1, voltages_cell2], axis=0)

    max_error = np.max(np.abs(voltages_net - voltages_cells))
    assert max_error < 1e-8, f"Error is {max_error}"


def test_api_init_step_to_integrate(SimpleCell):
    cell = SimpleCell(3, 2)
    cell.insert(HH())
    cell[0, 1].record()

    # Internal integration function API
    delta_t = 0.025  # Default delta_t is 0.025
    v1 = jx.integrate(cell, t_max=4.0, delta_t=delta_t)

    # Flexibe init and step API
    init_fn, step_fn = build_init_and_step_fn(cell)

    params = cell.get_parameters()
    states, params = init_fn(params)
    step_fn_ = jax.jit(step_fn)
    rec_inds = cell.recordings.rec_index.to_numpy()
    rec_states = cell.recordings.state.to_numpy()

    steps = int(4.0 / delta_t)  # Steps to integrate
    recordings = [
        states[rec_state][rec_ind][None]
        for rec_state, rec_ind in zip(rec_states, rec_inds)
    ]
    externals = cell.externals
    for _ in range(steps):
        states = step_fn_(states, params, externals, delta_t=delta_t)
        recs = jnp.asarray(
            [
                states[rec_state][rec_ind]
                for rec_state, rec_ind in zip(rec_states, rec_inds)
            ]
        )
        recordings.append(recs)

    rec = jnp.stack(recordings, axis=0).T

    assert jnp.allclose(v1, rec)
