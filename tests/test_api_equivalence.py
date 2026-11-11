# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import connect, connectivity_matrix_connect, fully_connect
from jaxley.integrate import build_init_and_step_fn
from jaxley.synapses import IonotropicSynapse
from tests import builders as build


def test_api_equivalence_morphology():
    """Test the API for how one can build morphologies from scratch."""
    ncomp_per_branch = 2
    depth = 2
    dt = 0.025

    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    parents = jnp.asarray(parents)
    num_branches = len(parents)

    comp = build.comp()

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


def test_solver_backends_comp():
    """Test whether ways of adding synapses are equivalent."""
    comp = build.comp()

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


def test_solver_backends_branch():
    """Test whether ways of adding synapses are equivalent."""
    branch = build.branch(ncomp=4)

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
def test_solver_backends_cell():
    """Test whether ways of adding synapses are equivalent."""
    cell = build.cell(nbranches=4, ncomp=4)

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


def test_solver_backends_net():
    """Test whether ways of adding synapses are equivalent."""
    net = build.net(ncells=2, nbranches=4, ncomp=4)

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


def test_api_equivalence_synapses():
    """Test whether the ways of adding synapses are equivalent.

    `connect` with explicit locations, `fully_connect` and
    `connectivity_matrix_connect` all place the synapses 0 -> 1 and 1 -> 0 at branch
    0, compartment 0 of each cell.
    """
    net1 = build.net(ncells=2, nbranches=4, ncomp=4)
    connect(
        net1.cell(0).branch(0).comp(0),
        net1.cell(1).branch(0).comp(0),
        IonotropicSynapse(),
    )
    connect(
        net1.cell(1).branch(0).comp(0),
        net1.cell(0).branch(0).comp(0),
        IonotropicSynapse(),
    )

    net2 = build.net(ncells=2, nbranches=4, ncomp=4)
    fully_connect(net2.cell(0), net2.cell(1), IonotropicSynapse())
    fully_connect(net2.cell(1), net2.cell(0), IonotropicSynapse())

    net3 = build.net(ncells=2, nbranches=4, ncomp=4)
    connectivity_matrix_connect(
        net3.cell("all"),
        net3.cell("all"),
        IonotropicSynapse(),
        np.array([[False, True], [True, False]]),
    )

    nets = [net1, net2, net3]
    endpoints = ["pre_index", "post_index"]
    for net in nets[1:]:
        pd.testing.assert_frame_equal(
            net.edges[endpoints].sort_values(endpoints).reset_index(drop=True),
            net1.edges[endpoints].sort_values(endpoints).reset_index(drop=True),
        )

    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )
    for net in nets:
        net.cell(0).branch(0).loc(0.0).stimulate(current)
        net.cell(0).branch(0).loc(0.5).record()
        net.cell(1).branch(3).loc(0.5).record()

    voltages = [jx.integrate(net) for net in nets]
    for v in voltages[1:]:
        assert (
            np.max(np.abs(voltages[0] - v)) < 1e-8
        ), "Voltages do not match between synapse APIs."


def test_api_equivalence_continued_simulation():
    cell = build.cell(nbranches=3, ncomp=2)
    cell.insert(HH())
    cell[0, 1].record()

    v1 = jx.integrate(cell, t_max=4.0)
    v21, states = jx.integrate(cell, return_states=True, t_max=2.0)
    v22 = jx.integrate(cell, all_states=states, t_max=2.0)

    v2 = jnp.concatenate([v21, v22[:, 1:]], axis=1)
    assert np.max(np.abs(v1 - v2)) < 1e-8


def test_api_equivalence_network_matches_cell():
    """Test whether a network with w=0 synapses equals the individual cells.

    This runs an unequal number of compartments per branch."""
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )

    branch1 = build.branch(ncomp=1)
    branch2 = build.branch(ncomp=2)
    branch3 = build.branch(ncomp=3)
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


def test_api_init_step_to_integrate():
    cell = build.cell(nbranches=3, ncomp=2)
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
    rec_inds = cell.rec_info.rec_index.to_numpy()
    rec_states = cell.rec_info.state.to_numpy()

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
