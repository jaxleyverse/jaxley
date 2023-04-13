from typing import List

import jax
from jax import lax, grad, vmap, vjp
import jax.numpy as jnp
from neurax.mechanisms.hh_neuron import hh_neuron_gate
from neurax.mechanisms.glutamate_synapse import glutamate
from neurax.build_branched_tridiag import define_all_tridiags
from neurax.solver_voltage import solve_branched
from neurax.stimulus import get_external_input
from neurax.utils.cell_utils import index_of_loc


NUM_BRANCHES = -1
NSEG_PER_BRANCH = -1
SOLVER = ""
CHANNELS = []


def solve(
    cells,
    init_v,
    membrane_states,
    membrane_params,
    membrane_channels,
    syn_params,
    connectivity,
    stimuli,
    recordings,
    t_max,
    dt: float = 0.025,
    solver: str = "stone",
    checkpoint_inds: List[int] = [],
):
    """
    Solve function.
    """
    global CHANNELS
    CHANNELS = membrane_channels

    assert len(membrane_params) == len(membrane_channels)
    assert len(membrane_params) == len(membrane_states)

    state = prepare_state(
        cells,
        init_v,
        membrane_states,
        membrane_params,
        syn_params,
        stimuli,
        recordings,
        connectivity,
        t_max,
        dt,
        solver,
    )
    num_time_steps = int(t_max / dt) + 1

    # Initialize arrays for checkpoints.
    assert 0 not in checkpoint_inds, "Checkpoints at index 0 is not implemented."
    checkpoint_inds = [0] + checkpoint_inds  # Add 0 index for convenience later.

    current_state = state

    def inner_loop(current_state, cp_ind1, cp_ind2):
        current_state = lax.fori_loop(
            cp_ind1,
            cp_ind2,
            body_fun,
            current_state,
        )
        return current_state

    for cp_ind in range(len(checkpoint_inds) - 1):
        current_state = jax.checkpoint(inner_loop, static_argnums=(1, 2))(
            current_state,
            checkpoint_inds[cp_ind],
            checkpoint_inds[cp_ind + 1],
        )
    # The trace from the last checkpoint until the end should not be checkpointed to
    # avoid useless recomputation.
    current_state = inner_loop(current_state, checkpoint_inds[-1], num_time_steps)

    return current_state[-1]


def prepare_state(
    cells,
    init_v,
    membrane_states,
    membrane_params,
    syn_params,
    stimuli,
    recordings,
    connectivity,
    t_max,
    dt: float = 0.025,
    solver: str = "stone",
):
    global NUM_BRANCHES
    global NSEG_PER_BRANCH
    global SOLVER

    NUM_BRANCHES = cells[0].num_branches
    NSEG_PER_BRANCH = cells[0].nseg_per_branch
    SOLVER = solver

    for cell in cells:
        assert (
            cell.num_branches == NUM_BRANCHES
        ), "Different num_branches between cells."
        assert (
            cell.nseg_per_branch == NSEG_PER_BRANCH
        ), "Different nseg_per_branch between cells."

    num_recordings = len(recordings)
    num_time_steps = int(t_max / dt) + 1
    saveat = jnp.zeros((num_recordings, num_time_steps))

    rec_inds = [index_of_loc(r.branch_ind, r.loc, NSEG_PER_BRANCH) for r in recordings]
    rec_inds = jnp.asarray(rec_inds)
    rec_cell_inds = jnp.asarray([r.cell_ind for r in recordings])

    stim_inds = [index_of_loc(s.branch_ind, s.loc, NSEG_PER_BRANCH) for s in stimuli]
    stim_cell_inds = jnp.asarray([s.cell_ind for s in stimuli])
    stim_inds = jnp.asarray(stim_inds)
    stim_currents = jnp.asarray([s.current for s in stimuli])  # nA

    # Save voltage at the beginning.
    saveat = saveat.at[:, 0].set(init_v[rec_cell_inds, rec_inds])  # 0 = voltage

    t = 0.0
    init_state = (
        t,
        init_v,
        membrane_states,
        connectivity.init_syn_states,
        membrane_params,
        syn_params,
        stim_cell_inds,
        stim_inds,
        stim_currents,
        dt,
        cells[0].radius,
        cells[0].length_single_compartment,
        cells[0].num_neighbours,
        cells[0].coupling_conds,
        cells[0].parents_in_each_level,
        cells[0].branches_in_each_level,
        cells[0].parents,
        rec_cell_inds,
        rec_inds,
        connectivity.pre_syn_inds,
        connectivity.pre_syn_cell_inds,
        connectivity.grouped_post_syn_inds,
        connectivity.grouped_post_syns,
        saveat,
    )
    return init_state


def find_root(
    t,
    v,
    u,
    ss,
    params,
    syn_params,
    i_cell_inds,
    i_inds,
    i_stim,
    pre_syn_inds,
    pre_syn_cell_inds,
    grouped_post_syn_inds,
    grouped_post_syns,
    dt,
    radius,
    length_single_compartment,
    num_neighbours,
    coupling_conds,
    parents_in_each_level,
    branches_in_each_level,
    parents,
):
    voltages = v  # mV

    # Membrane input.
    voltage_terms = jnp.zeros_like(v)
    constant_terms = jnp.zeros_like(v)
    new_states = []
    for i, update_fn in enumerate(CHANNELS):
        membrane_current_terms, states = update_fn(voltages, u[i], dt, params[i])
        voltage_terms += membrane_current_terms[0]
        constant_terms += membrane_current_terms[1]

        new_states.append(states)

    # External input.
    i_ext = get_external_input(
        voltages, i_cell_inds, i_inds, i_stim, radius, length_single_compartment
    )

    # Synaptic input.
    synapse_current_terms, synapse_states = glutamate(
        voltages,
        ss,
        pre_syn_inds,
        pre_syn_cell_inds,
        grouped_post_syn_inds,
        grouped_post_syns,
        dt,
        syn_params,
    )
    (new_s,) = synapse_states
    syn_voltage_terms, syn_constant_terms = synapse_current_terms

    # Define quasi-tridiagonal system.
    lowers, diags, uppers, solves = vmap(
        define_all_tridiags, in_axes=(0, 0, 0, None, None, None, None, None)
    )(
        voltages,
        voltage_terms + syn_voltage_terms,
        i_ext + constant_terms + syn_constant_terms,
        num_neighbours,
        NSEG_PER_BRANCH,
        NUM_BRANCHES,
        dt,
        coupling_conds,
    )

    # Solve quasi-tridiagonal system.
    solves = vmap(solve_branched, in_axes=(None, None, None, 0, 0, 0, 0, None, None))(
        parents_in_each_level,
        branches_in_each_level,
        parents,
        lowers,
        diags,
        uppers,
        solves,
        -dt * coupling_conds,
        SOLVER,
    )
    ncells = len(voltages)
    new_v = jnp.reshape(solves, (ncells, NUM_BRANCHES * NSEG_PER_BRANCH))
    return new_v, new_states, new_s


def body_fun(i, state):
    """
    Body for fori_loop.
    """
    (
        t,
        v,
        u_inner,
        syn_states,
        params,
        syn_params,
        i_cell_inds,
        i_inds,
        i_stim,
        dt,
        radius,
        length_single_compartment,
        num_neighbours,
        coupling_conds,
        parents_in_each_level,
        branches_in_each_level,
        parents,
        rec_cell_inds,
        rec_inds,
        pre_syn_inds,
        pre_syn_cell_inds,
        grouped_post_syn_inds,
        grouped_post_syns,
        saveat,
    ) = state

    v, u_inner, syn_states = find_root(
        t,
        v,
        u_inner,
        syn_states,
        params,
        syn_params,
        i_cell_inds,
        i_inds,
        i_stim[:, i],
        pre_syn_inds,
        pre_syn_cell_inds,
        grouped_post_syn_inds,
        grouped_post_syns,
        dt,
        radius,
        length_single_compartment,
        num_neighbours,
        coupling_conds,
        parents_in_each_level,
        branches_in_each_level,
        parents,
    )
    t += dt

    saveat = saveat.at[:, i + 1].set(v[rec_cell_inds, rec_inds])  # 0 = voltage

    return (
        t,
        v,
        u_inner,
        syn_states,
        params,
        syn_params,
        i_cell_inds,
        i_inds,
        i_stim,
        dt,
        radius,
        length_single_compartment,
        num_neighbours,
        coupling_conds,
        parents_in_each_level,
        branches_in_each_level,
        parents,
        rec_cell_inds,
        rec_inds,
        pre_syn_inds,
        pre_syn_cell_inds,
        grouped_post_syn_inds,
        grouped_post_syns,
        saveat,
    )
