from typing import List

import jax
from jax import lax, vmap
import jax.numpy as jnp

from neurax.build_branched_tridiag import define_all_tridiags
from neurax.solver_voltage import solve_branched
from neurax.stimulus import get_external_input
from neurax.utils.cell_utils import index_of_loc
from neurax.utils.syn_utils import postsyn_voltage_updates


NUM_BRANCHES = []
PARENTS = []
NSEG_PER_BRANCH = -1
SOLVER = ""
MEM_CHANNELS = []
SYN_CHANNELS = []


def solve(
    cells,
    connectivities,
    init_v,
    mem_states,
    mem_params,
    mem_channels,
    syn_states,
    syn_params,
    syn_channels,
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
    global MEM_CHANNELS
    global SYN_CHANNELS
    MEM_CHANNELS = mem_channels
    SYN_CHANNELS = syn_channels

    assert len(mem_params) == len(mem_channels)
    assert len(mem_params) == len(mem_states)

    state = prepare_state(
        cells,
        init_v,
        mem_states,
        mem_params,
        syn_states,
        syn_params,
        stimuli,
        recordings,
        connectivities,
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
    mem_states,
    mem_params,
    syn_states,
    syn_params,
    stimuli,
    recordings,
    connectivities,
    t_max,
    dt: float = 0.025,
    solver: str = "stone",
):
    global NUM_BRANCHES
    global NSEG_PER_BRANCH
    global SOLVER
    global PARENTS

    NUM_BRANCHES = [cell.num_branches for cell in cells]
    PARENTS = [cell.parents for cell in cells]
    NSEG_PER_BRANCH = cells[0].nseg_per_branch
    SOLVER = solver

    for cell in cells:
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
    cumsum_num_branches = jnp.cumsum(jnp.asarray([0] + NUM_BRANCHES))
    cated_voltage = jnp.concatenate(init_v)
    saveat = saveat.at[:, 0].set(
        cated_voltage[cumsum_num_branches[rec_cell_inds] + rec_inds]
    )

    t = 0.0
    init_state = (
        t,
        init_v,
        mem_states,
        syn_states,
        mem_params,
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
        rec_cell_inds,
        rec_inds,
        [c.pre_syn_inds for c in connectivities],
        [c.pre_syn_cell_inds for c in connectivities],
        [c.grouped_post_syn_inds for c in connectivities],
        [c.grouped_post_syns for c in connectivities],
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
):
    voltages = v  # mV
    cated_voltages = jnp.concatenate(voltages)
    cumsum_num_branches = jnp.cumsum(jnp.asarray([0] + NUM_BRANCHES))

    # Membrane input.
    voltage_terms = jnp.zeros_like(cated_voltages)
    constant_terms = jnp.zeros_like(cated_voltages)
    new_states = []
    for i, update_fn in enumerate(MEM_CHANNELS):
        membrane_current_terms, states = update_fn(
            cated_voltages,
            jnp.concatenate(u[i], axis=1),
            jnp.concatenate(params[i], axis=1),
            dt,
        )
        voltage_terms += membrane_current_terms[0]
        constant_terms += membrane_current_terms[1]
        # Above, we concatenated the voltages, states, and params in order to allow
        # embarassingly parallel exectution. Now we have to bring the states back into
        # the original shape.
        reconstructed_states = []
        k = 0
        for ind in NUM_BRANCHES:
            reconstructed_states.append(states[:, k : k + ind * NSEG_PER_BRANCH])
            k += ind
        new_states.append(reconstructed_states)

    # External input.
    i_ext = get_external_input(
        cated_voltages,
        cumsum_num_branches[i_cell_inds] + i_inds,
        i_stim,
        radius,
        length_single_compartment,
    )

    # Synaptic input.
    syn_voltage_terms = jnp.zeros_like(cated_voltages)
    syn_constant_terms = jnp.zeros_like(cated_voltages)
    new_syn_states = []
    for i, update_fn in enumerate(SYN_CHANNELS):
        synapse_current_terms, synapse_states = update_fn(
            cated_voltages,
            ss[i],
            cumsum_num_branches[pre_syn_cell_inds[i]] + pre_syn_inds[i],
            dt,
            syn_params[i],
        )
        synapse_current_terms = postsyn_voltage_updates(
            cumsum_num_branches,
            cated_voltages,
            grouped_post_syn_inds[i],
            grouped_post_syns[i],
            *synapse_current_terms
        )
        syn_voltage_terms += synapse_current_terms[0]
        syn_constant_terms += synapse_current_terms[1]
        new_syn_states.append(synapse_states)

    v_terms = voltage_terms + syn_voltage_terms
    c_terms = i_ext + constant_terms + syn_constant_terms

    reconstructed_v_terms = []
    reconstructed_c_terms = []
    k = 0
    for ind in NUM_BRANCHES:
        reconstructed_v_terms.append(v_terms[k : k + ind * NSEG_PER_BRANCH])
        reconstructed_c_terms.append(c_terms[k : k + ind * NSEG_PER_BRANCH])
        k += ind

    # Define quasi-tridiagonal system.
    lowers, diags, uppers, solves = vmap(
        define_all_tridiags, in_axes=(0, 0, 0, None, None, None, None, None)
    )(
        jnp.stack(voltages),  # TODO
        jnp.stack(reconstructed_v_terms),
        jnp.stack(reconstructed_c_terms),
        num_neighbours,
        NSEG_PER_BRANCH,
        NUM_BRANCHES[0],  # TODO
        dt,
        coupling_conds,
    )
    # lowers has shape [2, 15, 3]

    # Solve quasi-tridiagonal system.
    solves = vmap(solve_branched, in_axes=(None, None, None, 0, 0, 0, 0, None, None))(
        parents_in_each_level,
        branches_in_each_level,
        PARENTS[0],  # TODO
        lowers,
        diags,
        uppers,
        solves,
        -dt * coupling_conds,
        SOLVER,
    )
    ncells = len(voltages)
    new_v = jnp.reshape(solves, (ncells, NUM_BRANCHES[0] * NSEG_PER_BRANCH))  # TODO
    return [new_v[0], new_v[1]], new_states, new_syn_states


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
    )
    t += dt

    cumsum_num_branches = jnp.cumsum(jnp.asarray([0] + NUM_BRANCHES))
    cated_voltage = jnp.concatenate(v)
    saveat = saveat.at[:, i + 1].set(
        cated_voltage[cumsum_num_branches[rec_cell_inds] + rec_inds]
    )

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
        rec_cell_inds,
        rec_inds,
        pre_syn_inds,
        pre_syn_cell_inds,
        grouped_post_syn_inds,
        grouped_post_syns,
        saveat,
    )
