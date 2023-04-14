from typing import List

import jax
from jax import lax, vmap
import jax.numpy as jnp

from neurax.cell import merge_cells
from neurax.build_branched_tridiag import define_all_tridiags
from neurax.solver_voltage import solve_branched
from neurax.stimulus import get_external_input
from neurax.utils.cell_utils import index_of_loc
from neurax.utils.syn_utils import postsyn_voltage_updates


NUM_BRANCHES = []
CUMSUM_NUM_BRANCHES = []
COMB_PARENTS = []
COMB_PARENTS_IN_EACH_LEVEL = []
COMB_BRANCHES_IN_EACH_LEVEL = []
NUM_NEIGHBOURS = []

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
    global CUMSUM_NUM_BRANCHES
    global COMB_PARENTS
    global COMB_PARENTS_IN_EACH_LEVEL
    global COMB_BRANCHES_IN_EACH_LEVEL
    global NUM_NEIGHBOURS
    global NSEG_PER_BRANCH
    global SOLVER

    NUM_BRANCHES = [cell.num_branches for cell in cells]
    CUMSUM_NUM_BRANCHES = jnp.cumsum(jnp.asarray([0] + NUM_BRANCHES))

    parents = [cell.parents for cell in cells]
    COMB_PARENTS = jnp.concatenate(
        [p.at[1:].add(CUMSUM_NUM_BRANCHES[i]) for i, p in enumerate(parents)]
    )
    COMB_PARENTS_IN_EACH_LEVEL = merge_cells(
        CUMSUM_NUM_BRANCHES, [cell.parents_in_each_level for cell in cells]
    )
    COMB_BRANCHES_IN_EACH_LEVEL = merge_cells(
        CUMSUM_NUM_BRANCHES,
        [cell.branches_in_each_level for cell in cells],
        exclude_first=False,
    )
    NUM_NEIGHBOURS = [cell.num_neighbours for cell in cells]

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
    concat_voltage = jnp.concatenate(init_v)
    saveat = saveat.at[:, 0].set(
        concat_voltage[CUMSUM_NUM_BRANCHES[rec_cell_inds] + rec_inds]
    )

    t = 0.0
    init_state = (
        t,
        concat_voltage,
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
        cells[0].coupling_conds,
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
    coupling_conds,
):
    # Membrane input.
    voltage_terms = jnp.zeros_like(v)
    constant_terms = jnp.zeros_like(v)
    new_states = []
    for i, update_fn in enumerate(MEM_CHANNELS):
        membrane_current_terms, states = update_fn(
            v,
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
        v,
        CUMSUM_NUM_BRANCHES[i_cell_inds] + i_inds,
        i_stim,
        radius,
        length_single_compartment,
    )

    # Synaptic input.
    syn_voltage_terms = jnp.zeros_like(v)
    syn_constant_terms = jnp.zeros_like(v)
    new_syn_states = []
    for i, update_fn in enumerate(SYN_CHANNELS):
        synapse_current_terms, synapse_states = update_fn(
            v,
            ss[i],
            CUMSUM_NUM_BRANCHES[pre_syn_cell_inds[i]] + pre_syn_inds[i],
            dt,
            syn_params[i],
        )
        synapse_current_terms = postsyn_voltage_updates(
            CUMSUM_NUM_BRANCHES,
            v,
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
    lowers, diags, uppers, solves = define_all_tridiags(
        v,
        v_terms,
        c_terms,
        jnp.concatenate(NUM_NEIGHBOURS),
        NSEG_PER_BRANCH,
        sum(NUM_BRANCHES),
        dt,
        coupling_conds,
    )

    # Solve quasi-tridiagonal system.
    sol_tri = solve_branched(
        COMB_PARENTS_IN_EACH_LEVEL,
        COMB_BRANCHES_IN_EACH_LEVEL,
        COMB_PARENTS,
        lowers,
        diags,
        uppers,
        solves,
        -dt * coupling_conds,
        SOLVER,
    )
    return sol_tri.flatten(), new_states, new_syn_states


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
        coupling_conds,
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
        coupling_conds,
    )
    t += dt

    saveat = saveat.at[:, i + 1].set(v[CUMSUM_NUM_BRANCHES[rec_cell_inds] + rec_inds])

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
        coupling_conds,
        rec_cell_inds,
        rec_inds,
        pre_syn_inds,
        pre_syn_cell_inds,
        grouped_post_syn_inds,
        grouped_post_syns,
        saveat,
    )
