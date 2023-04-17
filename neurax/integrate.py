from typing import List

import jax
from jax import lax, vmap
import jax.numpy as jnp

from neurax.build_branched_tridiag import define_all_tridiags
from neurax.solver_voltage import solve_branched
from neurax.stimulus import get_external_input
from neurax.utils.cell_utils import index_of_loc
from neurax.utils.syn_utils import postsyn_voltage_updates


NUM_BRANCHES = -1
CUMSUM_NUM_BRANCHES = []
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
    global NSEG_PER_BRANCH
    global SOLVER

    NUM_BRANCHES = [c.num_branches for c in cells]
    CUMSUM_NUM_BRANCHES = jnp.cumsum(jnp.asarray([0] + NUM_BRANCHES))
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
    saveat = saveat.at[:, 0].set(init_v[rec_cell_inds, rec_inds])  # 0 = voltage

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
        cells[0].parents,
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
    parents,
):
    voltages = v  # mV

    # Membrane input.
    voltage_terms = jnp.zeros_like(jnp.concatenate(v))
    constant_terms = jnp.zeros_like(jnp.concatenate(v))
    new_states = []
    for i, update_fn in enumerate(MEM_CHANNELS):
        membrane_current_terms, states = update_fn(
            jnp.concatenate(voltages),
            jnp.concatenate(u[i], axis=1),
            dt,
            jnp.concatenate(params[i], axis=1),
        )
        voltage_terms += membrane_current_terms[0]
        constant_terms += membrane_current_terms[1]
        new_states.append(jnp.stack([states[:, :60], states[:, 60:]]))

    voltage_terms = jnp.reshape(voltage_terms, (2, 60))
    constant_terms = jnp.reshape(constant_terms, (2, 60))

    # External input.
    i_ext = get_external_input(
        jnp.concatenate(voltages),
        CUMSUM_NUM_BRANCHES[i_cell_inds] * NSEG_PER_BRANCH + i_inds,
        i_stim,
        radius,
        length_single_compartment,
    )
    i_ext = jnp.reshape(i_ext, (2, 60))

    # Synaptic input.
    syn_voltage_terms = jnp.zeros_like(jnp.concatenate(voltages))
    syn_constant_terms = jnp.zeros_like(jnp.concatenate(voltages))
    new_syn_states = []
    for i, update_fn in enumerate(SYN_CHANNELS):
        synapse_current_terms, synapse_states = update_fn(
            jnp.concatenate(voltages),
            ss[i],
            CUMSUM_NUM_BRANCHES[pre_syn_cell_inds[i]] * NSEG_PER_BRANCH
            + pre_syn_inds[i],
            dt,
            syn_params[i],
        )
        synapse_current_terms = postsyn_voltage_updates(
            NSEG_PER_BRANCH,
            CUMSUM_NUM_BRANCHES,
            jnp.concatenate(voltages),
            grouped_post_syn_inds[i],
            grouped_post_syns[i],
            *synapse_current_terms
        )
        syn_voltage_terms += synapse_current_terms[0]
        syn_constant_terms += synapse_current_terms[1]
        new_syn_states.append(synapse_states)

    syn_voltage_terms = jnp.reshape(syn_voltage_terms, (2, 60))
    syn_constant_terms = jnp.reshape(syn_constant_terms, (2, 60))

    # Define quasi-tridiagonal system.
    lowers, diags, uppers, solves = vmap(
        define_all_tridiags, in_axes=(0, 0, 0, None, None, None, None, None)
    )(
        voltages,
        voltage_terms + syn_voltage_terms,
        i_ext + constant_terms + syn_constant_terms,
        num_neighbours,
        NSEG_PER_BRANCH,
        NUM_BRANCHES[0],
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
    new_v = jnp.reshape(solves, (ncells, NUM_BRANCHES[0] * NSEG_PER_BRANCH))
    return new_v, new_states, new_syn_states


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
