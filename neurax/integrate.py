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
RADIUSES = []
LENGTHS = []

I_CELL_INDS = []
I_INDS = []

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
    global RADIUSES
    global LENGTHS
    global NSEG_PER_BRANCH
    global SOLVER
    global I_CELL_INDS
    global I_INDS

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
    # Flatten because we flatten all vars.
    RADIUSES = jnp.concatenate([c.radiuses.flatten() for c in cells])
    LENGTHS = jnp.concatenate([c.lengths.flatten() for c in cells])
    NSEG_PER_BRANCH = cells[0].nseg_per_branch
    SOLVER = solver

    for cell in cells:
        assert (
            cell.nseg_per_branch == NSEG_PER_BRANCH
        ), "Different nseg_per_branch between cells."

    num_recordings = len(recordings)
    num_time_steps = int(t_max / dt) + 1
    saveat = jnp.zeros((num_recordings, num_time_steps))

    # TODO: do I actually need this conversion if I assume NSEG_PER_BRANCH to be const?
    # Can I not just keep a 2D array everywhere?
    rec_inds = [index_of_loc(r.branch_ind, r.loc, NSEG_PER_BRANCH) for r in recordings]
    rec_inds = jnp.asarray(rec_inds)
    rec_cell_inds = jnp.asarray([r.cell_ind for r in recordings])

    stim_inds = [index_of_loc(s.branch_ind, s.loc, NSEG_PER_BRANCH) for s in stimuli]
    stim_currents = jnp.asarray([s.current for s in stimuli])  # nA
    I_CELL_INDS = jnp.asarray([s.cell_ind for s in stimuli])
    I_INDS = jnp.asarray(stim_inds)

    concat_voltage = jnp.concatenate(init_v)
    # Save voltage at the beginning.
    saveat = saveat.at[:, 0].set(
        concat_voltage[NSEG_PER_BRANCH * CUMSUM_NUM_BRANCHES[rec_cell_inds] + rec_inds]
    )

    t = 0.0
    init_state = (
        t,
        concat_voltage,
        [jnp.concatenate(m, axis=1) for m in mem_states],
        syn_states,
        [jnp.concatenate(m, axis=1) for m in mem_params],
        syn_params,
        stim_currents,
        dt,
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
    i_stim,
    pre_syn_inds,
    pre_syn_cell_inds,
    grouped_post_syn_inds,
    grouped_post_syns,
    dt,
    coupling_conds,
):
    voltages = v  # mV

    # Membrane input.
    voltage_terms = jnp.zeros_like(v)
    constant_terms = jnp.zeros_like(v)
    new_states = []
    for i, update_fn in enumerate(MEM_CHANNELS):
        membrane_current_terms, states = update_fn(voltages, u[i], params[i], dt)
        voltage_terms += membrane_current_terms[0]
        constant_terms += membrane_current_terms[1]
        new_states.append(states)

    # External input.
    i_ext = get_external_input(
        voltages,
        CUMSUM_NUM_BRANCHES[I_CELL_INDS] * NSEG_PER_BRANCH + I_INDS,
        i_stim,
        RADIUSES,
        LENGTHS,
    )

    # Synaptic input.
    syn_voltage_terms = jnp.zeros_like(voltages)
    syn_constant_terms = jnp.zeros_like(voltages)
    new_syn_states = []
    for i, update_fn in enumerate(SYN_CHANNELS):
        synapse_current_terms, synapse_states = update_fn(
            voltages,
            ss[i],
            CUMSUM_NUM_BRANCHES[pre_syn_cell_inds[i]] * NSEG_PER_BRANCH
            + pre_syn_inds[i],
            dt,
            syn_params[i],
        )
        synapse_current_terms = postsyn_voltage_updates(
            NSEG_PER_BRANCH,
            CUMSUM_NUM_BRANCHES,
            voltages,
            grouped_post_syn_inds[i],
            grouped_post_syns[i],
            *synapse_current_terms
        )
        syn_voltage_terms += synapse_current_terms[0]
        syn_constant_terms += synapse_current_terms[1]
        new_syn_states.append(synapse_states)

    # Define quasi-tridiagonal system.
    lowers, diags, uppers, solves = define_all_tridiags(
        voltages,
        voltage_terms + syn_voltage_terms,
        i_ext + constant_terms + syn_constant_terms,
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
    return sol_tri.flatten(order="C"), new_states, new_syn_states


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
        i_stim,
        dt,
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
        i_stim[:, i],
        pre_syn_inds,
        pre_syn_cell_inds,
        grouped_post_syn_inds,
        grouped_post_syns,
        dt,
        coupling_conds,
    )
    t += dt

    saveat = saveat.at[:, i + 1].set(
        v[NSEG_PER_BRANCH * CUMSUM_NUM_BRANCHES[rec_cell_inds] + rec_inds]
    )

    return (
        t,
        v,
        u_inner,
        syn_states,
        params,
        syn_params,
        i_stim,
        dt,
        coupling_conds,
        rec_cell_inds,
        rec_inds,
        pre_syn_inds,
        pre_syn_cell_inds,
        grouped_post_syn_inds,
        grouped_post_syns,
        saveat,
    )
