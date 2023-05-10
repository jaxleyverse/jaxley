from typing import List
from math import prod

import jax
from jax import lax, vmap
import jax.numpy as jnp

from neurax.build_branched_tridiag import define_all_tridiags
from neurax.solver_voltage import solve_branched
from neurax.stimulus import get_external_input
from neurax.utils.cell_utils import index_of_loc
from neurax.utils.syn_utils import postsyn_voltage_updates
from neurax.utils.jax_utils import nested_checkpoint_scan

NUM_BRANCHES = []
CUMSUM_NUM_BRANCHES = []
COMB_CUM_KID_INDS_IN_EACH_LEVEL = []
MAX_NUM_KIDS = None

COMB_PARENTS = []
COMB_PARENTS_IN_EACH_LEVEL = []
COMB_BRANCHES_IN_EACH_LEVEL = []
RADIUSES = []
LENGTHS = []
COUPLING_CONDS_FWD = []
COUPLING_CONDS_BWD = []
BRANCH_CONDS_FWD = []
BRANCH_CONDS_BWD = []
SUMMED_COUPLING_CONDS = []

I_CELL_INDS = []
I_INDS = []
REC_INDS = []

NSEG_PER_BRANCH = -1
SOLVER = ""
MEM_CHANNELS = []
SYN_CHANNELS = []
DELTA_T = 0.0

PRE_SYN_INDS = []
PRE_SYN_CELL_INDS = []
GROUPED_POST_SYN_INDS = []
GROUPED_POST_SYNS = []


def solve(
    network,
    init_v,
    mem_states,
    mem_params,
    mem_channels,
    syn_states,
    syn_params,
    syn_channels,
    stimuli,
    recordings,
    dt: float = 0.025,
    solver: str = "stone",
    checkpoint_lengths: List[int] = [],
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
        network,
        init_v,
        mem_states,
        mem_params,
        syn_states,
        syn_params,
        stimuli,
        recordings,
        dt,
        solver,
    )

    i_ext = jnp.asarray([s.current for s in stimuli]).T  # nA
    nsteps_to_return = len(i_ext)
    init_recording = jnp.expand_dims(state[0][REC_INDS], axis=0)

    # If necessary, pad the stimulus with zeros in order to simulate sufficiently long.
    # The total simulation length will be `prod(checkpoint_lengths)`. At the end, we
    # return only the first `nsteps_to_return` elements (plus the initial state).
    if checkpoint_lengths is None:
        checkpoint_lengths = [len(i_ext)]
        length = len(i_ext)
    else:
        length = prod(checkpoint_lengths)
        assert (
            len(i_ext) <= length
        ), "The external current is longer than `prod(nested_length)`."
        size_difference = length - len(i_ext)
        dummy_stimulus = jnp.zeros((size_difference, i_ext.shape[1]))
        i_ext = jnp.concatenate([i_ext, dummy_stimulus])

    _, recordings = nested_checkpoint_scan(
        body_fun, state, i_ext, length=length, nested_lengths=checkpoint_lengths
    )
    return jnp.concatenate([init_recording, recordings[:nsteps_to_return]], axis=0).T


def prepare_state(
    network,
    init_v,
    mem_states,
    mem_params,
    syn_states,
    syn_params,
    stimuli,
    recordings,
    dt: float = 0.025,
    solver: str = "stone",
):
    global NUM_BRANCHES
    global CUMSUM_NUM_BRANCHES
    global COMB_CUM_KID_INDS_IN_EACH_LEVEL
    global MAX_NUM_KIDS

    global COMB_PARENTS
    global COMB_PARENTS_IN_EACH_LEVEL
    global COMB_BRANCHES_IN_EACH_LEVEL
    global RADIUSES
    global LENGTHS
    global COUPLING_CONDS_FWD
    global COUPLING_CONDS_BWD
    global BRANCH_CONDS_FWD
    global BRANCH_CONDS_BWD
    global SUMMED_COUPLING_CONDS

    global NSEG_PER_BRANCH
    global SOLVER
    global DELTA_T
    global I_CELL_INDS
    global I_INDS

    global PRE_SYN_INDS
    global PRE_SYN_CELL_INDS
    global GROUPED_POST_SYN_INDS
    global GROUPED_POST_SYNS
    global REC_INDS

    # Define everything related to morphology as global variables.
    NUM_BRANCHES = network.num_branches
    CUMSUM_NUM_BRANCHES = network.cumsum_num_branches
    MAX_NUM_KIDS = network.max_num_kids
    COMB_PARENTS = network.comb_parents
    COMB_PARENTS_IN_EACH_LEVEL = network.comb_parents_in_each_level
    COMB_BRANCHES_IN_EACH_LEVEL = network.comb_branches_in_each_level
    COMB_CUM_KID_INDS_IN_EACH_LEVEL = network.comb_cum_kid_inds_in_each_level
    RADIUSES = network.radiuses
    LENGTHS = network.lengths
    COUPLING_CONDS_FWD = network.coupling_conds_fwd
    COUPLING_CONDS_BWD = network.coupling_conds_bwd
    BRANCH_CONDS_FWD = network.branch_conds_fwd
    BRANCH_CONDS_BWD = network.branch_conds_bwd
    SUMMED_COUPLING_CONDS = network.summed_coupling_conds
    NSEG_PER_BRANCH = network.nseg_per_branch

    # Define morphology of synapses.
    PRE_SYN_INDS = [c.pre_syn_inds for c in network.connectivities]
    PRE_SYN_CELL_INDS = [c.pre_syn_cell_inds for c in network.connectivities]
    GROUPED_POST_SYN_INDS = [c.grouped_post_syn_inds for c in network.connectivities]
    GROUPED_POST_SYNS = [c.grouped_post_syns for c in network.connectivities]

    # Define the solver.
    SOLVER = solver
    DELTA_T = dt

    # TODO: do I actually need this conversion if I assume NSEG_PER_BRANCH to be const?
    # Can I not just keep a 2D array everywhere?
    rec_inds = [index_of_loc(r.branch_ind, r.loc, NSEG_PER_BRANCH) for r in recordings]
    rec_inds = jnp.asarray(rec_inds)
    rec_cell_inds = jnp.asarray([r.cell_ind for r in recordings])
    REC_INDS = NSEG_PER_BRANCH * CUMSUM_NUM_BRANCHES[rec_cell_inds] + rec_inds

    stim_inds = [index_of_loc(s.branch_ind, s.loc, NSEG_PER_BRANCH) for s in stimuli]
    I_CELL_INDS = jnp.asarray([s.cell_ind for s in stimuli])
    I_INDS = jnp.asarray(stim_inds)

    concat_voltage = jnp.concatenate(init_v)

    init_state = (
        concat_voltage,
        [jnp.concatenate(m, axis=1) for m in mem_states],
        syn_states,
        [jnp.concatenate(m, axis=1) for m in mem_params],
        syn_params,
    )
    return init_state


def find_root(
    v,
    u,
    ss,
    params,
    syn_params,
    i_stim,
):
    voltages = v  # mV

    # Membrane input.
    voltage_terms = jnp.zeros_like(v)
    constant_terms = jnp.zeros_like(v)
    new_states = []
    for i, update_fn in enumerate(MEM_CHANNELS):
        membrane_current_terms, states = update_fn(voltages, u[i], params[i], DELTA_T)
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
            CUMSUM_NUM_BRANCHES[PRE_SYN_CELL_INDS[i]] * NSEG_PER_BRANCH
            + PRE_SYN_INDS[i],
            DELTA_T,
            syn_params[i],
        )
        synapse_current_terms = postsyn_voltage_updates(
            NSEG_PER_BRANCH,
            CUMSUM_NUM_BRANCHES,
            voltages,
            GROUPED_POST_SYN_INDS[i],
            GROUPED_POST_SYNS[i],
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
        sum(NUM_BRANCHES),
        COUPLING_CONDS_BWD,
        COUPLING_CONDS_FWD,
        SUMMED_COUPLING_CONDS,
        DELTA_T,
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
        -DELTA_T * BRANCH_CONDS_BWD,
        -DELTA_T * BRANCH_CONDS_FWD,
        COMB_CUM_KID_INDS_IN_EACH_LEVEL,
        MAX_NUM_KIDS,
        SOLVER,
    )
    return sol_tri.flatten(order="C"), new_states, new_syn_states


def body_fun(state, i_stim):
    """
    Body for fori_loop.
    """
    (
        v,
        u_inner,
        syn_states,
        params,
        syn_params,
    ) = state

    v, u_inner, syn_states = find_root(
        v,
        u_inner,
        syn_states,
        params,
        syn_params,
        i_stim,
    )

    return (
        v,
        u_inner,
        syn_states,
        params,
        syn_params,
    ), v[REC_INDS]
