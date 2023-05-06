from typing import List

import jax
from jax import lax, vmap
import jax.numpy as jnp

from neurax.build_branched_tridiag import define_all_tridiags

# from neurax.solver_voltage import solve_branched
from neurax.stimulus import get_external_input
from neurax.utils.cell_utils import index_of_loc
from neurax.utils.syn_utils import postsyn_voltage_updates

from tridiax.thomas import thomas_triang, thomas_backsub
from tridiax.stone import stone_triang, stone_backsub

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
        network,
        init_v,
        mem_states,
        mem_params,
        syn_states,
        syn_params,
        stimuli,
        recordings,
        t_max,
        dt,
        solver,
    )
    num_time_steps = int(t_max / dt) + 1

    # Initialize arrays for checkpoints.
    # assert 0 not in checkpoint_inds, "Checkpoints at index 0 is not implemented."
    # checkpoint_inds = [0] + checkpoint_inds  # Add 0 index for convenience later.

    current_state = state

    def inner_loop(current_state, cp_ind1, cp_ind2):
        current_state = lax.fori_loop(
            cp_ind1,
            cp_ind2,
            body_fun,
            current_state,
        )
        return current_state

    # for cp_ind in range(len(checkpoint_inds) - 1):
    #     current_state = jax.checkpoint(inner_loop, static_argnums=(1, 2))(
    #         current_state,
    #         checkpoint_inds[cp_ind],
    #         checkpoint_inds[cp_ind + 1],
    #     )
    # # The trace from the last checkpoint until the end should not be checkpointed to
    # # avoid useless recomputation.
    current_state = inner_loop(current_state, 0, num_time_steps)

    return current_state[-1]


def prepare_state(
    network,
    init_v,
    mem_states,
    mem_params,
    syn_states,
    syn_params,
    stimuli,
    recordings,
    t_max,
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

    init_state = (
        concat_voltage,
        [jnp.concatenate(m, axis=1) for m in mem_states],
        syn_states,
        [jnp.concatenate(m, axis=1) for m in mem_params],
        syn_params,
        stim_currents,
        rec_cell_inds,
        rec_inds,
        saveat,
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
        lowers,
        diags,
        uppers,
        solves,
    )
    return sol_tri.flatten(order="C"), new_states, new_syn_states


def body_fun(i, state):
    """
    Body for fori_loop.
    """
    (
        v,
        u_inner,
        syn_states,
        params,
        syn_params,
        i_stim,
        rec_cell_inds,
        rec_inds,
        saveat,
    ) = state

    v, u_inner, syn_states = find_root(
        v,
        u_inner,
        syn_states,
        params,
        syn_params,
        i_stim[:, i],
    )

    saveat = saveat.at[:, i + 1].set(
        v[NSEG_PER_BRANCH * CUMSUM_NUM_BRANCHES[rec_cell_inds] + rec_inds]
    )

    return (
        v,
        u_inner,
        syn_states,
        params,
        syn_params,
        i_stim,
        rec_cell_inds,
        rec_inds,
        saveat,
    )


def solve_branched(
    lowers,
    diags,
    uppers,
    solves,
):
    """
    Solve branched.
    """

    diags, uppers, solves = triang_branched(
        lowers,
        diags,
        uppers,
        solves,
    )
    solves = backsub_branched(
        diags,
        uppers,
        solves,
    )
    return solves


def triang_branched(
    lowers,
    diags,
    uppers,
    solves,
):
    """
    Triang.
    """

    for bil, parents_in_level, kids_in_level in zip(
        reversed(COMB_BRANCHES_IN_EACH_LEVEL[1:]),
        reversed(COMB_PARENTS_IN_EACH_LEVEL[1:]),
        reversed(COMB_CUM_KID_INDS_IN_EACH_LEVEL[1:]),
    ):
        diags, uppers, solves = _triang_level(bil, lowers, diags, uppers, solves)
        diags, solves = _eliminate_parents_upper(
            parents_in_level,
            bil,
            diags,
            solves,
            kids_in_level,
        )
    # At last level, we do not want to eliminate anymore.
    diags, uppers, solves = _triang_level(
        COMB_BRANCHES_IN_EACH_LEVEL[0], lowers, diags, uppers, solves
    )

    return diags, uppers, solves


def backsub_branched(
    diags,
    uppers,
    solves,
):
    """
    Backsub.
    """
    # At first level, we do not want to eliminate.
    solves = _backsub_level(
        COMB_BRANCHES_IN_EACH_LEVEL[0], diags, uppers, solves, SOLVER
    )
    for bil in COMB_BRANCHES_IN_EACH_LEVEL[1:]:
        solves = _eliminate_children_lower(
            bil,
            COMB_PARENTS,
            solves,
            -DELTA_T * BRANCH_CONDS_FWD,
        )
        solves = _backsub_level(bil, diags, uppers, solves, SOLVER)
    return solves


def _triang_level(branches_in_level, lowers, diags, uppers, solves):
    bil = branches_in_level
    if SOLVER == "stone":
        triang_fn = stone_triang
    elif SOLVER == "thomas":
        triang_fn = thomas_triang
    else:
        raise NameError
    new_diags, new_uppers, new_solves = vmap(triang_fn, in_axes=(0, 0, 0, 0))(
        lowers[bil], diags[bil], uppers[bil], solves[bil]
    )
    diags = diags.at[bil].set(new_diags)
    uppers = uppers.at[bil].set(new_uppers)
    solves = solves.at[bil].set(new_solves)

    return diags, uppers, solves


def _backsub_level(branches_in_level, diags, uppers, solves, solver):
    bil = branches_in_level
    if solver == "stone":
        backsub_fn = stone_backsub
    elif solver == "thomas":
        backsub_fn = thomas_backsub
    else:
        raise NameError
    solves = solves.at[bil].set(
        vmap(backsub_fn, in_axes=(0, 0, 0))(solves[bil], uppers[bil], diags[bil])
    )
    return solves


def _eliminate_single_parent_upper(
    diag_at_branch, solve_at_branch, branch_cond_fwd, branch_cond_bwd
):
    last_of_3 = diag_at_branch
    last_of_3_solve = solve_at_branch

    multiplying_factor = branch_cond_fwd / last_of_3

    update_diag = -multiplying_factor * branch_cond_bwd
    update_solve = -multiplying_factor * last_of_3_solve
    return update_diag, update_solve


def _eliminate_parents_upper(
    parents_in_level,
    branches_in_level,
    diags,
    solves,
    kid_inds_in_each_level,
):
    bil = branches_in_level
    new_diag, new_solve = vmap(_eliminate_single_parent_upper, in_axes=(0, 0, 0, 0))(
        diags[bil, -1],
        solves[bil, -1],
        -DELTA_T * BRANCH_CONDS_BWD[bil],
        -DELTA_T * BRANCH_CONDS_FWD[bil],
    )
    parallel_elim = True
    if parallel_elim:
        update_diags = jnp.zeros((MAX_NUM_KIDS * len(parents_in_level)))
        update_solves = jnp.zeros((MAX_NUM_KIDS * len(parents_in_level)))
        update_diags = update_diags.at[kid_inds_in_each_level].set(new_diag)
        update_solves = update_solves.at[kid_inds_in_each_level].set(new_solve)
        diags = diags.at[parents_in_level, 0].set(
            diags[parents_in_level, 0]
            + jnp.sum(jnp.reshape(update_diags, (-1, MAX_NUM_KIDS)), axis=1)
        )
        solves = solves.at[parents_in_level, 0].set(
            solves[parents_in_level, 0]
            + jnp.sum(jnp.reshape(update_solves, (-1, MAX_NUM_KIDS)), axis=1)
        )
        return diags, solves
    else:
        result = lax.fori_loop(
            0,
            len(bil),
            _body_fun_eliminate_parents_upper,
            (diags, solves, bil, new_diag, new_solve),
        )
        return result[0], result[1]


def _body_fun_eliminate_parents_upper(i, vals):
    diags, solves, bil, new_diag, new_solve = vals
    diags = diags.at[COMB_PARENTS[bil[i]], 0].set(
        diags[COMB_PARENTS[bil[i]], 0] + new_diag[i]
    )
    solves = solves.at[COMB_PARENTS[bil[i]], 0].set(
        solves[COMB_PARENTS[bil[i]], 0] + new_solve[i]
    )
    return (diags, solves, bil, new_diag, new_solve)


def _eliminate_children_lower(
    branches_in_level,
    parents,
    solves,
    branch_cond,
):
    bil = branches_in_level
    solves = solves.at[bil, -1].set(
        solves[bil, -1] - branch_cond[bil] * solves[parents[bil], 0]
    )
    return solves
