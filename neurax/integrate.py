from jax import lax
import jax.numpy as jnp
from jax import vmap
from neurax.mechanisms.hh_neuron import hh_neuron_gate
from neurax.build_branched_tridiag import define_all_tridiags
from neurax.solver_voltage import solve_branched
from neurax.stimulus import get_external_input
from neurax.utils.cell_utils import index_of_loc


NUM_BRANCHES = -1
NSEG_PER_BRANCH = -1


def solve(cells, init, params, stimuli, recordings, t_max, dt: float = 0.025):
    """
    Solve function.
    """
    global NUM_BRANCHES
    global NSEG_PER_BRANCH

    NUM_BRANCHES = cells[0].num_branches
    NSEG_PER_BRANCH = cells[0].nseg_per_branch

    num_recordings = len(recordings)
    num_time_steps = int(t_max / dt) + 1
    saveat = jnp.zeros((num_recordings, num_time_steps))

    num_states = 4
    rec_inds = [index_of_loc(r.branch_ind, r.loc, NSEG_PER_BRANCH) for r in recordings]
    rec_inds = jnp.asarray(rec_inds) * num_states
    rec_cell_inds = jnp.asarray([r.cell_ind for r in recordings])

    stim_inds = [index_of_loc(s.branch_ind, s.loc, NSEG_PER_BRANCH) for s in stimuli]
    stim_cell_inds = jnp.asarray([s.cell_ind for s in stimuli])
    stim_inds = jnp.asarray(stim_inds)
    stim_currents = jnp.asarray([s.current for s in stimuli])  # nA

    # Save voltage at the beginning.
    saveat = saveat.at[:, 0].set(init[rec_cell_inds, rec_inds])  # TODO

    t = 0.0
    init_state = (
        t,
        init,
        params,
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
        saveat,
        rec_cell_inds,
        rec_inds,
    )

    final_state = lax.fori_loop(0, num_time_steps, body_fun, init_state)
    return final_state[-3]


def find_root(
    t,
    u,
    params,
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
):
    voltages = u[:, ::4]  # mV
    ms = u[:, 1::4]
    hs = u[:, 2::4]
    ns = u[:, 3::4]

    membrane_current_terms, states = hh_neuron_gate(voltages, ms, hs, ns, dt, params)
    new_m, new_h, new_n = states
    voltage_terms, constant_terms = membrane_current_terms

    # External input
    i_ext = get_external_input(
        voltages, i_cell_inds, i_inds, i_stim, radius, length_single_compartment
    )
    lowers, diags, uppers, solves = vmap(
        define_all_tridiags, in_axes=(0, 0, 0, None, None, None, None, None)
    )(
        voltages,
        voltage_terms,
        i_ext + constant_terms,
        num_neighbours,
        NSEG_PER_BRANCH,
        NUM_BRANCHES,
        dt,
        coupling_conds,
    )
    solves = vmap(solve_branched, in_axes=(None, None, None, 0, 0, 0, 0, None))(
        parents_in_each_level,
        branches_in_each_level,
        parents,
        lowers,
        diags,
        uppers,
        solves,
        -dt * coupling_conds,
    )
    ncells = len(u)
    new_v = jnp.reshape(solves, (ncells, NUM_BRANCHES * NSEG_PER_BRANCH))

    out = jnp.concatenate((new_v, new_m, new_h, new_n), axis=1)
    return out


def body_fun(i, state):
    """
    Body for fori_loop.
    """
    (
        t,
        u_inner,
        params,
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
        saveat,
        rec_cell_inds,
        rec_inds,
    ) = state

    u_inner = find_root(
        t,
        u_inner,
        params,
        i_cell_inds,
        i_inds,
        i_stim[:, i],
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

    saveat = saveat.at[:, i + 1].set(u_inner[rec_cell_inds, rec_inds])

    return (
        t,
        u_inner,
        params,
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
        saveat,
        rec_cell_inds,
        rec_inds,
    )
