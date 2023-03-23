from jax import lax
import jax.numpy as jnp
from neurax.mechanisms.hh_neuron import hh_neuron_gate
from neurax.build_branched_tridiag import define_all_tridiags
from neurax.solver_voltage import solve_branched
from neurax.stimulus import get_external_input
from neurax.utils.cell_utils import index_of_loc


NUM_BRANCHES = -1
NSEG_PER_BRANCH = -1


def solve(cell, init, params, stimuli, recordings, t_max, dt: float = 0.025):
    """
    Solve function.
    """
    global NUM_BRANCHES
    global NSEG_PER_BRANCH

    NUM_BRANCHES = cell.num_branches
    NSEG_PER_BRANCH = cell.nseg_per_branch

    num_recordings = len(recordings)
    num_time_steps = int(t_max / dt) + 1
    saveat = jnp.zeros((num_recordings, num_time_steps))

    num_states = 4
    rec_inds = [
        index_of_loc(r.branch_ind, r.loc, cell.nseg_per_branch) for r in recordings
    ]
    rec_inds = jnp.asarray(rec_inds) * num_states
    stim_inds = [
        index_of_loc(s.branch_ind, s.loc, cell.nseg_per_branch) for s in stimuli
    ]
    stim_inds = jnp.asarray(stim_inds)
    stim_currents = jnp.asarray([s.current for s in stimuli])  # nA

    # Save voltage at the beginning.
    saveat = saveat.at[:, 0].set(init[rec_inds])

    t = 0.0
    init_state = (
        t,
        init,
        params,
        stim_inds,
        stim_currents,
        dt,
        cell.radius,
        cell.length_single_compartment,
        cell.num_neighbours,
        cell.coupling_conds,
        cell.branches_in_each_level,
        cell.parents,
        saveat,
        rec_inds,
    )

    final_state = lax.fori_loop(0, num_time_steps, body_fun, init_state)
    return final_state[-2]


def find_root(
    t,
    u,
    params,
    i_inds,
    i_stim,
    dt,
    radius,
    length_single_compartment,
    num_neighbours,
    coupling_conds,
    branches_in_each_level,
    parents,
):
    voltages = u[::4]  # mV
    ms = u[1::4]
    hs = u[2::4]
    ns = u[3::4]

    membrane_current_terms, states = hh_neuron_gate(voltages, (ms, hs, ns), dt, params)
    new_m, new_h, new_n = states
    voltage_terms, constant_terms = membrane_current_terms

    # External input
    i_ext = get_external_input(
        voltages=voltages,
        i_inds=i_inds,
        i_stim=i_stim,
        radius=radius,
        length_single_compartment=length_single_compartment,
    )
    lowers, diags, uppers, solves = define_all_tridiags(
        voltages,
        voltage_terms,
        i_ext=i_ext + constant_terms,
        num_neighbours=num_neighbours,
        nseg_per_branch=NSEG_PER_BRANCH,
        num_branches=NUM_BRANCHES,
        dt=dt,
        coupling_conds=coupling_conds,
    )
    solves = solve_branched(
        branches_in_each_level,
        parents,
        lowers,
        diags,
        uppers,
        solves,
        -dt * coupling_conds,
    )
    new_v = jnp.concatenate(solves)

    return jnp.ravel(jnp.column_stack((new_v, new_m, new_h, new_n)))


def body_fun(i, state):
    """
    Body for fori_loop.
    """
    (
        t,
        u_inner,
        params,
        i_inds,
        i_stim,
        dt,
        radius,
        length_single_compartment,
        num_neighbours,
        coupling_conds,
        branches_in_each_level,
        parents,
        saveat,
        rec_inds,
    ) = state

    u_inner = find_root(
        t,
        u_inner,
        params,
        i_inds,
        i_stim[:, i],
        dt,
        radius,
        length_single_compartment,
        num_neighbours,
        coupling_conds,
        branches_in_each_level,
        parents,
    )
    t += dt

    saveat = saveat.at[:, i + 1].set(u_inner[rec_inds])

    return (
        t,
        u_inner,
        params,
        i_inds,
        i_stim,
        dt,
        radius,
        length_single_compartment,
        num_neighbours,
        coupling_conds,
        branches_in_each_level,
        parents,
        saveat,
        rec_inds,
    )
