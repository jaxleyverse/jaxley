from jax import lax
import jax.numpy as jnp
from neurax.mechanisms.hh_neuron import (
    m_gate,
    h_gate,
    n_gate,
)
from neurax.build_branched_tridiag import define_all_tridiags
from neurax.solver_voltage import solve_branched
from neurax.solver_gate import solve_gate_exponential, solve_gate_implicit
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
    rec_inds = [index_of_loc(r.branch_ind, r.loc, cell.nseg_per_branch) for r in recordings]
    rec_inds = jnp.asarray(rec_inds) * num_states
    stim_inds = [index_of_loc(s.branch_ind, s.loc, cell.nseg_per_branch) for s in stimuli]
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

    new_m = solve_gate_exponential(ms, dt, *m_gate(voltages))
    new_h = solve_gate_exponential(hs, dt, *h_gate(voltages))
    new_n = solve_gate_exponential(ns, dt, *n_gate(voltages))

    # Multiply with 1000 to convert Siemens to milli Siemens.
    na_conds = params[::3] * (ms**3) * hs * 1000  # mS/cm^2
    kd_conds = params[1::3] * ns**4 * 1000 # mS/cm^2
    leak_conds = params[2::3] * 1000 # mS/cm^2

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
        na_conds,
        kd_conds,
        leak_conds=leak_conds,
        i_ext=i_ext,
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
