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


NUM_BRANCHES = -1
NSEG_PER_BRANCH = -1


def solve(cell, init, params, stimulus, t_max, dt: float = 0.025):
    """
    Solve function.
    """
    global NUM_BRANCHES
    global NSEG_PER_BRANCH

    NUM_BRANCHES = cell.num_branches
    NSEG_PER_BRANCH = cell.nseg_per_branch

    num_time_steps = int(t_max / dt)
    saveat = jnp.zeros((2, num_time_steps))

    saveat = saveat.at[0, 0].set(init[4 * (NSEG_PER_BRANCH - 1)])
    saveat = saveat.at[1, 0].set(init[0])

    t = 0.0
    init_state = (
        t,
        init,
        params,
        stimulus.i_delay,
        stimulus.i_amp,
        stimulus.i_dur,
        dt,
        cell.radius,
        cell.length_single_compartment,
        cell.num_neighbours,
        cell.coupling_conds,
        cell.branches_in_each_level,
        cell.parents,
        saveat,
    )

    final_state = lax.fori_loop(0, num_time_steps, body_fun, init_state)
    return final_state[-1]


def find_root(
    t,
    u,
    params,
    i_delay,
    i_amp,
    i_dur,
    dt,
    radius,
    length_single_compartment,
    num_neighbours,
    coupling_conds,
    branches_in_each_level,
    parents,
):

    voltages = u[::4]
    ms = u[1::4]
    hs = u[2::4]
    ns = u[3::4]

    new_m = solve_gate_exponential(ms, dt, *m_gate(voltages))
    new_h = solve_gate_exponential(hs, dt, *h_gate(voltages))
    new_n = solve_gate_exponential(ns, dt, *n_gate(voltages))

    na_conds = params[::3] * (ms**3) * hs
    kd_conds = params[1::3] * ns**4
    leak_conds = params[2::3]

    # External input
    i_ext = get_external_input(
        voltages=voltages,
        t=t,
        i_delay=i_delay,
        i_dur=i_dur,
        i_amp=i_amp,
        radius=radius,
        length_single_compartment=length_single_compartment,
        nseg_per_branch=NSEG_PER_BRANCH,
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
        0.0,  # -dt * coupling_conds,
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
        i_delay,
        i_amp,
        i_dur,
        dt,
        radius,
        length_single_compartment,
        num_neighbours,
        coupling_conds,
        branches_in_each_level,
        parents,
        saveat,
    ) = state

    u_inner = find_root(
        t,
        u_inner,
        params,
        i_delay,
        i_amp,
        i_dur,
        dt,
        radius,
        length_single_compartment,
        num_neighbours,
        coupling_conds,
        branches_in_each_level,
        parents,
    )
    t += dt

    saveat = saveat.at[0, i + 1].set(u_inner[4 * (NSEG_PER_BRANCH - 1)])
    saveat = saveat.at[1, i + 1].set(u_inner[0])

    return (
        t,
        u_inner,
        params,
        i_delay,
        i_amp,
        i_dur,
        dt,
        radius,
        length_single_compartment,
        num_neighbours,
        coupling_conds,
        branches_in_each_level,
        parents,
        saveat,
    )
