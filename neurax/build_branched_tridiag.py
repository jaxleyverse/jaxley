from math import pi
import jax.numpy as jnp
from jax import lax, vmap


def define_all_tridiags(
    voltages: jnp.ndarray,
    na_conds: jnp.ndarray,
    kd_conds: jnp.ndarray,
    leak_conds: jnp.ndarray,
    i_ext: jnp.ndarray,
    num_neighbours: jnp.ndarray,
    nseg_per_branch: int,
    num_branches: int,
    dt: float,
    coupling_conds: float,
):
    """
    Set up tridiagonal system for each branch.
    """
    lowers, diags, uppers, solves = [], [], [], []

    voltages = jnp.reshape(voltages, (num_branches, -1))
    na_conds = jnp.reshape(na_conds, (num_branches, -1))
    kd_conds = jnp.reshape(kd_conds, (num_branches, -1))
    leak_conds = jnp.reshape(leak_conds, (num_branches, -1))
    i_ext = jnp.reshape(i_ext, (num_branches, -1))
    num_neighbours = jnp.reshape(num_neighbours, (num_branches, -1))

    lowers, diags, uppers, solves = vmap(
        _define_tridiag_for_branch, in_axes=(0, 0, 0, 0, 0, None, 0, None, None)
    )(
        voltages,
        na_conds,
        kd_conds,
        leak_conds,
        i_ext,
        dt,
        num_neighbours,
        coupling_conds,
        nseg_per_branch,
    )

    return (lowers, diags, uppers, solves)


def _define_tridiag_for_branch(
    voltages: jnp.ndarray,
    na_conds: jnp.ndarray,
    kd_conds: jnp.ndarray,
    leak_conds: jnp.ndarray,
    i_ext: jnp.ndarray,
    dt: float,
    num_neighbours: jnp.ndarray,
    coupling_conds: float,
    nseg_per_branch: int,
):
    """
    Defines the tridiagonal system to solve for a single branch.
    """
    voltage_terms = na_conds + kd_conds + leak_conds
    constant_terms = 50.0 * na_conds + (-77.0) * kd_conds + (-54.3) * leak_conds + i_ext

    # Diagonal and solve.
    a_v = 1.0 + dt * voltage_terms + dt * num_neighbours * coupling_conds
    b_v = voltages + dt * constant_terms

    # Subdiagonals.
    upper = jnp.asarray([-dt * coupling_conds] * (nseg_per_branch - 1))
    lower = jnp.asarray([-dt * coupling_conds] * (nseg_per_branch - 1))
    return lower, a_v, upper, b_v
