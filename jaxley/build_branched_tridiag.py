# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from math import pi

import jax.numpy as jnp
from jax import lax, vmap


def define_all_tridiags(
    voltages: jnp.ndarray,
    voltage_terms: jnp.asarray,
    constant_terms: jnp.ndarray,
    coupling_conds_upper: float,
    coupling_conds_lower: float,
    summed_coupling_conds: float,
    nbranches: int,
    nseg: int,
    dt: float,
):
    """
    Set up tridiagonal system for each branch.
    """
    diags = jnp.ones((nbranches * nseg))
    diags = diags.at[branch_ind * nseg + comp_ind].add(voltage_terms * dt)


    diags = 1.0 + dt * voltage_terms + dt * summed_coupling_conds
    lowers, diags, uppers, solves = vmap(
        _define_tridiag_for_branch, in_axes=(0, 0, 0, None, 0, 0, 0)
    )(
        voltages,
        voltage_terms,
        constant_terms,
        dt,
        coupling_conds_upper,
        coupling_conds_lower,
        summed_coupling_conds,
    )

    return (lowers, diags, uppers, solves)


def _define_tridiag_for_branch(
    voltages: jnp.ndarray,
    voltage_terms: jnp.ndarray,
    i_ext: jnp.ndarray,
    dt: float,
    coupling_conds_upper: float,
    coupling_conds_lower: float,
    summed_coupling_conds: float,
):
    """
    Defines the tridiagonal system to solve for a single branch.
    """

    # Diagonal and solve.
    diags = 1.0 + dt * voltage_terms + dt * summed_coupling_conds
    solves = voltages + dt * i_ext

    # Subdiagonals.
    upper = jnp.asarray(-dt * coupling_conds_upper)
    lower = jnp.asarray(-dt * coupling_conds_lower)
    return lower, diags, upper, solves
