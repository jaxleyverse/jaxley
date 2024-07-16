# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax.numpy as jnp
from jax import vmap
from jax.lax import ScatterDimensionNumbers, scatter_add
from tridiax.stone import stone_backsub, stone_triang
from tridiax.thomas import thomas_backsub, thomas_triang

from jaxley.build_branched_tridiag import define_all_tridiags


def step_voltage_explicit(
    voltages,
    voltage_terms,
    constant_terms,
    coupling_conds_bwd,
    coupling_conds_fwd,
    branch_cond_fwd,
    branch_cond_bwd,
    nbranches,
    parents,
    delta_t,
):
    """Solve one timestep of branched nerve equations with explicit (forward) Euler."""
    voltages = jnp.reshape(voltages, (nbranches, -1))
    voltage_terms = jnp.reshape(voltage_terms, (nbranches, -1))
    constant_terms = jnp.reshape(constant_terms, (nbranches, -1))

    update = voltage_vectorfield(
        parents,
        voltages,
        voltage_terms,
        constant_terms,
        coupling_conds_bwd,
        coupling_conds_fwd,
        branch_cond_fwd,
        branch_cond_bwd,
    )
    new_voltates = voltages + delta_t * update
    return new_voltates


def step_voltage_implicit(
    voltages,
    voltage_terms,
    constant_terms,
    coupling_conds_bwd,
    coupling_conds_fwd,
    summed_coupling_conds,
    branch_cond_fwd,
    branch_cond_bwd,
    nbranches,
    parents,
    branches_in_each_level,
    tridiag_solver,
    delta_t,
):
    """Solve one timestep of branched nerve equations with implicit (backward) Euler."""
    voltages = jnp.reshape(voltages, (nbranches, -1))
    voltage_terms = jnp.reshape(voltage_terms, (nbranches, -1))
    constant_terms = jnp.reshape(constant_terms, (nbranches, -1))
    coupling_conds_bwd = jnp.reshape(coupling_conds_bwd, (nbranches, -1))
    coupling_conds_fwd = jnp.reshape(coupling_conds_fwd, (nbranches, -1))
    summed_coupling_conds = jnp.reshape(summed_coupling_conds, (nbranches, -1))

    # Define quasi-tridiagonal system.
    lowers, diags, uppers, solves = define_all_tridiags(
        voltages,
        voltage_terms,
        constant_terms,
        nbranches,
        coupling_conds_bwd,
        coupling_conds_fwd,
        summed_coupling_conds,
        delta_t,
    )

    # Solve quasi-tridiagonal system.
    diags, uppers, solves = _triang_branched(
        parents,
        branches_in_each_level,
        lowers,
        diags,
        uppers,
        solves,
        -delta_t * branch_cond_fwd,
        -delta_t * branch_cond_bwd,
        tridiag_solver,
    )
    solves = _backsub_branched(
        branches_in_each_level,
        parents,
        diags,
        uppers,
        solves,
        -delta_t * branch_cond_bwd,
        tridiag_solver,
    )
    return solves


def voltage_vectorfield(
    parents,
    voltages,
    voltage_terms,
    constant_terms,
    coupling_conds_bwd,
    coupling_conds_fwd,
    branch_cond_fwd,
    branch_cond_bwd,
):
    """Evaluate the vectorfield of the nerve equation."""
    # Membrane current update.
    vecfield = -voltage_terms * voltages + constant_terms

    # Current through segments within the same branch.
    vecfield = vecfield.at[:, :-1].add(
        (voltages[:, 1:] - voltages[:, :-1]) * coupling_conds_bwd
    )
    vecfield = vecfield.at[:, 1:].add(
        (voltages[:, :-1] - voltages[:, 1:]) * coupling_conds_fwd
    )

    # Current through branch points.
    if len(branch_cond_bwd) > 0:
        vecfield = vecfield.at[:, -1].add(
            (voltages[parents, 0] - voltages[:, -1]) * branch_cond_bwd
        )

        # Several branches might have the same parent, so we have to either update these
        # entries sequentially or we have to build a matrix with width being the maximum
        # number of children and then sum.
        term_to_add = (voltages[:, -1] - voltages[parents, 0]) * branch_cond_fwd
        inds = jnp.stack([parents, jnp.zeros_like(parents)]).T
        dnums = ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0, 1),
            scatter_dims_to_operand_dims=(0, 1),
        )
        vecfield = scatter_add(vecfield, inds, term_to_add, dnums)

    return vecfield


def _triang_branched(
    parents,
    branches_in_each_level,
    lowers,
    diags,
    uppers,
    solves,
    branch_cond_fwd,
    branch_cond_bwd,
    tridiag_solver,
):
    """
    Triangulation.
    """
    for bil in reversed(branches_in_each_level[1:]):
        diags, uppers, solves = _triang_level(
            bil, lowers, diags, uppers, solves, tridiag_solver
        )
        diags, solves = _eliminate_parents_upper(
            parents,
            bil,
            diags,
            solves,
            branch_cond_fwd,
            branch_cond_bwd,
        )
    # At last level, we do not want to eliminate anymore.
    diags, uppers, solves = _triang_level(
        branches_in_each_level[0], lowers, diags, uppers, solves, tridiag_solver
    )

    return diags, uppers, solves


def _backsub_branched(
    branches_in_each_level, parents, diags, uppers, solves, branch_cond, tridiag_solver
):
    """
    Backsubstitution.
    """
    # At first level, we do not want to eliminate.
    solves = _backsub_level(
        branches_in_each_level[0], diags, uppers, solves, tridiag_solver
    )
    for bil in branches_in_each_level[1:]:
        solves = _eliminate_children_lower(bil, parents, solves, branch_cond)
        solves = _backsub_level(bil, diags, uppers, solves, tridiag_solver)
    return solves


def _triang_level(branches_in_level, lowers, diags, uppers, solves, tridiag_solver):
    bil = branches_in_level
    if tridiag_solver == "stone":
        triang_fn = stone_triang
    elif tridiag_solver == "thomas":
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


def _backsub_level(branches_in_level, diags, uppers, solves, tridiag_solver):
    bil = branches_in_level
    if tridiag_solver == "stone":
        backsub_fn = stone_backsub
    elif tridiag_solver == "thomas":
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
    parents,
    branches_in_level,
    diags,
    solves,
    branch_cond_fwd,
    branch_cond_bwd,
):
    bil = branches_in_level
    new_diag, new_solve = vmap(_eliminate_single_parent_upper, in_axes=(0, 0, 0, 0))(
        diags[bil, -1],
        solves[bil, -1],
        branch_cond_fwd[bil],
        branch_cond_bwd[bil],
    )

    # Update the diagonal elements and `b` in `Ax=b` (called `solves`).
    dnums = ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1),
    )
    inds = jnp.stack([parents[bil], jnp.zeros_like(parents[bil])]).T
    diags = scatter_add(diags, inds, new_diag, dnums)
    solves = scatter_add(solves, inds, new_solve, dnums)
    return diags, solves


def _eliminate_children_lower(branches_in_level, parents, solves, branch_cond):
    bil = branches_in_level
    solves = solves.at[bil, -1].set(
        solves[bil, -1] - branch_cond[bil] * solves[parents[bil], 0]
    )
    return solves
