# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Tuple

import jax.numpy as jnp
from jax import vmap
from jax.experimental.sparse.linalg import spsolve as jspsolve
from jax.lax import ScatterDimensionNumbers, scatter_add
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from tridiax.stone import stone_backsub, stone_triang
from tridiax.thomas import thomas_backsub_lower, thomas_triang_upper

from jaxley.build_branched_tridiag import define_all_tridiags
from jaxley.utils.voltage_solver_utils import (
    build_voltage_matrix_elements,
    convert_to_csc,
)


def step_voltage_explicit(
    voltages: jnp.ndarray,
    voltage_terms: jnp.ndarray,
    constant_terms: jnp.ndarray,
    coupling_conds_bwd: jnp.ndarray,
    coupling_conds_fwd: jnp.ndarray,
    branch_cond_fwd: jnp.ndarray,
    branch_cond_bwd: jnp.ndarray,
    nbranches: int,
    parents: jnp.ndarray,
    delta_t: float,
) -> jnp.ndarray:
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
    coupling_conds_upper,
    coupling_conds_lower,
    summed_coupling_conds,
    branchpoint_conds_children,
    branchpoint_conds_parents,
    branchpoint_weights_children,
    branchpoint_weights_parents,
    child_belongs_to_branchpoint,
    par_inds,
    child_inds,
    nbranches,
    parents,
    branches_in_each_level,
    solver: str,
    delta_t,
    branchpoint_group_inds,
    row_inds,
    col_inds,
):
    """Solve one timestep of branched nerve equations with implicit (backward) Euler."""
    voltages = jnp.reshape(voltages, (nbranches, -1))
    voltage_terms = jnp.reshape(voltage_terms, (nbranches, -1))
    constant_terms = jnp.reshape(constant_terms, (nbranches, -1))
    coupling_conds_upper = jnp.reshape(coupling_conds_upper, (nbranches, -1))
    coupling_conds_lower = jnp.reshape(coupling_conds_lower, (nbranches, -1))
    summed_coupling_conds = jnp.reshape(summed_coupling_conds, (nbranches, -1))

    # Define quasi-tridiagonal system.
    lowers, diags, uppers, solves = define_all_tridiags(
        voltages,
        voltage_terms,
        constant_terms,
        nbranches,
        coupling_conds_upper,
        coupling_conds_lower,
        summed_coupling_conds,
        delta_t,
    )

    nseg = 2
    if solver == "scipy":
        solves = sparse_scipy_solve(
            delta_t,
            uppers,
            lowers,
            diags,
            solves,
            branchpoint_conds_children,
            branchpoint_conds_parents,
            branchpoint_weights_children,
            branchpoint_weights_parents,
            nseg,
            nbranches,
            branchpoint_group_inds,
            row_inds,
            col_inds,
        )
    elif solver == "jax.scipy":
        solves = sparse_jax_scipy_solve(
            delta_t,
            uppers,
            lowers,
            diags,
            solves,
            branchpoint_conds_children,
            branchpoint_conds_parents,
            branchpoint_weights_children,
            branchpoint_weights_parents,
            nseg,
            nbranches,
            branchpoint_group_inds,
            row_inds,
            col_inds,
        )
    elif solver == "custom_thomas":
        raise NotImplementedError
    elif solver == "custom_stone":
        raise NotImplementedError
    else:
        raise ValueError

    # # Solve quasi-tridiagonal system.
    # diags, lowers, solves = _triang_branched(
    #     parents,
    #     branches_in_each_level,
    #     lowers,
    #     diags,
    #     uppers,
    #     solves,
    #     -delta_t * branch_cond_fwd,
    #     -delta_t * branch_cond_bwd,
    #     tridiag_solver,
    # )
    # solves = _backsub_branched(
    #     branches_in_each_level,
    #     parents,
    #     diags,
    #     lowers,
    #     solves,
    #     -delta_t * branch_cond_bwd,
    #     tridiag_solver,
    # )
    return solves


def sparse_scipy_solve(
    delta_t,
    uppers,
    lowers,
    diags,
    solves,
    branchpoint_conds_children,
    branchpoint_conds_parents,
    branchpoint_weights_children,
    branchpoint_weights_parents,
    nseg,
    nbranches,
    branchpoint_group_inds,
    row_inds,
    col_inds,
):
    elements, solve, num_entries, start_ind_for_branchpoints = (
        build_voltage_matrix_elements(
            delta_t,
            uppers,
            lowers,
            diags,
            solves,
            branchpoint_conds_children,
            branchpoint_conds_parents,
            branchpoint_weights_children,
            branchpoint_weights_parents,
            nseg,
            nbranches,
            branchpoint_group_inds,
        )
    )

    sparse_matrix = csc_matrix(
        (elements, (row_inds, col_inds)), shape=(num_entries, num_entries)
    )

    solution = spsolve(sparse_matrix, solve)
    solution = solution[:start_ind_for_branchpoints]  # Delete branchpoint voltages.
    return jnp.reshape(solution, (nseg, nbranches))


def sparse_jax_scipy_solve(
    delta_t,
    uppers,
    lowers,
    diags,
    solves,
    branchpoint_conds_children,
    branchpoint_conds_parents,
    branchpoint_weights_children,
    branchpoint_weights_parents,
    nseg,
    nbranches,
    branchpoint_group_inds,
    row_inds,
    col_inds,
):
    elements, solve, _, start_ind_for_branchpoints = (
        build_voltage_matrix_elements(
            delta_t,
            uppers,
            lowers,
            diags,
            solves,
            branchpoint_conds_children,
            branchpoint_conds_parents,
            branchpoint_weights_children,
            branchpoint_weights_parents,
            nseg,
            nbranches,
            branchpoint_group_inds,
        )
    )
    data_inds, indices, indptr = convert_to_csc(
        num_elements=len(elements), row_ind=row_inds, col_ind=col_inds
    )
    elements = elements[data_inds]

    solution = jspsolve(elements, indices=indices, indptr=indptr, b=solve)
    solution = solution[:start_ind_for_branchpoints]  # Delete branchpoint voltages.
    return jnp.reshape(solution, (nseg, nbranches))


def dense_solve(
    delta_t,
    uppers,
    lowers,
    diags,
    solves,
    branchpoint_conds_children,
    branchpoint_conds_parents,
    branchpoint_weights_children,
    branchpoint_weights_parents,
    child_belongs_to_branchpoint,
    par_inds,
    child_inds,
    nseg,
    nbranches,
):
    # Construct dense matrix.
    small_matrices = []
    for lower, upper, diag in zip(uppers, lowers, diags):
        mat = jnp.zeros((nseg, nseg))
        for i in range(nseg):
            for j in range(nseg):
                if i == j:
                    mat = mat.at[i, j].set(diag[i])
                elif i == j + 1:
                    mat = mat.at[i, j].set(upper[i])
                elif i == j - 1:
                    mat = mat.at[i, j].set(lower[i])
        small_matrices.append(mat)

    big_solve = jnp.concatenate(solves)

    num_branchpoints = len(branchpoint_conds_parents)
    start_ind_for_branchpoints = nseg * nbranches
    avg_value = jnp.zeros((num_branchpoints,))
    big_solve = jnp.concatenate([big_solve, avg_value])
    big_matrix = jnp.zeros(
        (nseg * nbranches + num_branchpoints, nseg * nbranches + num_branchpoints)
    )

    for i in range(len(diags)):
        small_mat = small_matrices[i]
        big_matrix = big_matrix.at[
            i * nseg : (i + 1) * nseg, i * nseg : (i + 1) * nseg
        ].set(small_mat)

    branchpoint_inds_parents = start_ind_for_branchpoints + jnp.arange(num_branchpoints)
    branchpoint_inds_children = (
        start_ind_for_branchpoints + child_belongs_to_branchpoint
    )
    branch_inds_parents = par_inds * nseg + (nseg - 1)
    branch_inds_children = child_inds * nseg

    # Entries for branch points (last columns in matrix).
    big_matrix = big_matrix.at[branch_inds_parents, branchpoint_inds_parents].set(
        -delta_t * branchpoint_conds_parents
    )
    big_matrix = big_matrix.at[branch_inds_children, branchpoint_inds_children].set(
        -delta_t * branchpoint_conds_children
    )

    # Enforce Kirchhofs current law at the branch point (last rows in matrix).
    # We do not have to multiply by `-delta_t` here because the row can be multiplied
    # by any value (`solve` in this row is `0.0`).
    big_matrix = big_matrix.at[branchpoint_inds_parents, branch_inds_parents].set(
        branchpoint_weights_parents
    )
    big_matrix = big_matrix.at[branchpoint_inds_children, branch_inds_children].set(
        branchpoint_weights_children
    )
    branchpoints = jnp.arange(
        start_ind_for_branchpoints, start_ind_for_branchpoints + num_branchpoints
    )
    big_matrix = big_matrix.at[branchpoints, branchpoints].set(
        -jnp.sum(big_matrix, axis=1)[start_ind_for_branchpoints:]
    )

    solution = jnp.linalg.solve(big_matrix, big_solve)
    solution = solution[:start_ind_for_branchpoints]  # Delete branchpoint voltages.
    return jnp.reshape(solution, (nseg, nbranches))


def voltage_vectorfield(
    parents: jnp.ndarray,
    voltages: jnp.ndarray,
    voltage_terms: jnp.ndarray,
    constant_terms: jnp.ndarray,
    coupling_conds_bwd: jnp.ndarray,
    coupling_conds_fwd: jnp.ndarray,
    branch_cond_fwd: jnp.ndarray,
    branch_cond_bwd: jnp.ndarray,
) -> jnp.ndarray:
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
    parents: jnp.ndarray,
    branches_in_each_level: jnp.ndarray,
    lowers: jnp.ndarray,
    diags: jnp.ndarray,
    uppers: jnp.ndarray,
    solves: jnp.ndarray,
    branch_cond_fwd: jnp.ndarray,
    branch_cond_bwd: jnp.ndarray,
    tridiag_solver: str,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Triangulation.
    """
    for bil in reversed(branches_in_each_level[1:]):
        diags, lowers, solves = _triang_level(
            bil, lowers, diags, uppers, solves, tridiag_solver
        )
        uppers = uppers.at[bil, :].set(0.0)
        diags, solves = _eliminate_parents_upper(
            parents,
            bil,
            diags,
            solves,
            branch_cond_fwd,
            branch_cond_bwd,
        )
    # At last level, we do not want to eliminate anymore.
    diags, lowers, solves = _triang_level(
        branches_in_each_level[0], lowers, diags, uppers, solves, tridiag_solver
    )
    uppers = uppers.at[branches_in_each_level[0], :].set(0.0)
    return diags, lowers, solves


def _backsub_branched(
    branches_in_each_level, parents, diags, lowers, solves, branch_cond, tridiag_solver
):
    """
    Backsubstitution.
    """
    # At first level, we do not want to eliminate.
    solves = _backsub_level(
        branches_in_each_level[0], diags, lowers, solves, tridiag_solver
    )
    for bil in branches_in_each_level[1:]:
        solves = _eliminate_children_lower(bil, parents, solves, branch_cond)
        solves = _backsub_level(bil, diags, lowers, solves, tridiag_solver)
    return solves


def _triang_level(
    branches_in_level: jnp.ndarray,
    lowers: jnp.ndarray,
    diags: jnp.ndarray,
    uppers: jnp.ndarray,
    solves: jnp.ndarray,
    tridiag_solver: str,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    bil = branches_in_level
    if tridiag_solver == "stone":
        triang_fn = stone_triang
    elif tridiag_solver == "thomas":
        triang_fn = thomas_triang_upper
    else:
        raise NameError
    new_diags, new_lowers, new_solves = vmap(triang_fn, in_axes=(0, 0, 0, 0))(
        lowers[bil], diags[bil], uppers[bil], solves[bil]
    )
    diags = diags.at[bil].set(new_diags)
    lowers = lowers.at[bil].set(new_lowers)
    solves = solves.at[bil].set(new_solves)

    return diags, lowers, solves


def _backsub_level(
    branches_in_level: jnp.ndarray,
    diags: jnp.ndarray,
    uppers: jnp.ndarray,
    solves: jnp.ndarray,
    tridiag_solver: str,
) -> jnp.ndarray:
    bil = branches_in_level
    if tridiag_solver == "stone":
        backsub_fn = stone_backsub
    elif tridiag_solver == "thomas":
        backsub_fn = thomas_backsub_lower
    else:
        raise NameError
    solves = solves.at[bil].set(
        vmap(backsub_fn, in_axes=(0, 0, 0))(solves[bil], uppers[bil], diags[bil])
    )
    return solves


def _eliminate_single_parent_upper(
    diag_at_branch: jnp.ndarray,
    solve_at_branch: jnp.ndarray,
    branch_cond_fwd: jnp.ndarray,
    branch_cond_bwd: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    last_of_3 = diag_at_branch
    last_of_3_solve = solve_at_branch

    multiplying_factor = branch_cond_fwd / last_of_3

    update_diag = -multiplying_factor * branch_cond_bwd
    update_solve = -multiplying_factor * last_of_3_solve
    return update_diag, update_solve


def _eliminate_parents_upper(
    parents: jnp.ndarray,
    branches_in_level: jnp.ndarray,
    diags: jnp.ndarray,
    solves: jnp.ndarray,
    branch_cond_fwd: jnp.ndarray,
    branch_cond_bwd: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    bil = branches_in_level
    new_diag, new_solve = vmap(_eliminate_single_parent_upper, in_axes=(0, 0, 0, 0))(
        diags[bil, 0],
        solves[bil, 0],
        branch_cond_fwd[bil],
        branch_cond_bwd[bil],
    )

    # Update the diagonal elements and `b` in `Ax=b` (called `solves`).
    dnums = ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1),
    )
    nseg = diags.shape[1]
    inds = jnp.stack(
        [parents[bil], (nseg - 1) * jnp.ones_like(parents[bil]).astype(int)]
    ).T
    diags = scatter_add(diags, inds, new_diag, dnums)
    solves = scatter_add(solves, inds, new_solve, dnums)
    return diags, solves


def _eliminate_children_lower(
    branches_in_level: jnp.ndarray,
    parents: jnp.ndarray,
    solves: jnp.ndarray,
    branch_cond: jnp.ndarray,
) -> jnp.ndarray:
    bil = branches_in_level
    solves = solves.at[bil, 0].set(
        solves[bil, 0] - branch_cond[bil] * solves[parents[bil], -1]
    )
    return solves
