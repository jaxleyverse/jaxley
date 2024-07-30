# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Tuple

import jax.numpy as jnp
from jax import vmap
from jax.experimental.sparse.linalg import spsolve as jspsolve
from jax.lax import ScatterDimensionNumbers, scatter_add
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from tridiax.stone import stone_backsub_lower, stone_triang_upper
from tridiax.thomas import thomas_backsub_lower, thomas_triang_upper

from jaxley.build_branched_tridiag import define_all_tridiags
from jaxley.utils.voltage_solver_utils import (
    build_voltage_matrix_elements,
    convert_to_csc,
    group_and_sum,
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
    children_in_level,
    parents_in_level,
    root_inds,
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
    # [1:] because of the dummy that we insert.
    all_branchpoint_vals = jnp.concatenate(
        [branchpoint_weights_parents, branchpoint_weights_children]
    )
    # Find unique group identifiers
    num_branchpoints = len(branchpoint_conds_parents)
    branchpoint_diags = -group_and_sum(
        all_branchpoint_vals, branchpoint_group_inds, num_branchpoints
    )
    branchpoint_solves = jnp.zeros((num_branchpoints,))

    branchpoint_conds_children = -delta_t * branchpoint_conds_children
    branchpoint_conds_parents = -delta_t * branchpoint_conds_parents

    nseg = voltages.shape[1]

    # Solve quasi-tridiagonal system.
    if solver == "jax.dense":
        solves = dense_solve(
            delta_t,
            uppers,
            lowers,
            diags,
            solves,
            branchpoint_conds_children,
            branchpoint_conds_parents,
            branchpoint_weights_children,
            branchpoint_weights_parents,
            branchpoint_diags,
            nseg,
            nbranches,
            row_inds,
            col_inds,
        )
    elif solver == "scipy.sparse":
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
            branchpoint_diags,
            branchpoint_solves,
            nseg,
            nbranches,
            row_inds,
            col_inds,
        )
    elif solver == "jax.scipy.sparse":
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
            branchpoint_diags,
            branchpoint_solves,
            nseg,
            nbranches,
            row_inds,
            col_inds,
        )
    elif solver.startswith("jaxley"):
        # Here, I move all child and parent indices towards a branchpoint into a larger
        # vector. This is wasteful, but it makes indexing much easier. JIT compiling
        # makes the speed difference negligible.
        # Children.
        bp_conds_children = jnp.zeros(nbranches)
        bp_weights_children = jnp.zeros(nbranches)
        # Parents.
        bp_conds_parents = jnp.zeros(nbranches)
        bp_weights_parents = jnp.zeros(nbranches)

        # `.at[inds]` requires that `inds` is not empty, so we need an if-case here.
        # `len(inds) == 0` is the case for branches and compartments.
        if num_branchpoints > 0:
            bp_conds_children = bp_conds_children.at[child_inds].set(
                branchpoint_conds_children
            )
            bp_weights_children = bp_weights_children.at[child_inds].set(
                branchpoint_weights_children
            )
            bp_conds_parents = bp_conds_parents.at[par_inds].set(branchpoint_conds_parents)
            bp_weights_parents = bp_weights_parents.at[par_inds].set(
                branchpoint_weights_parents
            )
        (
            diags,
            lowers,
            solves,
            uppers,
            branchpoint_diags,
            branchpoint_solves,
            bp_weights_children,
            bp_conds_parents,
        ) = _triang_branched(
            nseg,
            nbranches,
            delta_t,
            parents,
            branches_in_each_level,
            lowers,
            diags,
            uppers,
            solves,
            bp_conds_children,
            bp_conds_parents,
            bp_weights_children,
            bp_weights_parents,
            branchpoint_diags,
            branchpoint_solves,
            solver,
            row_inds,
            col_inds,
            children_in_level,
            parents_in_level,
            root_inds,
            child_inds,
            par_inds,
        )
        (
            solves,
            lowers,
            diags,
            bp_weights_parents,
            branchpoint_solves,
            bp_conds_children,
        ) = _backsub_branched(
            nseg,
            nbranches,
            delta_t,
            parents,
            branches_in_each_level,
            lowers,
            diags,
            uppers,
            solves,
            bp_conds_children,
            bp_conds_parents,
            bp_weights_children,
            bp_weights_parents,
            branchpoint_diags,
            branchpoint_solves,
            solver,
            row_inds,
            col_inds,
            children_in_level,
            parents_in_level,
            root_inds,
            child_inds,
            par_inds,
        )
    else:
        raise ValueError

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
    branchpoint_diags,
    branchpoint_solves,
    nseg,
    nbranches,
    row_inds,
    col_inds,
):
    (
        elements,
        solve,
        num_entries,
        start_ind_for_branchpoints,
    ) = build_voltage_matrix_elements(
        uppers,
        lowers,
        diags,
        solves,
        branchpoint_conds_children,
        branchpoint_conds_parents,
        branchpoint_weights_children,
        branchpoint_weights_parents,
        branchpoint_diags,
        branchpoint_solves,
        nseg,
        nbranches,
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
    branchpoint_diags,
    branchpoint_solves,
    nseg,
    nbranches,
    row_inds,
    col_inds,
):
    elements, solve, _, start_ind_for_branchpoints = build_voltage_matrix_elements(
        uppers,
        lowers,
        diags,
        solves,
        branchpoint_conds_children,
        branchpoint_conds_parents,
        branchpoint_weights_children,
        branchpoint_weights_parents,
        branchpoint_diags,
        branchpoint_solves,
        nseg,
        nbranches,
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
    branchpoint_solves = jnp.zeros((1,))
    big_matrix, big_solve, start_ind_for_branchpoints = build_dense(
        delta_t,
        uppers,
        lowers,
        diags,
        solves,
        branchpoint_conds_children,
        branchpoint_conds_parents,
        branchpoint_weights_children,
        branchpoint_weights_parents,
        branchpoint_solves,
        child_belongs_to_branchpoint,
        par_inds,
        child_inds,
        nseg,
        nbranches,
    )
    solution = jnp.linalg.solve(big_matrix, big_solve)
    solution = solution[:start_ind_for_branchpoints]  # Delete branchpoint voltages.
    return jnp.reshape(solution, (nseg, nbranches))


def build_dense(
    delta_t,
    uppers,
    lowers,
    diags,
    solves,
    branchpoint_conds_children,
    branchpoint_conds_parents,
    branchpoint_weights_children,
    branchpoint_weights_parents,
    branchpoint_solves,
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
    big_solve = jnp.concatenate([big_solve, branchpoint_solves])
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
    return big_matrix, big_solve, start_ind_for_branchpoints


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
    nseg,
    nbranches,
    delta_t,
    parents,
    branches_in_each_level,
    lowers,
    diags,
    uppers,
    solves,
    branchpoint_conds_children,
    branchpoint_conds_parents,
    branchpoint_weights_children,
    branchpoint_weights_parents,
    branchpoint_diags,
    branchpoint_solves,
    tridiag_solver,
    row_inds,
    col_inds,
    children_in_level,
    parents_in_level,
    root_inds,
    child_inds,
    par_inds,
):
    """Triangulation."""
    for cil, pil in zip(reversed(children_in_level), reversed(parents_in_level)):
        diags, lowers, solves, uppers = _triang_level(
            cil[:, 0], lowers, diags, uppers, solves, tridiag_solver
        )
        (
            branchpoint_diags,
            branchpoint_solves,
            branchpoint_weights_children,
        ) = _eliminate_children_lower(
            parents,
            cil,
            diags,
            solves,
            branchpoint_conds_children,
            branchpoint_weights_children,
            branchpoint_diags,
            branchpoint_solves,
        )
        diags, solves, branchpoint_conds_parents = _eliminate_parents_upper(
            parents,
            pil,
            diags,
            solves,
            branchpoint_conds_parents,
            branchpoint_weights_parents,
            branchpoint_diags,
            branchpoint_solves,
        )
    # At last level, we do not want to eliminate anymore.
    diags, lowers, solves, uppers = _triang_level(
        root_inds, lowers, diags, uppers, solves, tridiag_solver
    )
    # ###################
    # elements, solve, num_entries, start_ind_for_branchpoints = (
    #     build_voltage_matrix_elements(
    #         uppers,
    #         lowers,
    #         diags,
    #         solves,
    #         branchpoint_conds_children[child_inds],
    #         branchpoint_conds_parents[par_inds],
    #         branchpoint_weights_children[child_inds],
    #         branchpoint_weights_parents[par_inds],
    #         branchpoint_diags,
    #         branchpoint_solves,
    #         nseg,
    #         nbranches,
    #     )
    # )
    # sparse_matrix = csc_matrix(
    #     (elements, (row_inds, col_inds)), shape=(num_entries, num_entries)
    # )
    # solution = spsolve(sparse_matrix, solve)
    # solution = solution[:start_ind_for_branchpoints]  # Delete branchpoint voltages.
    # solves = jnp.reshape(solution, (nseg, nbranches))
    # z = jnp.zeros((1,))
    # return z, z, solves, z, z, z, z, z
    # ###################
    return (
        diags,
        lowers,
        solves,
        uppers,
        branchpoint_diags,
        branchpoint_solves,
        branchpoint_weights_children,
        branchpoint_conds_parents,
    )


def _backsub_branched(
    nseg,
    nbranches,
    delta_t,
    parents,
    branches_in_each_level,
    lowers,
    diags,
    uppers,
    solves,
    branchpoint_conds_children,
    branchpoint_conds_parents,
    branchpoint_weights_children,
    branchpoint_weights_parents,
    branchpoint_diags,
    branchpoint_solves,
    tridiag_solver,
    row_inds,
    col_inds,
    children_in_level,
    parents_in_level,
    root_inds,
    child_inds,
    par_inds,
):
    """
    Backsubstitution.
    """
    # At first level, we do not want to eliminate.
    solves, lowers, diags = _backsub_level(
        root_inds, diags, lowers, solves, tridiag_solver
    )
    counter = 0
    for cil, pil in zip(children_in_level, parents_in_level):
        branchpoint_weights_parents, branchpoint_solves = _eliminate_parents_lower(
            pil,
            parents,
            diags,
            solves,
            branchpoint_weights_parents,
            branchpoint_diags,
            branchpoint_solves,
        )
        branchpoint_conds_children, solves = _eliminate_children_upper(
            cil,
            parents,
            solves,
            branchpoint_conds_children,
            branchpoint_diags,
            branchpoint_solves,
        )
        # if counter == 2:
        #     ###################
        #     # print("branchpoint_conds_children", branchpoint_conds_children)
        #     # print("branchpoint_conds_parents", branchpoint_conds_parents)
        #     # print("branchpoint_weights_children", branchpoint_weights_children)
        #     # print("branchpoint_weights_parents", branchpoint_weights_parents)
        #     # print("bil", cil[:, 0])
        #     # print("bpil", cil[:, 1])
        #     # print("branchpoint_diags", branchpoint_diags)
        #     elements, solve, num_entries, start_ind_for_branchpoints = (
        #         build_voltage_matrix_elements(
        #             uppers,
        #             lowers,
        #             diags,
        #             solves,
        #             branchpoint_conds_children[child_inds],
        #             branchpoint_conds_parents[par_inds],
        #             branchpoint_weights_children[child_inds],
        #             branchpoint_weights_parents[par_inds],
        #             branchpoint_diags,
        #             branchpoint_solves,
        #             nseg,
        #             nbranches,
        #         )
        #     )
        #     sparse_matrix = csc_matrix(
        #         (elements, (row_inds, col_inds)), shape=(num_entries, num_entries)
        #     )
        #     solution = spsolve(sparse_matrix, solve)
        #     solution = solution[:start_ind_for_branchpoints]  # Delete branchpoint voltages.
        #     solves = jnp.reshape(solution, (nseg, nbranches))
        #     z = jnp.zeros((1,))
        #     return solves, z, z, z, z, z
        #     ###################
        solves, lowers, diags = _backsub_level(
            cil[:, 0], diags, lowers, solves, tridiag_solver
        )
        counter += 1
    return (
        solves,
        lowers,
        diags,
        branchpoint_weights_parents,
        branchpoint_solves,
        branchpoint_conds_children,
    )


def _triang_level(cil, lowers, diags, uppers, solves, tridiag_solver):
    bil = cil
    if tridiag_solver == "jaxley.stone":
        triang_fn = stone_triang_upper
    elif tridiag_solver == "jaxley.thomas":
        triang_fn = thomas_triang_upper
    else:
        raise NameError
    new_diags, new_lowers, new_solves = vmap(triang_fn, in_axes=(0, 0, 0, 0))(
        lowers[cil], diags[cil], uppers[cil], solves[cil]
    )
    diags = diags.at[cil].set(new_diags)
    lowers = lowers.at[cil].set(new_lowers)
    solves = solves.at[cil].set(new_solves)
    uppers = uppers.at[cil].set(0.0)

    return diags, lowers, solves, uppers


def _backsub_level(
    branches_in_level: jnp.ndarray,
    diags: jnp.ndarray,
    uppers: jnp.ndarray,
    solves: jnp.ndarray,
    tridiag_solver: str,
) -> jnp.ndarray:
    bil = branches_in_level
    if tridiag_solver == "jaxley.stone":
        backsub_fn = stone_backsub_lower
    elif tridiag_solver == "jaxley.thomas":
        backsub_fn = thomas_backsub_lower
    else:
        raise NameError
    solves = solves.at[bil].set(
        vmap(backsub_fn, in_axes=(0, 0, 0))(solves[bil], lowers[bil], diags[bil])
    )
    lowers = lowers.at[bil].set(0.0)
    diags = diags.at[bil].set(1.0)
    return solves, lowers, diags


def _eliminate_children_lower(
    parents,
    cil,
    diags,
    solves,
    branchpoint_conds_children,
    branchpoint_weights_children,
    branchpoint_diags,
    branchpoint_solves,
):
    bil = cil[:, 0]
    bpil = cil[:, 1]
    new_diag, new_solve = vmap(_eliminate_single_child_lower, in_axes=(0, 0, 0, 0))(
        diags[bil, 0],
        solves[bil, 0],
        branchpoint_conds_children[bil],
        branchpoint_weights_children[bil],
    )
    branchpoint_diags = branchpoint_diags.at[bpil].add(new_diag)
    branchpoint_solves = branchpoint_solves.at[bpil].add(new_solve)
    branchpoint_weights_children = branchpoint_weights_children.at[bil].set(0.0)
    return branchpoint_diags, branchpoint_solves, branchpoint_weights_children


def _eliminate_single_child_lower(
    diags,
    solves,
    branchpoint_conds_children,
    branchpoint_weights_children,
):
    multiplying_factor = -branchpoint_weights_children / diags

    update_diag = multiplying_factor * branchpoint_conds_children
    update_solve = multiplying_factor * solves
    return update_diag, update_solve


def _eliminate_parents_upper(
    parents,
    pil,
    diags,
    solves,
    branchpoint_conds_parents,
    branchpoint_weights_parents,
    branchpoint_diags,
    branchpoint_solves,
):
    bil = pil[:, 0]
    bpil = pil[:, 1]
    new_diag, new_solve = vmap(_eliminate_single_parent_upper, in_axes=(0, 0, 0, 0))(
        branchpoint_conds_parents[bil],
        branchpoint_weights_parents[bil],
        branchpoint_diags[bpil],
        branchpoint_solves[bpil],
    )

    # Update the diagonal elements and `b` in `Ax=b` (called `solves`).
    diags = diags.at[bil, -1].add(new_diag)
    solves = solves.at[bil, -1].add(new_solve)
    branchpoint_conds_parents = branchpoint_conds_parents.at[bil].set(0.0)

    return diags, solves, branchpoint_conds_parents


def _eliminate_single_parent_upper(
    branchpoint_conds_parents,
    branchpoint_weights_parents,
    branchpoint_diags,
    branchpoint_solves,
):
    multiplying_factor = branchpoint_conds_parents / branchpoint_diags

    update_diag = -multiplying_factor * branchpoint_weights_parents
    update_solve = -multiplying_factor * branchpoint_solves
    return update_diag, update_solve


def _eliminate_parents_lower(
    pil,
    parents,
    diags,
    solves,
    branchpoint_weights_parents,
    branchpoint_diags,
    branchpoint_solves,
):
    bil = pil[:, 0]
    bpil = pil[:, 1]
    branchpoint_solves = branchpoint_solves.at[bpil].add(
        -solves[bil, -1] * branchpoint_weights_parents[bil] / diags[bil, -1]
    )
    branchpoint_weights_parents = branchpoint_weights_parents.at[bil].set(0.0)
    return branchpoint_weights_parents, branchpoint_solves


def _eliminate_children_upper(
    cil,
    parents,
    solves,
    branchpoint_conds_children,
    branchpoint_diags,
    branchpoint_solves,
):
    bil = cil[:, 0]
    bpil = cil[:, 1]
    solves = solves.at[bil, 0].add(
        -branchpoint_solves[bpil]
        * branchpoint_conds_children[bil]
        / branchpoint_diags[bpil]
    )
    branchpoint_conds_children = branchpoint_conds_children.at[bil].set(0.0)
    return branchpoint_conds_children, solves
