from typing import Tuple

import jax.numpy as jnp
import numpy as np


def compute_morphology_indices(
    num_branchpoints,
    child_belongs_to_branchpoint,
    par_inds,
    child_inds,
    nseg,
    nbranches,
):
    """Return (row, col) to build the sparse matrix defining the voltage eqs.

    This is only used in `Base._init_morph_for_debugging()`, so is only ever used for
    debugging.

    This is run at `init`, not during runtime.

    The sparse matrix will contain entries in the following order:
    1) All diags
    2) All within-branch uppers
    3) All within-branch lowers
    4) All parent branchpoint columns
    5) All child branchpoint columns
    6) All parent branchpoint rows
    7) All child branchpoint rows
    8) All branchpoint diagonals
    """
    diag_col_inds = jnp.arange(nseg * nbranches)
    diag_row_inds = jnp.arange(nseg * nbranches)

    upper_col_inds = drop_nseg_th_element(diag_col_inds, nseg, nbranches, 0)
    upper_row_inds = drop_nseg_th_element(diag_row_inds, nseg, nbranches, nseg - 1)

    lower_col_inds = drop_nseg_th_element(diag_col_inds, nseg, nbranches, nseg - 1)
    lower_row_inds = drop_nseg_th_element(diag_row_inds, nseg, nbranches, 0)

    start_ind_for_branchpoints = nseg * nbranches
    branchpoint_inds_parents = start_ind_for_branchpoints + jnp.arange(num_branchpoints)
    branchpoint_inds_children = (
        start_ind_for_branchpoints + child_belongs_to_branchpoint
    )

    branch_inds_parents = par_inds * nseg + (nseg - 1)
    branch_inds_children = child_inds * nseg

    branchpoint_parent_columns_col_inds = branchpoint_inds_parents
    branchpoint_parent_columns_row_inds = branch_inds_parents

    branchpoint_children_columns_col_inds = branchpoint_inds_children
    branchpoint_children_columns_row_inds = branch_inds_children

    branchpoint_parent_row_col_inds = branch_inds_parents
    branchpoint_parent_row_row_inds = branchpoint_inds_parents

    branchpoint_children_row_col_inds = branch_inds_children
    branchpoint_children_row_row_inds = branchpoint_inds_children

    branchpoint_diags_col_inds = jnp.arange(
        start_ind_for_branchpoints, start_ind_for_branchpoints + num_branchpoints
    )
    branchpoint_diags_row_inds = jnp.arange(
        start_ind_for_branchpoints, start_ind_for_branchpoints + num_branchpoints
    )

    col_inds = jnp.concatenate(
        [
            diag_col_inds,
            upper_col_inds,
            lower_col_inds,
            branchpoint_parent_columns_col_inds,
            branchpoint_children_columns_col_inds,
            branchpoint_parent_row_col_inds,
            branchpoint_children_row_col_inds,
            branchpoint_diags_col_inds,
        ]
    )
    row_inds = jnp.concatenate(
        [
            diag_row_inds,
            upper_row_inds,
            lower_row_inds,
            branchpoint_parent_columns_row_inds,
            branchpoint_children_columns_row_inds,
            branchpoint_parent_row_row_inds,
            branchpoint_children_row_row_inds,
            branchpoint_diags_row_inds,
        ]
    )
    return {"col_inds": col_inds, "row_inds": row_inds}


def build_voltage_matrix_elements(
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
):
    """Return data to build the sparse matrix defining the voltage equations.

    The sparse matrix will contain entries in the following order:
    1) All diags
    2) All within-branch uppers
    3) All within-branch lowers
    4) All parent branchpoint columns
    5) All child branchpoint columns
    6) All parent branchpoint rows
    7) All child branchpoint rows
    8) All branchpoint diagonals
    """
    num_branchpoints = len(branchpoint_conds_parents)
    num_entries = nseg * nbranches + num_branchpoints

    diag_elements = diags.flatten()
    upper_elements = uppers.flatten()
    lower_elements = lowers.flatten()

    start_ind_for_branchpoints = nseg * nbranches
    branchpoint_parent_columns_elements = branchpoint_conds_parents
    branchpoint_children_columns_elements = branchpoint_conds_children
    branchpoint_parent_row_elements = branchpoint_weights_parents
    branchpoint_children_row_elements = branchpoint_weights_children

    elements = jnp.concatenate(
        [
            diag_elements,
            upper_elements,
            lower_elements,
            branchpoint_parent_columns_elements,
            branchpoint_children_columns_elements,
            branchpoint_parent_row_elements,
            branchpoint_children_row_elements,
            branchpoint_diags,
        ]
    )

    # Build the full solve, including zeros at branchpoints.
    big_solve = jnp.concatenate(solves)
    num_branchpoints = num_entries - start_ind_for_branchpoints
    big_solve = jnp.concatenate([big_solve, branchpoint_solves])

    return (
        elements,
        big_solve,
        num_entries,
        start_ind_for_branchpoints,
    )


def drop_nseg_th_element(
    arr: jnp.ndarray, nseg: int, nbranches: int, start: int
) -> jnp.ndarray:
    """
    Create an array of integers from 0 to limit, dropping every n-th element.

    Written by ChatGPT.

    Args:
        arr: The array from which to drop elements.
        nseg: The interval of elements to drop (every n-th element).
        start: An offset on where to start removing.

    Returns:
        An array of integers with every n-th element dropped.
    """
    # Drop every n-th element
    result = jnp.delete(
        arr, jnp.arange(start, nseg * nbranches, nseg), assume_unique_indices=True
    )

    return result


def convert_to_csc(
    num_elements: int, row_ind: np.ndarray, col_ind: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert between two representations for sparse systems.

    This is needed because `jax.scipy.linalg.spsolve` requires the `(ind, indptr)`
    representation, but for `scipy` (and for intuition) we build the `(row, col)`
    representation.

    This function uses `np` instead of `jnp` because it only deals with indexing which
    can be dealt with only based on the branch structure (i.e. independent of any
    parameter values).

    Written by ChatGPT.
    """
    data_inds = np.arange(num_elements)
    # Step 1: Sort by (col_ind, row_ind)
    sorted_indices = np.lexsort((row_ind, col_ind))
    data_inds = data_inds[sorted_indices]
    row_ind = row_ind[sorted_indices]
    col_ind = col_ind[sorted_indices]

    # Step 2: Create indptr array
    n_cols = col_ind.max() + 1
    indptr = np.zeros(n_cols + 1, dtype=int)
    np.add.at(indptr, col_ind + 1, 1)
    np.cumsum(indptr, out=indptr)

    # Step 3: The row indices are already sorted
    indices = row_ind

    return data_inds, indices, indptr
