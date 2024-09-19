# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd


def remap_index_to_masked(
    index, nodes: pd.DataFrame, max_nseg: int, nseg_per_branch: jnp.ndarray
):
    """Convert actual index of the compartment to the index in the masked system.

    E.g. if `nsegs = [2, 4]`, then the index `3` would be mapped to `5` because the
    masked `nsegs` are `[4, 4]`.
    """
    cumsum_nseg_per_branch = jnp.concatenate(
        [
            jnp.asarray([0]),
            jnp.cumsum(nseg_per_branch),
        ]
    )
    branch_inds = nodes.loc[index, "branch_index"].to_numpy()
    remainders = index - cumsum_nseg_per_branch[branch_inds]
    return branch_inds * max_nseg + remainders


def convert_to_csc(
    num_elements: int, row_ind: np.ndarray, col_ind: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert between two representations for sparse systems.

    This is needed because `jax.scipy.linalg.spsolve` requires the `(ind, indptr)`
    representation, but the `(row, col)` is more intuitive and easier to build.

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


def comp_edges_to_indices(
    comp_edges: pd.DataFrame,
) -> Tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generates sparse matrix indices from the table of node edges.

    This is only used for the `jax.sparse` voltage solver.

    Args:
        comp_edges: Dataframe with three columns (sink, source, type).

    Returns:
        n_nodes: The number of total nodes (including branchpoints).
        data_inds: The indices to reorder the data.
        indices and indptr: Indices passed to the sparse matrix solver.
    """
    # Build indices for diagonals.
    sources = np.asarray(comp_edges["source"].to_list())
    sinks = np.asarray(comp_edges["sink"].to_list())
    n_nodes = np.max(sinks) + 1 if len(sinks) > 0 else 1
    diagonal_inds = jnp.stack([jnp.arange(n_nodes), jnp.arange(n_nodes)])

    # Build indices for off-diagonals.
    off_diagonal_inds = jnp.stack([sources, sinks]).astype(int)

    # Concatenate indices of diagonals and off-diagonals.
    all_inds = jnp.concatenate([diagonal_inds, off_diagonal_inds], axis=1)

    # Cast (row, col) indices to the format required for the `jax` sparse solver.
    data_inds, indices, indptr = convert_to_csc(
        num_elements=all_inds.shape[1],
        row_ind=all_inds[0],
        col_ind=all_inds[1],
    )
    return n_nodes, data_inds, indices, indptr
