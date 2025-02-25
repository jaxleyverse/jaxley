# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd


def remap_index_to_masked(
    index, nodes: pd.DataFrame, padded_cumsum_ncomp, ncomp_per_branch: jnp.ndarray
):
    """Convert actual index of the compartment to the index in the masked system.

    E.g. if `ncomps = [2, 4]`, then the index `3` would be mapped to `5` because the
    masked `ncomps` are `[4, 4]`. I.e.:

    original: [0, 1,           2, 3, 4, 5]
    masked:   [0, 1, (2) ,(3) ,4, 5, 6, 7]
    """
    cumsum_ncomp_per_branch = jnp.concatenate(
        [
            jnp.asarray([0]),
            jnp.cumsum(ncomp_per_branch),
        ]
    )
    branch_inds = nodes.loc[index, "global_branch_index"].to_numpy()
    remainders = index - cumsum_ncomp_per_branch[branch_inds]
    return padded_cumsum_ncomp[branch_inds] + remainders


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


class JaxleySolveIndexer:
    """Indexer for easy access to compartment indices given a branch index.

    Used only by the custom Jaxley solvers. This class has two purposes:

    1) It simplifies indexing. Indexing is difficult because every branch has a
    different number of compartments (in the solve, every branch within a level has
    the same number of compartments, but the number can still differ between levels).

    2) It stores several attributes such that we do not have to track all of them
    separately before they are used in `step()`.
    """

    def __init__(
        self,
        cumsum_ncomp: np.ndarray,
        ncomp_per_branch: np.ndarray,
        branchpoint_group_inds: Optional[np.ndarray] = None,
        children_in_level: Optional[np.ndarray] = None,
        parents_in_level: Optional[np.ndarray] = None,
        root_inds: Optional[np.ndarray] = None,
        remapped_node_indices: Optional[np.ndarray] = None,
    ):
        self.cumsum_ncomp = np.asarray(cumsum_ncomp)
        self.ncomp_per_branch = np.asarray(ncomp_per_branch)

        # Save items for easier access.
        self.branchpoint_group_inds = branchpoint_group_inds
        self.children_in_level = children_in_level
        self.parents_in_level = parents_in_level
        self.root_inds = root_inds
        self.remapped_node_indices = remapped_node_indices

    def first(self, branch_inds: np.ndarray) -> np.ndarray:
        """Return the indices of the first compartment of all `branch_inds`."""
        return self.cumsum_ncomp[branch_inds]

    def masked_last(self, branch_inds: np.ndarray) -> np.ndarray:
        """Return the indices of the last compartment of all `branch_inds`.

        Notably, this returns the index of the last compartment _with_ taking into
        account the masking structure which is needed to make all branches in a level
        have the same number of compartments.

        Example:
        ```
        parents = [-1, 0, 0, 1]
        ncomp_per_branch = [2, 2, 4, 4]
        masked_last([0, 1, 2, 3]) => [1, 5, 9, 13]
        # The important one is `masked_last(1) = 5`, because the second level has a
        # branch with 4 compartments.
        ```
        """
        return self.cumsum_ncomp[branch_inds + 1] - 1

    def last(self, branch_inds: np.ndarray) -> np.ndarray:
        """Return the indices of the last compartment of all `branch_inds`.

        Unlike `masked_last`, this returns the index of the last compartment that
        acutally has a value.

        Example:
        ```
        parents = [-1, 0, 0, 1]
        ncomp_per_branch = [2, 2, 4, 4]
        masked_last([0, 1, 2, 3]) => [1, 3, 9, 13]
        # The important one is `masked_last(1) = 3`, because the branch has only
        # 2 compartments.
        ```
        """
        return self.cumsum_ncomp[branch_inds] + self.ncomp_per_branch[branch_inds] - 1

    def branch(self, branch_inds: np.ndarray) -> np.ndarray:
        """Return indices of all compartments in all `branch_inds`."""
        start_inds = self.first(branch_inds)
        end_inds = self.masked_last(branch_inds) + 1
        return self._consecutive_indices(start_inds, end_inds)

    def lower(self, branch_inds: np.ndarray) -> np.ndarray:
        """Return indices of all lowers in all `branch_inds`.

        This is needed because the `lowers` array in the voltage solve is instantiated
        to have as many elements as the `diagonal`. In this method, we get rid of
        this additional element."""
        start_inds = self.first(branch_inds) + 1
        end_inds = self.masked_last(branch_inds) + 1
        return self._consecutive_indices(start_inds, end_inds)

    def upper(self, branch_inds: np.ndarray) -> np.ndarray:
        """Return indices of all uppers in all `branch_inds`.

        This is needed because the `uppers` array in the voltage solve is instantiated
        to have as many elements as the `diagonal`. In this method, we get rid of
        this additional element."""
        start_inds = self.first(branch_inds)
        end_inds = self.masked_last(branch_inds)
        return self._consecutive_indices(start_inds, end_inds)

    def _consecutive_indices(
        self, start_inds: np.ndarray, end_inds: np.ndarray
    ) -> np.ndarray:
        """Return array of all indices in [start, end], for every start, end.

        It also reshape the indices to `(nbranches, ncomp)`.

        E.g.:
        ```
        start_inds = [0, 6]
        end_inds = [3, 9]
        -->
        [[0, 1, 2], [6, 7, 8]]
        ```
        """
        n_inds = end_inds - start_inds
        assert np.all(n_inds[0] == n_inds), (
            "The indexer only supports indexing into branches with the same number "
            "of compartments."
        )
        if n_inds[0] > 0:
            repeated_starts = np.reshape(np.repeat(start_inds, n_inds), (-1, n_inds[0]))
            # For single compartment neurons there are no uppers or lowers, so `n_inds`
            # can be zero.
            return repeated_starts + np.arange(n_inds[0]).astype(int)
        else:
            return np.asarray([[]] * len(start_inds)).astype(int)

    def mask(self, indices: np.ndarray) -> np.ndarray:
        """Return the masked index given the global compartment index.

        The masked index is the one which occurs because all branches within a level
        must have the same number of compartments for the solve.
        """
        return self.remapped_node_indices[indices]
