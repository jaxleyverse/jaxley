# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array, vmap
from jax.typing import ArrayLike

from jaxley.utils.misc_utils import cumsum_leading_zero


def compute_levels(parents):
    levels = np.zeros_like(parents)

    for i, p in enumerate(parents):
        if p == -1:
            levels[i] = 0
        else:
            levels[i] = levels[p] + 1
    return levels


def _compute_num_children(parents) -> np.ndarray:
    """Return the number of children of every branch."""
    parents = np.asarray(parents)
    return np.bincount(parents[parents >= 0], minlength=len(parents))


def _compute_index_of_child(parents):
    """For every branch, it returns the how many-eth child of its parent it is.

    Example:
    ```
    parents = [-1, 0, 0, 1, 1, 1]
    _compute_index_of_child(parents) -> [-1, 0, 1, 0, 1, 2]
    ```
    """
    num_branches = len(parents)
    current_num_children_for_each_branch = np.zeros((num_branches,), np.dtype("int"))
    index_of_child = [-1]
    for b in range(1, num_branches):
        index_of_child.append(current_num_children_for_each_branch[parents[b]])
        current_num_children_for_each_branch[parents[b]] += 1
    return index_of_child


def compute_children_indices(parents) -> list[np.ndarray]:
    """Return all children indices of every branch.

    Example:
    ```
    parents = [-1, 0, 0]
    compute_children_indices(parents) -> [[1, 2], [], []]
    ```
    """
    parents = np.asarray(parents)
    num_children = _compute_num_children(parents)
    # Sorting by parent puts the roots (-1) first, then the children of every branch,
    # in ascending order (the sort is stable).
    by_parent = np.argsort(parents, kind="stable")[len(parents) - num_children.sum() :]
    return np.split(by_parent, np.cumsum(num_children)[:-1])


def local_index_of_loc(
    loc: float, global_branch_ind: int, ncomp_per_branch: int
) -> int:
    """Returns the local index of a comp given a loc [0, 1] and the index of a branch.

    This is used because we specify locations such as synapses as a value between 0 and
    1. We have to convert this onto a discrete segment here.

    Args:
        branch_ind: Index of the branch.
        loc: Location (in [0, 1]) along that branch.
        ncomp_per_branch: Number of segments of each branch.

    Returns:
        The local index of the compartment.
    """
    ncomp = ncomp_per_branch[global_branch_ind]  # only for convenience.
    possible_locs = np.linspace(0.5 / ncomp, 1 - 0.5 / ncomp, ncomp)
    ind_along_branch = np.argmin(np.abs(possible_locs - loc))
    return ind_along_branch


def loc_of_index(global_comp_index, global_branch_index, ncomp_per_branch):
    """Return location corresponding to global compartment index."""
    cumsum_ncomp = cumsum_leading_zero(ncomp_per_branch)
    index = global_comp_index - cumsum_ncomp[global_branch_index]
    ncomp = ncomp_per_branch[global_branch_index]
    return (0.5 + index) / ncomp


def remap_to_consecutive(arr):
    """Maps an array of integers to an array of consecutive integers.

    E.g. `[0, 0, 1, 4, 4, 6, 6] -> [0, 0, 1, 2, 2, 3, 3]`
    """
    _, inverse_indices = np.unique(arr, return_inverse=True)
    return inverse_indices


v_interp = vmap(jnp.interp, in_axes=(None, None, 1))


def interpolate_xyzr(loc: float, coords: np.ndarray):
    """Perform a linear interpolation between xyz-coordinates.

    Args:
        loc: The location in [0,1] along the branch.
        coords: Array containing the reconstructed xyzr points of the branch.

    Return:
        Interpolated xyz coordinate at `loc`, shape `(3,).
    """
    dl = np.sqrt(np.sum(np.diff(coords[:, :3], axis=0) ** 2, axis=1))
    pathlens = np.insert(np.cumsum(dl), 0, 0)  # cumulative length of sections
    norm_pathlens = pathlens / np.maximum(1e-8, pathlens[-1])  # norm lengths to [0,1].

    return v_interp(loc, norm_pathlens, coords)


def params_to_pstate(
    params: list[dict[str, ArrayLike]],
    indices_set_by_trainables: list[ArrayLike],
):
    """Make outputs `get_parameters()` conform with outputs of `.data_set()`.

    `make_trainable()` followed by `params=get_parameters()` does not return indices
    because these indices would also be differentiated by `jax.grad` (as soon as
    the `params` are passed to `def simulate(params)`. Therefore, in `jx.integrate`,
    we run the function to add indices to the dict. The outputs of `params_to_pstate`
    are of the same shape as the outputs of `.data_set()`."""
    return [
        {"key": list(p.keys())[0], "val": list(p.values())[0], "indices": i}
        for p, i in zip(params, indices_set_by_trainables)
    ]


def convert_point_process_to_distributed(current: ArrayLike, area: ArrayLike) -> Array:
    """Convert current point process (nA) to distributed current (uA/cm2).

    This function gets called for synapses and for external stimuli.

    Args:
        current: Current in `nA`.
        area: Membrane surface area radius in `um^2`.

    Return:
        Current in `uA/cm2`.
    """
    current /= area  # nA / um^2
    return current * 100_000  # Convert (nA / um^2) to (uA / cm^2)


def query_channel_states_and_params(d, keys, idcs):
    """Get dict with subset of keys and values from d.

    This is used to restrict a dict where every item contains __all__ states to only
    the ones that are relevant for the channel. E.g.

    ```states = {'eCa': Array([ 0.,  0., nan]}```

    will be
    ```states = {'eCa': Array([ 0.,  0.]}```

    Only loops over necessary keys, as opposed to looping over `d.items()`."""
    return dict(zip(keys, (v[idcs] for v in map(d.get, keys))))


def compute_children_and_parents(
    branch_edges: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build indices used during `._init_morph_custom_spsolve()."""
    par_inds = branch_edges["parent_branch_index"].to_numpy()
    child_inds = branch_edges["child_branch_index"].to_numpy()
    child_belongs_to_branchpoint = remap_to_consecutive(par_inds)
    par_inds = np.unique(par_inds)
    return par_inds, child_inds, child_belongs_to_branchpoint


def _get_comp_edges_in_view(
    comp_edges: pd.DataFrame, incl_comps, comp_edge_condition: str
):
    """Return the `_comp_edges` within a current `View`.

    Args:
        base_comp_edges: `comp_edges` to be filtered.
        incl_comps: Sequence of compartments that are within the current `View`.
        comp_edge_condition: Either of
            {`source_and_sink`, `source_or_sink`, `endpoint`, `startpoint`}. Sets
            how the `comp_edges` are built. If `source_and_sink`, an edge between
            compartments is kept only if source and sink compartments are within
            the view. If `source_or_sink`, an edge is kept if either the source
            or the sink are within the view. If `endpoint`, then the edge is kept
            if the compartment is in source or sink and if it is an edge between
            parent compartment and branchpoint. If `startpoint`, then the edge is
            kept if the compartment is in source or sink and if it is an edge
            between child compartment and branchpoint. This is used because we
            want different treatment of the `comp_edges` depending on whether we
            index with `.branch()` (`source_or_sink`), `.comp()`
            (`source_and_sink`), `.loc(0.0)` (`startpoint`), or `.loc(1.0)`
            (`endpoint`).

    Returns:
        A Sequence of indices of the `comp_edges` that are within the `View`.
    """
    pre = comp_edges["source"].isin(incl_comps).to_numpy()
    post = comp_edges["sink"].isin(incl_comps).to_numpy()
    comp_edge_inds = comp_edges.index.to_numpy()
    if comp_edge_condition == "source_and_sink":
        possible_edges_in_view = comp_edge_inds[(pre & post).flatten()]
    elif comp_edge_condition == "source_or_sink":
        possible_edges_in_view = comp_edge_inds[(pre | post).flatten()]
    elif comp_edge_condition == "startpoint":
        # Type 2 and 4 are comp_edges between branchpoints and the
        # child compartments.
        is_child = comp_edges["type"].isin([2, 4]).to_numpy()
        possible_edges_in_view = comp_edge_inds[((pre | post) & is_child).flatten()]
    elif comp_edge_condition == "endpoint":
        # Type 2 and 4 are comp_edges between branchpoints and the
        # parent compartments.
        is_parent = comp_edges["type"].isin([1, 3]).to_numpy()
        possible_edges_in_view = comp_edge_inds[((pre | post) & is_parent).flatten()]
    else:
        raise ValueError(
            f"comp_edge_condition is {comp_edge_condition}, but must be in "
            "{source_and_sink, source_or_sink, startpoint, endpoint}."
        )
    return possible_edges_in_view
