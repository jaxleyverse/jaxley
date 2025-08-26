# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array, vmap
from jax.typing import ArrayLike

from jaxley.utils.misc_utils import cumsum_leading_zero


def surface_area_segments(ls, r1, r2, dr):
    """Surface area of truncated cone segments."""
    slant_height = np.sqrt(ls**2 + dr**2)
    return 2 * np.pi * (r1 + r2) / 2 * slant_height


def volume_segments(ls, r1, r2):
    """Volume of truncated cone segments."""
    return (np.pi * ls / 3) * (r1**2 + r1 * r2 + r2**2)


def resistive_load_segments(ls, r1, r2, dr):
    """Resistive load using truncated cone approximation."""
    segment_integrals = np.empty_like(ls)
    is_constant = np.isclose(dr, 0)

    # Constant-radius segments: simple cylinder
    segment_integrals[is_constant] = ls[is_constant] / (r1[is_constant] ** 2)

    # Varying-radius segments: truncated cone integral
    segment_integrals[~is_constant] = (
        ls[~is_constant]
        / dr[~is_constant]
        * (1 / r1[~is_constant] - 1 / r2[~is_constant])
    )

    return np.sum(segment_integrals) / np.pi


def average_radius(lengths, r1, r2):
    """Average radius weighted by segment length."""
    return np.sum((r1 + r2) / 2 * lengths) / np.sum(lengths)


def compute_cone_props(ls, rs, l_start=None, l_end=None):
    """
    Given positions ls and radii rs along a path, compute average radius,
    surface area, volume, resistive load at start and end segments.

    Returns:
        avg_r: float
        surface_area: float
        volume: float
        res_in: float
        res_out: float
    """
    # Handle default bounds
    l1 = ls[0] if l_start is None else l_start
    l2 = ls[-1] if l_end is None else l_end

    # TODO: Add handling spherical compartments? Check if this even needs a special case?

    assert l1 < l2, "Invalid integration bounds"

    # Select segment within bounds
    mask = (ls >= l1) & (ls <= l2)
    l_seg = ls[mask]
    r_seg = rs[mask]

    # Add boundary points via interpolation
    if l1 < l_seg[0]:
        r1 = np.interp(l1, ls, rs)
        l_seg = np.insert(l_seg, 0, l1)
        r_seg = np.insert(r_seg, 0, r1)

    if l2 > l_seg[-1]:
        r2 = np.interp(l2, ls, rs)
        l_seg = np.append(l_seg, l2)
        r_seg = np.append(r_seg, r2)

    # add midpoint
    l_mid = (l_seg[-1] + l_seg[0]) / 2
    r_mid = np.interp(l_mid, ls, rs)
    mid_idx = np.searchsorted(l_seg, l_mid)
    l_seg = np.insert(l_seg, mid_idx, l_mid)
    r_seg = np.insert(r_seg, mid_idx, r_mid)

    # Compute segment lengths and properties
    dl = np.diff(l_seg)
    r1s = r_seg[:-1]
    r2s = r_seg[1:]
    dr = r2s - r1s

    # Compute quantities
    surface_area = np.sum(surface_area_segments(dl, r1s, r2s, dr))
    volume = np.sum(volume_segments(dl, r1s, r2s))
    avg_r = average_radius(dl, r1s, r2s)
    res_in = resistive_load_segments(
        dl[:mid_idx], r1s[:mid_idx], r2s[:mid_idx], dr[:mid_idx]
    )
    res_out = resistive_load_segments(
        dl[mid_idx:], r1s[mid_idx:], r2s[mid_idx:], dr[mid_idx:]
    )

    return avg_r, surface_area, volume, res_in, res_out


def radius_from_xyzr(
    xyzr: np.ndarray,
    min_radius: Optional[float],
) -> float:
    """Return the radius of a compartment given its SWC file xyzr.

    Args:
        radius_fns: Functions which, given compartment locations return the radius.
        branch_indices: The indices of the branches for which to return the radiuses.
        min_radius: If passed, the radiuses are clipped to be at least as large.
        ncomp: The number of compartments that every branch is discretized into.
    """
    xyz = xyzr[:, :3]
    radius = xyzr[:, 3]
    if len(xyzr) > 1:
        deltas = np.diff(xyz, axis=0)
        dists = np.linalg.norm(deltas, axis=1)
        weights = np.zeros(len(dists) + 1)
        weights[1:] += dists
        weights[:-1] += dists
        weights /= np.sum(weights)
        avg_radius = np.sum(radius * weights)
    else:
        avg_radius = radius.mean()

    if min_radius is None:
        assert (
            avg_radius > 0.0
        ), "Radius 0.0 in SWC file. Set `read_swc(..., min_radius=...)`."
    else:
        avg_radius = (
            min_radius
            if (avg_radius < min_radius or np.isnan(avg_radius))
            else avg_radius
        )

    return avg_radius


def split_xyzr_into_equal_length_segments(
    xyzr: np.ndarray, ncomp: int
) -> List[np.ndarray]:
    """Split xyzr into equal-length segments by inserting interpolated points as needed.

    This function was written by ChatGPT, based on the prompt:
    ```I have an array of shape 100x3. The 3 indicate x, y, z coordinates. I want to
    split this array into 4 segments, each with equal euclidean length. To have
    euclidean length exactly equal, I would like to insert additional points into
    the 100x3 array (to make it length 100 + 4 segments - 1). These points should be
    linear interpolation of neighboring points. In the final split array, the newly
    inserted nodes should be the last point of one segment and the first point of
    another segment.```

    Args:
        points: Array of 3D coordinates representing a path.
        num_segments: Number of segments to split the path into.

    Returns:
        A list of `num_segments` arrays, each containing the 3D coordinates
        of one segment. The segments have (approximately) equal Euclidean
        length, and split points are interpolated between original points.
    """
    if len(xyzr) == 1:
        return [xyzr] * ncomp

    # Compute distances between consecutive points
    xyz = xyzr[:, :3]

    # Compute distances and cumulative distances
    deltas = np.diff(xyz, axis=0)
    dists = np.linalg.norm(deltas, axis=1)
    cum_dists = np.concatenate([[0], np.cumsum(dists)])
    total_length = cum_dists[-1]

    # Target cumulative distances where we want to split
    target_dists = np.linspace(0, total_length, ncomp + 1)

    # Find insertion indices and interpolation factors
    idxs = np.searchsorted(cum_dists, target_dists, side="right") - 1
    idxs = np.clip(idxs, 0, len(xyz) - 2)  # Ensure valid indices
    local_dist = target_dists - cum_dists[idxs]
    segment_lens = dists[idxs]
    frac = (local_dist / segment_lens)[:, None]  # shape (n, 1)

    # Interpolate split points
    split_points = xyzr[idxs] + frac * (xyzr[idxs + 1] - xyzr[idxs])

    # Build final list of points with inserted nodes
    all_points = [split_points[0]]
    compartment_xyzrs = []

    for i in range(1, len(split_points)):
        # Collect original points between splits.
        mask = (cum_dists > target_dists[i - 1]) & (cum_dists < target_dists[i])
        between_points = xyzr[mask]
        segment = np.vstack([all_points[-1], *between_points, split_points[i]])
        compartment_xyzrs.append(segment)
        all_points.append(split_points[i])
    return compartment_xyzrs


def equal_segments(branch_property: list, ncomp_per_branch: int):
    """Generates segments where some property is the same in each segment.

    Args:
        branch_property: List of values of the property in each branch. Should have
            `len(branch_property) == num_branches`.
    """
    assert isinstance(branch_property, list), "branch_property must be a list."
    return jnp.asarray([branch_property] * ncomp_per_branch).T


def linear_segments(
    initial_val: float, endpoint_vals: list, parents: ArrayLike, ncomp_per_branch: int
):
    """Generates segments where some property is linearly interpolated.

    Args:
        initial_val: The value at the tip of the soma.
        endpoint_vals: The value at the endpoints of each branch.
    """
    branch_property = endpoint_vals + [initial_val]
    num_branches = len(parents)
    # Compute radiuses by linear interpolation.
    endpoint_radiuses = jnp.asarray(branch_property)

    def compute_rad(branch_ind, loc):
        start = endpoint_radiuses[parents[branch_ind]]
        end = endpoint_radiuses[branch_ind]
        return (end - start) * loc + start

    branch_inds_of_each_comp = jnp.tile(jnp.arange(num_branches), ncomp_per_branch)
    locs_of_each_comp = jnp.linspace(1, 0, ncomp_per_branch).repeat(num_branches)
    rad_of_each_comp = compute_rad(branch_inds_of_each_comp, locs_of_each_comp)

    return jnp.reshape(rad_of_each_comp, (ncomp_per_branch, num_branches)).T


def merge_cells(
    cumsum_num_branches: List[int],
    cumsum_num_branchpoints: List[int],
    arrs: List[List[np.ndarray]],
    exclude_first: bool = True,
) -> np.ndarray:
    """
    Build full list of which branches are solved in which iteration.

    From the branching pattern of single cells, this "merges" them into a single
    ordering of branches.

    Args:
        cumsum_num_branches: cumulative number of branches. E.g., for three cells with
            10, 15, and 5 branches respectively, this will should be a list containing
            `[0, 10, 25, 30]`.
        arrs: A list of a list of arrays that should be merged.
        exclude_first: If `True`, the first element of each list in `arrs` will remain
            unchanged. Useful if a `-1` (which indicates "no parent") entry should not
            be changed.

    Returns:
        A list of arrays which contain the branch indices that are computed at each
        level (i.e., iteration).
    """
    ps = []
    for i, att in enumerate(arrs):
        p = att
        if exclude_first:
            raise NotImplementedError
            p = [p[0]] + [p_in_level + cumsum_num_branches[i] for p_in_level in p[1:]]
        else:
            p = [
                p_in_level
                + np.asarray([cumsum_num_branches[i], cumsum_num_branchpoints[i]])
                for p_in_level in p
            ]
        ps.append(p)

    max_len = max([len(att) for att in arrs])
    combined_parents_in_level = []
    for i in range(max_len):
        current_ps = []
        for p in ps:
            if len(p) > i:
                current_ps.append(p[i])
        combined_parents_in_level.append(np.concatenate(current_ps))

    return combined_parents_in_level


def compute_levels(parents):
    levels = np.zeros_like(parents)

    for i, p in enumerate(parents):
        if p == -1:
            levels[i] = 0
        else:
            levels[i] = levels[p] + 1
    return levels


def compute_children_in_level(
    levels: np.ndarray, children_row_and_col: np.ndarray
) -> List[np.ndarray]:
    num_branches = len(levels)
    children_in_each_level = []
    for l in range(1, np.max(levels) + 1):
        children_in_current_level = []
        for b in range(num_branches):
            if levels[b] == l:
                children_in_current_level.append(children_row_and_col[b - 1])
        children_in_current_level = np.asarray(children_in_current_level)
        children_in_each_level.append(children_in_current_level)
    return children_in_each_level


def _compute_num_children(parents):
    num_branches = len(parents)
    num_children = []
    for b in range(num_branches):
        n = np.sum(np.asarray(parents) == b)
        num_children.append(n)
    return num_children


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


def compute_children_indices(parents) -> list[Array]:
    """Return all children indices of every branch.

    Example:
    ```
    parents = [-1, 0, 0]
    compute_children_indices(parents) -> [[1, 2], [], []]
    ```
    """
    num_branches = len(parents)
    child_indices = []
    for b in range(num_branches):
        child_indices.append(np.where(parents == b)[0])
    return child_indices


def get_num_neighbours(
    num_children: ArrayLike,
    ncomp_per_branch: int,
    num_branches: int,
):
    """
    Number of neighbours of each compartment.
    """
    num_neighbours = 2 * jnp.ones((num_branches * ncomp_per_branch))
    num_neighbours = num_neighbours.at[ncomp_per_branch - 1].set(1.0)
    num_neighbours = num_neighbours.at[jnp.arange(num_branches) * ncomp_per_branch].set(
        num_children + 1.0
    )
    return num_neighbours


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
    _, inverse_indices = jnp.unique(arr, return_inverse=True)
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
) -> tuple[Array, Array, Array, int]:
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
