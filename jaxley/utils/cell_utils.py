# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from functools import partial
from math import pi
from typing import Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import jit, vmap

from jaxley.utils.misc_utils import cumsum_leading_zero


def _split_into_branches_and_sort(
    content: np.ndarray,
    max_branch_len: float,
    is_single_point_soma: bool,
    sort: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    branches, types = _split_into_branches(content, is_single_point_soma)
    branches, types = _split_long_branches(
        branches,
        types,
        content,
        max_branch_len,
        is_single_point_soma=is_single_point_soma,
    )

    if sort:
        first_val = np.asarray([b[0] for b in branches])
        sorting = np.argsort(first_val, kind="mergesort")
        sorted_branches = [branches[s] for s in sorting]
        sorted_types = [types[s] for s in sorting]
    else:
        sorted_branches = branches
        sorted_types = types
    return sorted_branches, sorted_types


def _split_long_branches(
    branches: np.ndarray,
    types: np.ndarray,
    content: np.ndarray,
    max_branch_len: float,
    is_single_point_soma: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    pathlengths = _compute_pathlengths(
        branches, content[:, 1:6], is_single_point_soma=is_single_point_soma
    )
    pathlengths = [np.sum(length_traced) for length_traced in pathlengths]
    split_branches = []
    split_types = []
    for branch, type, length in zip(branches, types, pathlengths):
        num_subbranches = 1
        split_branch = [branch]
        while length > max_branch_len:
            num_subbranches += 1
            split_branch = _split_branch_equally(branch, num_subbranches)
            lengths_of_subbranches = _compute_pathlengths(
                split_branch,
                coords=content[:, 1:6],
                is_single_point_soma=is_single_point_soma,
            )
            lengths_of_subbranches = [
                np.sum(length_traced) for length_traced in lengths_of_subbranches
            ]
            length = max(lengths_of_subbranches)
            if num_subbranches > 10:
                warn(
                    """`num_subbranches > 10`, stopping to split. Most likely your
                     SWC reconstruction is not dense and some neighbouring traced
                     points are farther than `max_branch_len` apart."""
                )
                break
        split_branches += split_branch
        split_types += [type] * num_subbranches

    return split_branches, split_types


def _split_branch_equally(branch: np.ndarray, num_subbranches: int) -> List[np.ndarray]:
    num_points_each = len(branch) // num_subbranches
    branches = [branch[:num_points_each]]
    for i in range(1, num_subbranches - 1):
        branches.append(branch[i * num_points_each - 1 : (i + 1) * num_points_each])
    branches.append(branch[(num_subbranches - 1) * num_points_each - 1 :])
    return branches


def _split_into_branches(
    content: np.ndarray, is_single_point_soma: bool
) -> Tuple[np.ndarray, np.ndarray]:
    prev_ind = None
    prev_type = None
    n_branches = 0

    # Branch inds will contain the row identifier at which a branch point occurs
    # (i.e. the row of the parent of two branches).
    branch_inds = []
    for c in content:
        current_ind = c[0]
        current_parent = c[-1]
        current_type = c[1]
        if current_parent != prev_ind or current_type != prev_type:
            branch_inds.append(int(current_parent))
            n_branches += 1
        prev_ind = current_ind
        prev_type = current_type

    all_branches = []
    current_branch = []
    all_types = []

    # Loop over every line in the SWC file.
    for c in content:
        current_ind = c[0]  # First col is row_identifier
        current_parent = c[-1]  # Last col is parent in SWC specification.
        if current_parent == -1:
            all_types.append(c[1])
        else:
            current_type = c[1]

        if current_parent == -1 and is_single_point_soma and current_ind == 1:
            all_branches.append([int(current_ind)])
            all_types.append(int(current_type))

        # Either append the current point to the branch, or add the branch to
        # `all_branches`.
        if current_parent in branch_inds[1:]:
            if len(current_branch) > 1:
                all_branches.append(current_branch)
                all_types.append(current_type)
            current_branch = [int(current_parent), int(current_ind)]
        else:
            current_branch.append(int(current_ind))

    # Append the final branch (intermediate branches are already appended five lines
    # above.)
    all_branches.append(current_branch)
    return all_branches, all_types


def _build_parents(all_branches: List[np.ndarray]) -> List[int]:
    parents = [None] * len(all_branches)
    all_last_inds = [b[-1] for b in all_branches]
    for i, branch in enumerate(all_branches):
        parent_ind = branch[0]
        ind = np.where(np.asarray(all_last_inds) == parent_ind)[0]
        if len(ind) > 0 and ind != i:
            parents[i] = ind[0]
        else:
            assert (
                parent_ind == 1
            ), """Trying to connect a segment to the beginning of 
            another segment. This is not allowed. Please create an issue on github."""
            parents[i] = -1

    return parents


def _radius_generating_fns(
    all_branches: np.ndarray,
    radiuses: np.ndarray,
    each_length: np.ndarray,
    parents: np.ndarray,
    types: np.ndarray,
) -> List[Callable]:
    """For all branches in a cell, returns callable that return radius given loc."""
    radius_fns = []
    for i, branch in enumerate(all_branches):
        rads_in_branch = radiuses[np.asarray(branch) - 1]
        if parents[i] > -1 and types[i] != types[parents[i]]:
            # We do not want to linearly interpolate between the radius of the previous
            # branch if a new type of neurite is found (e.g. switch from soma to
            # apical). From looking at the SWC from n140.swc I believe that this is
            # also what NEURON does.
            rads_in_branch[0] = rads_in_branch[1]
        radius_fn = _radius_generating_fn(
            radiuses=rads_in_branch, each_length=each_length[i]
        )
        # Beause SWC starts counting at 1, but numpy counts from 0.
        # ind_of_branch_endpoint = np.asarray(b) - 1
        radius_fns.append(radius_fn)
    return radius_fns


def _padded_radius(loc: float, radiuses: np.ndarray) -> float:
    return radiuses * np.ones_like(loc)


def _radius(loc: float, cutoffs: np.ndarray, radiuses: np.ndarray) -> float:
    """Function which returns the radius via linear interpolation.

    Defined outside of `_radius_generating_fns` to allow for pickling of the resulting
    Cell object."""
    index = np.digitize(loc, cutoffs, right=False)
    left_rad = radiuses[index - 1]
    right_rad = radiuses[index]
    left_loc = cutoffs[index - 1]
    right_loc = cutoffs[index]
    loc_within_bin = (loc - left_loc) / (right_loc - left_loc)
    return left_rad + (right_rad - left_rad) * loc_within_bin


def _padded_radius_generating_fn(radiuses: np.ndarray) -> Callable:
    return partial(_padded_radius, radiuses=radiuses)


def _radius_generating_fn(radiuses: np.ndarray, each_length: np.ndarray) -> Callable:
    # Avoid division by 0 with the `summed_len` below.
    each_length[each_length < 1e-8] = 1e-8
    summed_len = np.sum(each_length)
    cutoffs = np.cumsum(np.concatenate([np.asarray([0]), each_length])) / summed_len
    cutoffs[0] -= 1e-8
    cutoffs[-1] += 1e-8

    # We have to linearly interpolate radiuses, therefore we need at least two radiuses.
    # However, jaxley allows somata which consist of a single traced point (i.e.
    # just one radius). Therefore, we just `tile` in order to generate an artificial
    # endpoint and startpoint radius of the soma.
    if len(radiuses) == 1:
        radiuses = np.tile(radiuses, 2)

    return partial(_radius, cutoffs=cutoffs, radiuses=radiuses)


def _compute_pathlengths(
    all_branches: np.ndarray, coords: np.ndarray, is_single_point_soma: bool
) -> List[np.ndarray]:
    """
    Args:
        coords: Has shape (num_traced_points, 5), where `5` is (type, x, y, z, radius).
    """
    branch_pathlengths = []
    for b in all_branches:
        coords_in_branch = coords[np.asarray(b) - 1]
        if len(coords_in_branch) > 1:
            # If the branch starts at a different neurite (e.g. the soma) then NEURON
            # ignores the distance from that initial point. To reproduce, use the
            # following SWC dummy file and read it in NEURON (and Jaxley):
            # 1 1 0.00 0.0 0.0 6.0 -1
            # 2 2 9.00 0.0 0.0 0.5 1
            # 3 2 10.0 0.0 0.0 0.3 2
            types = coords_in_branch[:, 0]
            if int(types[0]) == 1 and int(types[1]) != 1 and is_single_point_soma:
                coords_in_branch[0] = coords_in_branch[1]

            # Compute distances between all traced points in a branch.
            point_diffs = np.diff(coords_in_branch, axis=0)
            dists = np.sqrt(
                point_diffs[:, 1] ** 2 + point_diffs[:, 2] ** 2 + point_diffs[:, 3] ** 2
            )
        else:
            # Jaxley uses length and radius for every compartment and assumes the
            # surface area to be 2*pi*r*length. For branches consisting of a single
            # traced point we assume for them to have area 4*pi*r*r. Therefore, we have
            # to set length = 2*r.
            radius = coords_in_branch[0, 4]  # txyzr -> 4 is radius.
            dists = np.asarray([2 * radius])
        branch_pathlengths.append(dists)
    return branch_pathlengths


def build_radiuses_from_xyzr(
    radius_fns: List[Callable],
    branch_indices: List[int],
    min_radius: Optional[float],
    ncomp: int,
) -> jnp.ndarray:
    """Return the radiuses of branches given SWC file xyzr.

    Returns an array of shape `(num_branches, ncomp)`.

    Args:
        radius_fns: Functions which, given compartment locations return the radius.
        branch_indices: The indices of the branches for which to return the radiuses.
        min_radius: If passed, the radiuses are clipped to be at least as large.
        ncomp: The number of compartments that every branch is discretized into.
    """
    # Compartment locations are at the center of the internal nodes.
    non_split = 1 / ncomp
    range_ = np.linspace(non_split / 2, 1 - non_split / 2, ncomp)

    # Build radiuses.
    radiuses = np.asarray([radius_fns[b](range_) for b in branch_indices])
    radiuses_each = radiuses.ravel(order="C")
    if min_radius is None:
        assert np.all(
            radiuses_each > 0.0
        ), "Radius 0.0 in SWC file. Set `read_swc(..., min_radius=...)`."
    else:
        radiuses_each[radiuses_each < min_radius] = min_radius

    return radiuses_each


def equal_segments(branch_property: list, ncomp_per_branch: int):
    """Generates segments where some property is the same in each segment.

    Args:
        branch_property: List of values of the property in each branch. Should have
            `len(branch_property) == num_branches`.
    """
    assert isinstance(branch_property, list), "branch_property must be a list."
    return jnp.asarray([branch_property] * ncomp_per_branch).T


def linear_segments(
    initial_val: float, endpoint_vals: list, parents: jnp.ndarray, ncomp_per_branch: int
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


def compute_parents_in_level(levels, par_inds, parents_row_and_col):
    level_of_parent = levels[par_inds]
    parents_in_each_level = []
    for l in range(np.max(levels)):
        parents_inds_in_current_level = np.where(level_of_parent == l)[0]
        parents_in_current_level = parents_row_and_col[parents_inds_in_current_level]
        parents_in_current_level = np.asarray(parents_in_current_level)
        parents_in_each_level.append(parents_in_current_level)
    return parents_in_each_level


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


def compute_children_indices(parents) -> List[jnp.ndarray]:
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
    num_children: jnp.ndarray,
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


def compute_coupling_cond(rad1, rad2, r_a1, r_a2, l1, l2):
    """Return the coupling conductance between two compartments.

    Equations taken from `https://en.wikipedia.org/wiki/Compartmental_neuron_models`.

    `radius`: um
    `r_a`: ohm cm
    `length_single_compartment`: um
    `coupling_conds`: S * um / cm / um^2 = S / cm / um -> *10**7 -> mS / cm^2
    """
    # Multiply by 10**7 to convert (S / cm / um) -> (mS / cm^2).
    return rad1 * rad2**2 / (r_a1 * rad2**2 * l1 + r_a2 * rad1**2 * l2) / l1 * 10**7


def compute_coupling_cond_branchpoint(rad, r_a, l):
    r"""Return the coupling conductance between one compartment and a comp with l=0.

    From https://en.wikipedia.org/wiki/Compartmental_neuron_models

    If one compartment has l=0.0 then the equations simplify.

    R_long = \sum_i r_a * L_i/2 / crosssection_i

    with crosssection = pi * r**2

    For a single compartment with L>0, this turns into:
    R_long = r_a * L/2 / crosssection

    Then, g_long = crosssection * 2 / L / r_a

    Then, the effective conductance is g_long / zylinder_area. So:
    g = pi * r**2 * 2 / L / r_a / 2 / pi / r / L
    g = r / r_a / L**2
    """
    return rad / r_a / l**2 * 10**7  # Convert (S / cm / um) -> (mS / cm^2)


def compute_impact_on_node(rad, r_a, l):
    r"""Compute the weight with which a compartment influences its node.

    In order to satisfy Kirchhoffs current law, the current at a branch point must be
    proportional to the crosssection of the compartment. We only require proportionality
    here because the branch point equation reads:
    `g_1 * (V_1 - V_b) + g_2 * (V_2 - V_b) = 0.0`

    Because R_long = r_a * L/2 / crosssection, we get
    g_long = crosssection * 2 / L / r_a \propto rad**2 / L / r_a

    This equation can be multiplied by any constant."""
    return rad**2 / r_a / l


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
    pathlens = np.insert(np.cumsum(dl), 0, 0)  # cummulative length of sections
    norm_pathlens = pathlens / np.maximum(1e-8, pathlens[-1])  # norm lengths to [0,1].

    return v_interp(loc, norm_pathlens, coords)


def params_to_pstate(
    params: List[Dict[str, jnp.ndarray]],
    indices_set_by_trainables: List[jnp.ndarray],
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


def convert_point_process_to_distributed(
    current: jnp.ndarray, radius: jnp.ndarray, length: jnp.ndarray
) -> jnp.ndarray:
    """Convert current point process (nA) to distributed current (uA/cm2).

    This function gets called for synapses and for external stimuli.

    Args:
        current: Current in `nA`.
        radius: Compartment radius in `um`.
        length: Compartment length in `um`.

    Return:
        Current in `uA/cm2`.
    """
    area = 2 * pi * radius * length
    current /= area  # nA / um^2
    return current * 100_000  # Convert (nA / um^2) to (uA / cm^2)


def build_branchpoint_group_inds(
    num_branchpoints, child_belongs_to_branchpoint, start_ind_for_branchpoints
):
    branchpoint_inds_parents = start_ind_for_branchpoints + jnp.arange(num_branchpoints)
    branchpoint_inds_children = (
        start_ind_for_branchpoints + child_belongs_to_branchpoint
    )

    all_branchpoint_inds = jnp.concatenate(
        [branchpoint_inds_parents, branchpoint_inds_children]
    )
    branchpoint_group_inds = remap_to_consecutive(all_branchpoint_inds)
    return branchpoint_group_inds


def compute_morphology_indices_in_levels(
    num_branchpoints,
    child_belongs_to_branchpoint,
    par_inds,
    child_inds,
):
    """Return (row, col) to build the sparse matrix defining the voltage eqs.

    This is run at `init`, not during runtime.
    """
    branchpoint_inds_parents = jnp.arange(num_branchpoints)
    branchpoint_inds_children = child_belongs_to_branchpoint
    branch_inds_parents = par_inds
    branch_inds_children = child_inds

    children = jnp.stack([branch_inds_children, branchpoint_inds_children])
    parents = jnp.stack([branch_inds_parents, branchpoint_inds_parents])

    return {"children": children.T, "parents": parents.T}


def group_and_sum(
    values_to_sum: jnp.ndarray, inds_to_group_by: jnp.ndarray, num_branchpoints: int
) -> jnp.ndarray:
    """Group values by whether they have the same integer and sum values within group.

    This is used to construct the last diagonals at the branch points.

    Written by ChatGPT.
    """
    # Initialize an array to hold the sum of each group
    group_sums = jnp.zeros(num_branchpoints)

    # `.at[inds]` requires that `inds` is not empty, so we need an if-case here.
    # `len(inds) == 0` is the case for branches and compartments.
    if num_branchpoints > 0:
        group_sums = group_sums.at[inds_to_group_by].add(values_to_sum)

    return group_sums


def query_channel_states_and_params(d, keys, idcs):
    """Get dict with subset of keys and values from d.

    This is used to restrict a dict where every item contains __all__ states to only
    the ones that are relevant for the channel. E.g.

    ```states = {'eCa': Array([ 0.,  0., nan]}```

    will be
    ```states = {'eCa': Array([ 0.,  0.]}```

    Only loops over necessary keys, as opposed to looping over `d.items()`."""
    return dict(zip(keys, (v[idcs] for v in map(d.get, keys))))


def compute_axial_conductances(
    comp_edges: pd.DataFrame, params: Dict[str, jnp.ndarray]
) -> jnp.ndarray:
    """Given `comp_edges`, radius, length, r_a, cm, compute the axial conductances.

    Note that the resulting axial conductances will already by divided by the
    capacitance `cm`.
    """
    # `Compartment-to-compartment` (c2c) axial coupling conductances.
    condition = comp_edges["type"].to_numpy() == 0
    source_comp_inds = np.asarray(comp_edges[condition]["source"].to_list())
    sink_comp_inds = np.asarray(comp_edges[condition]["sink"].to_list())

    if len(sink_comp_inds) > 0:
        conds_c2c = (
            vmap(compute_coupling_cond, in_axes=(0, 0, 0, 0, 0, 0))(
                params["radius"][sink_comp_inds],
                params["radius"][source_comp_inds],
                params["axial_resistivity"][sink_comp_inds],
                params["axial_resistivity"][source_comp_inds],
                params["length"][sink_comp_inds],
                params["length"][source_comp_inds],
            )
            / params["capacitance"][sink_comp_inds]
        )
    else:
        conds_c2c = jnp.asarray([])

    # `branchpoint-to-compartment` (bp2c) axial coupling conductances.
    condition = comp_edges["type"].isin([1, 2])
    sink_comp_inds = np.asarray(comp_edges[condition]["sink"].to_list())

    if len(sink_comp_inds) > 0:
        conds_bp2c = (
            vmap(compute_coupling_cond_branchpoint, in_axes=(0, 0, 0))(
                params["radius"][sink_comp_inds],
                params["axial_resistivity"][sink_comp_inds],
                params["length"][sink_comp_inds],
            )
            / params["capacitance"][sink_comp_inds]
        )
    else:
        conds_bp2c = jnp.asarray([])

    # `compartment-to-branchpoint` (c2bp) axial coupling conductances.
    condition = comp_edges["type"].isin([3, 4])
    source_comp_inds = np.asarray(comp_edges[condition]["source"].to_list())

    if len(source_comp_inds) > 0:
        conds_c2bp = vmap(compute_impact_on_node, in_axes=(0, 0, 0))(
            params["radius"][source_comp_inds],
            params["axial_resistivity"][source_comp_inds],
            params["length"][source_comp_inds],
        )
        # For numerical stability. These values are very small, but their scale
        # does not matter.
        conds_c2bp *= 1_000
    else:
        conds_c2bp = jnp.asarray([])

    # All axial coupling conductances.
    return jnp.concatenate([conds_c2c, conds_bp2c, conds_c2bp])


def compute_children_and_parents(
    branch_edges: pd.DataFrame,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Build indices used during `._init_morph_custom_spsolve()."""
    par_inds = branch_edges["parent_branch_index"].to_numpy()
    child_inds = branch_edges["child_branch_index"].to_numpy()
    child_belongs_to_branchpoint = remap_to_consecutive(par_inds)
    par_inds = np.unique(par_inds)
    return par_inds, child_inds, child_belongs_to_branchpoint
