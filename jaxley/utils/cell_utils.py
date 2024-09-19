# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from math import pi
from typing import Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import jit, vmap


def equal_segments(branch_property: list, nseg_per_branch: int):
    """Generates segments where some property is the same in each segment.

    Args:
        branch_property: List of values of the property in each branch. Should have
            `len(branch_property) == num_branches`.
    """
    assert isinstance(branch_property, list), "branch_property must be a list."
    return jnp.asarray([branch_property] * nseg_per_branch).T


def linear_segments(
    initial_val: float, endpoint_vals: list, parents: jnp.ndarray, nseg_per_branch: int
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

    branch_inds_of_each_comp = jnp.tile(jnp.arange(num_branches), nseg_per_branch)
    locs_of_each_comp = jnp.linspace(1, 0, nseg_per_branch).repeat(num_branches)
    rad_of_each_comp = compute_rad(branch_inds_of_each_comp, locs_of_each_comp)

    return jnp.reshape(rad_of_each_comp, (nseg_per_branch, num_branches)).T


def merge_cells(
    cumsum_num_branches: List[int],
    cumsum_num_branchpoints: List[int],
    arrs: List[List[jnp.ndarray]],
    exclude_first: bool = True,
) -> jnp.ndarray:
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
                + jnp.asarray([cumsum_num_branches[i], cumsum_num_branchpoints[i]])
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
        combined_parents_in_level.append(jnp.concatenate(current_ps))

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
    levels: jnp.ndarray, children_row_and_col: jnp.ndarray
) -> List[jnp.ndarray]:
    num_branches = len(levels)
    children_in_each_level = []
    for l in range(1, np.max(levels) + 1):
        children_in_current_level = []
        for b in range(num_branches):
            if levels[b] == l:
                children_in_current_level.append(children_row_and_col[b - 1])
        children_in_current_level = jnp.asarray(children_in_current_level)
        children_in_each_level.append(children_in_current_level)
    return children_in_each_level


def compute_parents_in_level(levels, par_inds, parents_row_and_col):
    level_of_parent = levels[par_inds]
    parents_in_each_level = []
    for l in range(np.max(levels)):
        parents_inds_in_current_level = jnp.where(level_of_parent == l)[0]
        parents_in_current_level = parents_row_and_col[parents_inds_in_current_level]
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
    nseg_per_branch: int,
    num_branches: int,
):
    """
    Number of neighbours of each compartment.
    """
    num_neighbours = 2 * jnp.ones((num_branches * nseg_per_branch))
    num_neighbours = num_neighbours.at[nseg_per_branch - 1].set(1.0)
    num_neighbours = num_neighbours.at[jnp.arange(num_branches) * nseg_per_branch].set(
        num_children + 1.0
    )
    return num_neighbours


def index_of_loc(branch_ind: int, loc: float, nseg_per_branch: int) -> int:
    """Returns the index of a segment given a loc in [0, 1] and the index of a branch.

    This is used because we specify locations such as synapses as a value between 0 and
    1. We have to convert this onto a discrete segment here.

    Args:
        branch_ind: Index of the branch.
        loc: Location (in [0, 1]) along that branch.
        nseg_per_branch: Number of segments of each branch.

    Returns:
        The index of the compartment within the entire cell.
    """
    nseg = nseg_per_branch  # only for convenience.
    possible_locs = np.linspace(0.5 / nseg, 1 - 0.5 / nseg, nseg)
    ind_along_branch = np.argmin(np.abs(possible_locs - loc))
    return branch_ind * nseg + ind_along_branch


def loc_of_index(global_comp_index, nseg):
    """Return location corresponding to index."""
    index = global_comp_index % nseg
    possible_locs = np.linspace(0.5 / nseg, 1 - 0.5 / nseg, nseg)
    return possible_locs[index]


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


def interpolate_xyz(loc: float, coords: np.ndarray):
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

    return v_interp(loc, norm_pathlens, coords[:, :3])


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


def compute_axial_conductances(
    comp_edges: pd.DataFrame, params: Dict[str, jnp.ndarray]
) -> jnp.ndarray:
    """Given `comp_edges`, radius, length, r_a, compute the axial conductances."""
    # `Compartment-to-compartment` (c2c) conductances.
    condition = comp_edges["type"].to_numpy() == 0
    source_comp_inds = np.asarray(comp_edges[condition]["source"].to_list())
    sink_comp_inds = np.asarray(comp_edges[condition]["sink"].to_list())

    if len(sink_comp_inds) > 0:
        conds_c2c = vmap(compute_coupling_cond, in_axes=(0, 0, 0, 0, 0, 0))(
            params["radius"][sink_comp_inds],
            params["radius"][source_comp_inds],
            params["axial_resistivity"][sink_comp_inds],
            params["axial_resistivity"][source_comp_inds],
            params["length"][sink_comp_inds],
            params["length"][source_comp_inds],
        )
    else:
        conds_c2c = jnp.asarray([])

    # `branchpoint-to-compartment` (bp2c) conductances.
    condition = comp_edges["type"].to_numpy() == 1
    sink_comp_inds = np.asarray(comp_edges[condition]["sink"].to_list())

    if len(sink_comp_inds) > 0:
        conds_bp2c = vmap(compute_coupling_cond_branchpoint, in_axes=(0, 0, 0))(
            params["radius"][sink_comp_inds],
            params["axial_resistivity"][sink_comp_inds],
            params["length"][sink_comp_inds],
        )
    else:
        conds_bp2c = jnp.asarray([])

    # `compartment-to-branchpoint` (c2bp) conductances.
    condition = comp_edges["type"].to_numpy() == 2
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

    # All conductances.
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
