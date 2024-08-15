from math import pi
from typing import Dict, List, Optional, Union

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


def interpolate_xyz(loc: float, coords: np.ndarray):
    """Perform a linear interpolation between xyz-coordinates.

    Args:
        loc: The location in [0,1] along the branch.
        coords: Array containing the reconstructed xyzr points of the branch.

    Return:
        Interpolated xyz coordinate at `loc`, shape `(3,).
    """
    return vmap(lambda x: jnp.interp(loc, jnp.linspace(0, 1, len(x)), x), in_axes=(1,))(
        coords[:, :3]
    )


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
    num_branchpoints,
    child_belongs_to_branchpoint,
    nseg,
    nbranches,
):
    start_ind_for_branchpoints = nseg * nbranches
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


v_interp = jit(vmap(jnp.interp, in_axes=(None, None, 1)))
