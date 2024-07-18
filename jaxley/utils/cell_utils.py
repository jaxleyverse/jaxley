# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from math import pi
from typing import Dict, List, Optional, Union

import jax.numpy as jnp
import numpy as np
from jax import vmap


def equal_segments(branch_property: list, nseg_per_branch: int) -> jnp.ndarray:
    """Generates segments where some property is the same in each segment.

    Args:
        branch_property: List of values of the property in each branch. Should have
            `len(branch_property) == num_branches`.
    """
    assert isinstance(branch_property, list), "branch_property must be a list."
    return jnp.asarray([branch_property] * nseg_per_branch).T


def linear_segments(
    initial_val: float, endpoint_vals: list, parents: jnp.ndarray, nseg_per_branch: int
) -> jnp.ndarray:
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
    arrs: List[List[jnp.ndarray]],
    exclude_first: bool = True,
) -> jnp.ndarray:
    """
    Build full list of which branches are solved in which iteration.

    From the branching pattern of single cells, this "merges" them into a single
    ordering of branches.

    Args:
        cumsum_num_branches: cumulative number of branches. E.g., for three cells with
            10, 15, and 5 branches respectively, this will be a list containing
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
            p = [p[0]] + [p_in_level + cumsum_num_branches[i] for p_in_level in p[1:]]
        else:
            p = [p_in_level + cumsum_num_branches[i] for p_in_level in p]
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


def compute_levels(parents: jnp.ndarray) -> jnp.ndarray:
    """Compute the level of each branch in the tree."""
    levels = np.zeros_like(parents)

    for i, p in enumerate(parents):
        levels[i] = 0 if p == -1 else levels[p] + 1
    return levels


def compute_branches_in_level(levels: jnp.ndarray) -> List[jnp.ndarray]:
    """Collect the branches in each level of the tree."""
    num_branches = len(levels)
    branches_in_each_level = []
    for l in range(np.max(levels) + 1):
        branches_in_current_level = []
        for b in range(num_branches):
            if levels[b] == l:
                branches_in_current_level.append(b)
        branches_in_current_level = jnp.asarray(branches_in_current_level)
        branches_in_each_level.append(branches_in_current_level)
    return branches_in_each_level


def _compute_num_children(parents: jnp.ndarray) -> List[int]:
    """Compute the number of children of each branch."""
    num_branches = len(parents)
    num_children = []
    for b in range(num_branches):
        n = np.sum(np.asarray(parents) == b)
        num_children.append(n)
    return num_children


def _compute_index_of_child(parents: jnp.ndarray) -> List[int]:
    """Compute the index of each child of each branch."""
    num_branches = len(parents)
    current_num_children_for_each_branch = np.zeros((num_branches,), np.dtype("int"))
    index_of_child = [-1]
    for b in range(1, num_branches):
        index_of_child.append(current_num_children_for_each_branch[parents[b]])
        current_num_children_for_each_branch[parents[b]] += 1
    return index_of_child


def get_num_neighbours(
    num_children: jnp.ndarray,
    nseg_per_branch: int,
    num_branches: int,
) -> jnp.ndarray:
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


def loc_of_index(global_comp_index: int, nseg: int) -> float:
    """Return location corresponding to index."""
    index = global_comp_index % nseg
    possible_locs = np.linspace(0.5 / nseg, 1 - 0.5 / nseg, nseg)
    return possible_locs[index]


def flip_comp_indices(indices: np.ndarray, nseg: int) -> np.ndarray:
    """Flip ordering of compartments because the solver treats 0 as last compartment.

    E.g with nseg=8, this function will do:
    [2] -> [5]
    [13] -> [10] because this is the second branch (it only flips within branch).

    This is required to hide the weird compartment ordering from the user (#30) and is
    introduced in PR #305.
    """
    remainder = indices % nseg
    corrected_comp_ind = nseg - remainder - 1
    integer_division = indices // nseg * nseg
    return integer_division + corrected_comp_ind


def compute_coupling_cond(
    rad1: jnp.ndarray,
    rad2: jnp.ndarray,
    r_a1: jnp.ndarray,
    r_a2: jnp.ndarray,
    l1: jnp.ndarray,
    l2: jnp.ndarray,
) -> jnp.ndarray:
    midpoint_r_a = 0.5 * (r_a1 + r_a2)
    return rad1 * rad2**2 / midpoint_r_a / (rad2**2 * l1 + rad1**2 * l2) / l1
    # return midpoint_radius ** 2 / 2.0 / midpoint_axial_resistivity / rad1 / dx ** 2


def interpolate_xyz(loc: float, coords: np.ndarray) -> np.ndarray:
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
) -> List[Dict[str, jnp.ndarray]]:
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


def childview(
    module: "Module",
    index: Union[int, str, list, range, slice],
    child_name: Optional[str] = None,
) -> "View":
    """Return the child view of the current module.

    network.cell(index) at network level.
    cell.branch(index) at cell level.
    branch.comp(index) at branch level."""
    if child_name is None:
        parent_name = module.__class__.__name__.lower()
        views = np.array(["net", "cell", "branch", "comp", "/"])
        child_idx = np.roll([v in parent_name for v in views], 1)
        child_name = views[child_idx][0]
    if child_name != "/":
        return module.__getattr__(child_name)(index)
    raise AttributeError("Compartment does not support indexing")
