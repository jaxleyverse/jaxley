import jax.numpy as jnp
import numpy as np


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


def merge_cells(cumsum_num_branches, arrs, exclude_first=True):
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


def compute_levels(parents):
    levels = np.zeros_like(parents)

    for i, p in enumerate(parents):
        if p == -1:
            levels[i] = 0
        else:
            levels[i] = levels[p] + 1
    return levels


def compute_branches_in_level(levels):
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


def _compute_num_children(parents):
    num_branches = len(parents)
    num_children = []
    for b in range(num_branches):
        n = np.sum(np.asarray(parents) == b)
        num_children.append(n)
    return num_children


def _compute_index_of_child(parents):
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
    if nseg_per_branch > 1:
        possible_locs = np.arange(nseg_per_branch) / (nseg_per_branch - 1)
        closest = np.argmin(np.abs(possible_locs - loc))
        ind_along_branch = nseg_per_branch - closest - 1
    else:
        ind_along_branch = 0
    return branch_ind * nseg_per_branch + ind_along_branch


def compute_coupling_cond(rad1, rad2, r_a, l1, l2):
    average_rad = 0.5 * (rad1 + rad2)
    average_ra = r_a
    return 1 * average_rad**2 / 2.0 / average_ra / rad1 / l1**2
    # return rad1 * rad2**2 / r_a / (rad2**2 * l1 + rad1**2 * l2) / l1
