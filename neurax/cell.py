import numpy as np
import jax.numpy as jnp


class Cell:
    def __init__(
        self,
        num_branches,
        parents,
        nseg_per_branch,
        lengths,
        radiuses,
        r_a,
    ):
        self.num_branches = num_branches
        self.parents = parents
        self.r_a = r_a

        self.nseg_per_branch = nseg_per_branch

        self.lengths = lengths / nseg_per_branch
        self.radiuses = radiuses

        def compute_coupling_cond(rad1, rad2, r_a, l1, l2):
            return rad1 * rad2**2 / r_a / l1 / (rad2**2 * l1 + rad1**2 * l2)

        # Compute coupling conductance for segments within a branch.
        # `radius`: um
        # `r_a`: ohm cm
        # `length_single_compartment`: um
        # `coupling_conds`: S * um / cm / um^2 = S / cm / um
        rad1 = self.radiuses[:, 1:]
        rad2 = self.radiuses[:, :-1]
        l1 = self.lengths[:, 1:]
        l2 = self.lengths[:, :-1]
        self.coupling_conds_fwd = compute_coupling_cond(rad2, rad1, self.r_a, l2, l1)
        self.coupling_conds_bwd = compute_coupling_cond(rad1, rad2, self.r_a, l1, l2)

        # Compute coupling conductance for segments at branch points.
        rad1 = self.radiuses[jnp.arange(1, num_branches), -1]
        rad2 = self.radiuses[parents[jnp.arange(1, num_branches)], 0]
        l1 = self.lengths[jnp.arange(1, num_branches), -1]
        l2 = self.lengths[parents[jnp.arange(1, num_branches)], 0]
        self.branch_conds_fwd = compute_coupling_cond(rad2, rad1, self.r_a, l2, l1)
        self.branch_conds_bwd = compute_coupling_cond(rad1, rad2, self.r_a, l1, l2)

        # Convert (S / cm / um) -> (mS / cm^2)
        self.coupling_conds_fwd *= 10**7
        self.coupling_conds_bwd *= 10**7
        self.branch_conds_fwd *= 10**7
        self.branch_conds_bwd *= 10**7

        # Compute the summed coupling conductances of each compartment.
        self.summed_coupling_conds = jnp.zeros((num_branches, nseg_per_branch))
        self.summed_coupling_conds = self.summed_coupling_conds.at[:, 1:].add(
            self.coupling_conds_fwd
        )
        self.summed_coupling_conds = self.summed_coupling_conds.at[:, :-1].add(
            self.coupling_conds_bwd
        )
        for b in range(1, num_branches):
            self.summed_coupling_conds = self.summed_coupling_conds.at[b, -1].add(
                self.branch_conds_fwd[b]
            )
            self.summed_coupling_conds = self.summed_coupling_conds.at[
                parents[b], 0
            ].add(self.branch_conds_bwd[b])

        self.num_kids = jnp.asarray(_compute_num_kids(self.parents))
        self.levels = compute_levels(self.parents)
        self.branches_in_each_level = compute_branches_in_level(self.levels)

        self.parents_in_each_level = [
            jnp.unique(parents[c]) for c in self.branches_in_each_level
        ]


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


def _compute_num_kids(parents):
    num_branches = len(parents)
    num_kids = []
    for b in range(num_branches):
        n = np.sum(np.asarray(parents) == b)
        num_kids.append(n)
    return num_kids


def _compute_index_of_kid(parents):
    num_branches = len(parents)
    current_num_kids_for_each_branch = np.zeros((num_branches,), np.dtype("int"))
    index_of_kid = [None]
    for b in range(1, num_branches):
        index_of_kid.append(current_num_kids_for_each_branch[parents[b]])
        current_num_kids_for_each_branch[parents[b]] += 1
    return index_of_kid


def get_num_neighbours(
    num_kids: jnp.ndarray,
    nseg_per_branch: int,
    num_branches: int,
):
    """
    Number of neighbours of each compartment.
    """
    num_neighbours = 2 * jnp.ones((num_branches * nseg_per_branch))
    num_neighbours = num_neighbours.at[nseg_per_branch - 1].set(1.0)
    num_neighbours = num_neighbours.at[jnp.arange(num_branches) * nseg_per_branch].set(
        num_kids + 1.0
    )
    return num_neighbours
