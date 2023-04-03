import numpy as np
import jax.numpy as jnp


class Cell:
    def __init__(self, num_branches, parents, nseg_per_branch, length, radius, r_a):
        self.num_branches = num_branches
        self.parents = parents
        self.nseg_per_branch = nseg_per_branch
        self.length_single_compartment = length / nseg_per_branch
        self.radius = radius
        self.r_a = r_a

        # `radius`: um
        # `r_a`: ohm cm
        # `length_single_compartment`: um
        self.coupling_conds = (
            self.radius / 2.0 / self.r_a / self.length_single_compartment**2
        )  # S * um / cm / um^2 = S / cm / um
        self.coupling_conds *= 10**7  # Convert (S / cm / um) -> (mS / cm^2)
        self.num_kids = jnp.asarray(_compute_num_kids(self.parents))
        self.levels = compute_levels(self.parents)
        self.branches_in_each_level = compute_branches_in_level(self.levels)

        self.num_neighbours = get_num_neighbours(
            num_kids=self.num_kids,
            nseg_per_branch=self.nseg_per_branch,
            num_branches=self.num_branches,
        )
        self.parents_in_each_level = [
            jnp.unique(parents[c]) for c in self.branches_in_each_level
        ]


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
