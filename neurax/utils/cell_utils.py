import numpy as np


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
            return rad1 * rad2**2 / r_a / (rad2**2 * l1 + rad1**2 * l2) / l1