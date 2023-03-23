import numpy as np


def index_of_loc(branch_ind, loc, nseg_per_branch):
    possible_locs = np.arange(nseg_per_branch) / (nseg_per_branch - 1)
    closest = np.argmin(np.abs(possible_locs - loc))
    ind_along_branch = nseg_per_branch - closest - 1
    return branch_ind * nseg_per_branch + ind_along_branch