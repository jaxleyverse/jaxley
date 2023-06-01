from typing import List

import numpy as np

from neurax.synapses import Synapse


class Connection:
    """A simple wrapper to save all elements that are important for a single synapse."""

    def __init__(
        self,
        pre_cell_ind,
        pre_branch_ind,
        pre_loc,
        post_cell_ind,
        post_branch_ind,
        post_loc,
    ):
        self.pre_cell_ind = pre_cell_ind
        self.pre_branch_ind = pre_branch_ind
        self.pre_loc = pre_loc
        self.post_cell_ind = post_cell_ind
        self.post_branch_ind = post_branch_ind
        self.post_loc = post_loc


class Connectivity:
    def __init__(self, synapse_type: Synapse, conns: List[Connection]):
        self.synapse_type = synapse_type
        self.conns = conns


class ConnectivityBuilder:
    """Helper to build layers of connectivity patterns."""

    def __init__(self, nbranches_per_submodule: List[int]):
        self.nbranches_per_submodule = nbranches_per_submodule

    def fc(self, pre_cell_inds, post_cell_inds):
        """Returns a list of `Connection`s which build a fully connected layer.

        Connections are from branch 0 location 0 to a randomly chosen branch and loc.
        """
        conns = []
        for pre_ind in pre_cell_inds:
            for post_ind in post_cell_inds:
                num_branches_post = self.nbranches_per_submodule[post_ind]
                rand_branch = np.random.randint(0, num_branches_post)
                rand_loc = np.random.rand()
                conns.append(Connection(pre_ind, 0, 0, post_ind, rand_branch, rand_loc))
        return conns
