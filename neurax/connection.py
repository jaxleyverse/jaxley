from neurax.utils.syn_utils import prepare_presyn, prepare_postsyn


class Connection:
    """A simple wrapper to save all elements that are important for a single synapse.
    """
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
    """Given a list of all synapses, this prepares everything for simulation.

    There are two main functions of this class:
    (1) For presynaptic locations, we infer the index of the exact location on a 
    single neuron.
    (2) For postsyaptic locations, we also cluster locations such that we can allow
    multiple synaptic contances onto the same postsynaptic compartment.
    """
    def __init__(self, conns, nseg_per_branch):
        pre_syn = prepare_presyn(conns, nseg_per_branch)
        self.pre_syn_cell_inds = pre_syn[0]
        self.pre_syn_inds = pre_syn[1]
        self.init_syn_states = pre_syn[2]

        post_syn = prepare_postsyn(conns, nseg_per_branch)
        self.grouped_post_syn_inds = post_syn[0]
        self.grouped_post_syns = post_syn[1]
