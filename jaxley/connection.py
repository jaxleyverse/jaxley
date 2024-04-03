from dataclasses import dataclass
from typing import List

import numpy as np

from jaxley.synapses import Synapse


@dataclass
class Connection:
    """A simple wrapper to save all elements that are important for a single synapse."""

    pre_cell_ind: int
    pre_branch_ind: int
    pre_loc: float
    post_cell_ind: int
    post_branch_ind: int
    post_loc: float


@dataclass
class Connectivity:
    synapse_type: "Synapse"
    conns: List[Connection]


def connect(pre: "CompartmentView", post: "CompartmentView", synapse_type: "Synapse"):
    """Connect two compartments with a chemical synapse.

    High-level strategy:

    We need to first check if the network already has a type of this synapse, else
    we need to register it as a new synapse in a bunch of dictionaries which track
    synapse parameters, state and meta information.

    Next, we register the new connection in the synapse dataframe (`.edges`).
    Then, we update synapse parameter and state arrays with the new connection.
    Finally, we update synapse meta information.
    """
    pre._append_multiple_synapses(pre.view, post.view, synapse_type)


def fully_connect(
    pre_cell_view: "CellView", post_cell_view: "CellView", synapse_type: "Synapse"
):
    """Appends multiple connections which build a fully connected layer.

    Connections are from branch 0 location 0 to a randomly chosen branch and loc.
    """
    # Get pre- and postsynaptic cell indices.
    pre_cell_inds = np.unique(pre_cell_view.view["cell_index"].to_numpy())
    post_cell_inds = np.unique(post_cell_view.view["cell_index"].to_numpy())
    nbranches_post = np.asarray(pre_cell_view.pointer.nbranches_per_cell)[
        post_cell_inds
    ]
    num_pre = len(pre_cell_inds)
    num_post = len(post_cell_inds)

    # Infer indices of (random) postsynaptic compartments.
    # Each row of `rand_branch_post` is an integer in `[0, nbranches_post[i] - 1]`.
    rand_branch_post = np.floor(np.random.rand(num_pre, num_post) * nbranches_post)
    rand_comp_post = np.floor(
        np.random.rand(num_pre, num_post) * pre_cell_view.pointer.nseg
    )
    global_post_indices = post_cell_view.pointer._local_inds_to_global(
        post_cell_inds, rand_branch_post, rand_comp_post
    )
    global_post_indices = global_post_indices.ravel()

    post_rows = post_cell_view.view.loc[global_post_indices]

    # Pre-synapse is at the zero-eth branch and zero-eth compartment.
    pre_rows = pre_cell_view[0, 0].view
    # Repeat rows `num_post` times. See SO 50788508.
    pre_rows = pre_rows.loc[pre_rows.index.repeat(num_post)].reset_index(drop=True)

    pre_cell_view._append_multiple_synapses(pre_rows, post_rows, synapse_type)


def sparse_connect(
    pre_cell_view: "CellView",
    post_cell_view: "CellView",
    synapse_type: "Synapse",
    p: float,
):
    """Returns a list of `Connection`s forming a sparse, randomly connected layer.

    Connections are from branch 0 location 0 to a randomly chosen branch and loc.
    """
    pre_cell_inds = np.unique(pre_cell_view.view["cell_index"].to_numpy())
    post_cell_inds = np.unique(post_cell_view.view["cell_index"].to_numpy())
    num_pre = len(pre_cell_inds)
    num_post = len(post_cell_inds)

    num_connections = np.random.binomial(num_pre * num_post, p)
    pre_syn_neurons = np.random.choice(pre_cell_inds, size=num_connections)
    post_syn_neurons = np.random.choice(post_cell_inds, size=num_connections)

    # Sort the synapses only for convenience of inspecting `.edges`.
    sorting = np.argsort(pre_syn_neurons)
    pre_syn_neurons = pre_syn_neurons[sorting]
    post_syn_neurons = pre_syn_neurons[post_syn_neurons]

    # Post-synapse is a randomly chosen branch and compartment.
    nbranches_post = np.asarray(pre_cell_view.pointer.nbranches_per_cell)[
        post_syn_neurons
    ]
    rand_branch_post = np.floor(np.random.rand(num_connections) * nbranches_post)
    rand_comp_post = np.floor(
        np.random.rand(num_connections) * pre_cell_view.pointer.nseg
    )
    global_post_indices = post_cell_view.pointer._local_inds_to_global(
        post_syn_neurons, rand_branch_post, rand_comp_post
    ).astype(int)
    post_rows = post_cell_view.view.loc[global_post_indices]

    # Pre-synapse is at the zero-eth branch and zero-eth compartment.
    global_pre_indices = post_cell_view.pointer._local_inds_to_global(
        pre_syn_neurons, np.zeros(num_connections), np.zeros(num_connections)
    ).astype(int)
    pre_rows = pre_cell_view.view.loc[global_pre_indices]

    pre_cell_view._append_multiple_synapses(pre_rows, post_rows, synapse_type)
