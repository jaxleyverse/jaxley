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


def get_pre_post_inds(pre_cell_view, post_cell_view):
    pre_cell_inds = np.unique(pre_cell_view.view["cell_index"].to_numpy())
    post_cell_inds = np.unique(post_cell_view.view["cell_index"].to_numpy())
    return pre_cell_inds, post_cell_inds


def fully_connect(
    pre_cell_view: "CellView", post_cell_view: "CellView", synapse_type: "Synapse"
):
    """Appends multiple connections which build a fully connected layer.

    Connections are from branch 0 location 0 to a randomly chosen branch and loc.
    """
    # Get pre- and postsynaptic cell indices.
    pre_cell_inds, post_cell_inds = get_pre_post_inds(pre_cell_view, post_cell_view)
    num_pre, num_post = len(pre_cell_inds), len(post_cell_inds)

    # Infer indices of (random) postsynaptic compartments.
    global_post_indices = (
        post_cell_view.view.groupby("cell_index")
        .sample(num_pre, replace=True)
        .index.to_numpy()
    )
    global_post_indices = global_post_indices.reshape((-1, num_pre), order="F").ravel()

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
    # Get pre- and postsynaptic cell indices.
    pre_cell_inds, post_cell_inds = get_pre_post_inds(pre_cell_view, post_cell_view)
    num_pre, num_post = len(pre_cell_inds), len(post_cell_inds)

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


def custom_connect(
    pre_cell_view: "CellView",
    post_cell_view: "CellView",
    synapse_type: "Synapse",
    connectivity_matrix: np.ndarray,
):
    # Get pre- and postsynaptic cell indices.
    pre_cell_inds, post_cell_inds = get_pre_post_inds(pre_cell_view, post_cell_view)

    # get connection pairs from connectivity matrix
    from_idx, to_idx = np.where(connectivity_matrix)
    pre_cell_inds = pre_cell_inds[from_idx]
    post_cell_inds = post_cell_inds[to_idx]

    # Infer indices of (random) postsynaptic compartments.
    cell_idx_view = lambda view, cell_idx: view[view["cell_index"] == cell_idx]
    sample_comp = lambda view, cell_idx: cell_idx_view(view.view, cell_idx).sample()
    global_post_indices = [
        sample_comp(post_cell_view, cell_idx).index[0] for cell_idx in post_cell_inds
    ]
    post_rows = post_cell_view.view.loc[global_post_indices]

    idcs_to_zero = np.zeros_like(from_idx)
    get_global_idx = post_cell_view.pointer._local_inds_to_global
    global_pre_indices = get_global_idx(pre_cell_inds, idcs_to_zero, idcs_to_zero)
    pre_rows = pre_cell_view.view.loc[global_pre_indices]

    pre_cell_view._append_multiple_synapses(pre_rows, post_rows, synapse_type)
