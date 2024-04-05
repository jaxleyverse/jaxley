from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np


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


def get_pre_post_inds(
    pre_cell_view: "CellView", post_cell_view: "CellView"
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the unique cell indices of the pre- and postsynaptic cells."""
    pre_cell_inds = np.unique(pre_cell_view.view["cell_index"].to_numpy())
    post_cell_inds = np.unique(post_cell_view.view["cell_index"].to_numpy())
    return pre_cell_inds, post_cell_inds


def pre_comp_not_equal_post_comp(
    pre: "CompartmentView", post: "CompartmentView"
) -> bool:
    """Check if pre and post compartments are different."""
    cols = ["cell_index", "branch_index", "comp_index"]
    return np.any(pre.view[cols].values != post.view[cols].values, axis=1)


def is_part_of_network(pre: "View", post: "View") -> bool:
    """Check if views are from the same network."""
    is_in_net = "network" in pre.pointer.__class__.__name__.lower()
    is_in_same_net = pre.pointer is post.pointer
    return is_in_net and is_in_same_net


def sample_comp(
    cell_view: "CellView", cell_idx: int, num: int = 1, replace=True
) -> "CompartmentView":
    cell_idx_view = lambda view, cell_idx: view[view["cell_index"] == cell_idx]
    return cell_idx_view(cell_view.view, cell_idx).sample(num, replace=replace)


def connect(
    pre: "CompartmentView",
    post: "CompartmentView",
    synapse_type: Union["Synapse", List["Synapse"]],
):
    """Connect two compartments with a chemical synapse.

    High-level strategy:

    We need to first check if the network already has a type of this synapse, else
    we need to register it as a new synapse in a bunch of dictionaries which track
    synapse parameters, state and meta information.

    Next, we register the new connection in the synapse dataframe (`.edges`).
    Then, we update synapse parameter and state arrays with the new connection.
    Finally, we update synapse meta information.
    """
    assert is_part_of_network(
        pre, post
    ), "Pre and post compartments must be part of the same network."
    assert np.all(
        pre_comp_not_equal_post_comp(pre, post)
    ), "Pre and post compartments must be different."

    pre._append_multiple_synapses(pre.view, post.view, synapse_type)


def fully_connect(
    pre_cell_view: "CellView",
    post_cell_view: "CellView",
    synapse_type: Union["Synapse", List["Synapse"]],
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
    synapse_type: Union["Synapse", List["Synapse"]],
    p,
):
    """Appends multiple connections which build a sparse, randomly connected layer.

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
    post_syn_neurons = post_syn_neurons[sorting]

    # Post-synapse is a randomly chosen branch and compartment.
    global_post_indices = [
        sample_comp(post_cell_view, cell_idx).index[0] for cell_idx in post_syn_neurons
    ]
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
    synapse_type: Union["Synapse", List["Synapse"]],
    connectivity_matrix: np.ndarray,
):
    """Appends multiple connections which build a custom connected network.

    Connects pre- and postsynaptic cells according to a custom connectivity matrix.
    Entries > 0 in the matrix indicate a connection between the corresponding cells.
    Connections are from branch 0 location 0 to a randomly chosen branch and loc.
    """
    # Get pre- and postsynaptic cell indices.
    pre_cell_inds, post_cell_inds = get_pre_post_inds(pre_cell_view, post_cell_view)

    # get connection pairs from connectivity matrix
    from_idx, to_idx = np.where(connectivity_matrix)
    pre_cell_inds = pre_cell_inds[from_idx]
    post_cell_inds = post_cell_inds[to_idx]

    # Sample random postsynaptic compartments (global comp indices).
    global_post_indices = [
        sample_comp(post_cell_view, cell_idx).index[0] for cell_idx in post_cell_inds
    ]
    post_rows = post_cell_view.view.loc[global_post_indices]

    idcs_to_zero = np.zeros_like(from_idx)
    get_global_idx = post_cell_view.pointer._local_inds_to_global
    global_pre_indices = get_global_idx(pre_cell_inds, idcs_to_zero, idcs_to_zero)
    pre_rows = pre_cell_view.view.loc[global_pre_indices]

    pre_cell_view._append_multiple_synapses(pre_rows, post_rows, synapse_type)
