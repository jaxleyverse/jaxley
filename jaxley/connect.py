# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import numpy as np


def is_same_network(pre: "View", post: "View") -> bool:
    """Check if views are from the same network."""
    is_in_net = "network" in pre.base.__class__.__name__.lower()
    is_in_same_net = pre.base is post.base
    return is_in_net and is_in_same_net


def sample_comp(cell_view: "View", num: int = 1, replace=True) -> "CompartmentView":
    """Sample a compartment from a cell.

    Returns View with shape (num, num_cols)."""
    return np.random.choice(cell_view._comps_in_view, num, replace=replace)


def connect(
    pre: "View",
    post: "View",
    synapse_type: "Synapse",
):
    """Connect two compartments with a chemical synapse.

    The pre- and postsynaptic compartments must be different compartments of the
    same network.

    Args:
        pre: View of the presynaptic compartment.
        post: View of the postsynaptic compartment.
        synapse_type: The synapse to append
    """
    assert is_same_network(
        pre, post
    ), "Pre and post compartments must be part of the same network."

    pre.base._append_multiple_synapses(pre.nodes, post.nodes, synapse_type)


def fully_connect(
    pre_cell_view: "View",
    post_cell_view: "View",
    synapse_type: "Synapse",
):
    """Appends multiple connections which build a fully connected layer.

    Connections are from branch 0 location 0 to a randomly chosen branch and loc.

    Args:
        pre_cell_view: View of the presynaptic cell.
        post_cell_view: View of the postsynaptic cell.
        synapse_type: The synapse to append.
    """
    # Get pre- and postsynaptic cell indices.
    num_pre = len(pre_cell_view._cells_in_view)
    num_post = len(post_cell_view._cells_in_view)

    # Pre-synapse at the zero-eth branch and zero-eth compartment
    global_pre_comp_indices = (
        pre_cell_view.scope("local").branch(0).comp(0).nodes.index.to_numpy()
    )  # setting scope ensure that this works indep of current scope
    # Repeat comp indices `num_post` times. See SO 50788508 as before
    global_pre_comp_indices = np.repeat(global_pre_comp_indices, num_post)
    pre_rows = pre_cell_view.select(nodes=global_pre_comp_indices).nodes

    # Post-synapse also at the zero-eth branch and zero-eth compartment
    global_post_comp_indices = (
        post_cell_view.scope("local").branch(0).comp(0).nodes.index.to_numpy()
    )
    # Tile comp indices `num_pre` times
    global_post_comp_indices = np.tile(global_post_comp_indices, num_pre)
    post_rows = post_cell_view.select(nodes=global_post_comp_indices).nodes

    pre_cell_view.base._append_multiple_synapses(pre_rows, post_rows, synapse_type)


def sparse_connect(
    pre_cell_view: "View",
    post_cell_view: "View",
    synapse_type: "Synapse",
    p: float,
):
    """Appends multiple connections which build a sparse, randomly connected layer.

    Connections are from branch 0 location 0 to branch 0 location 0.

    Args:
        pre_cell_view: View of the presynaptic cell.
        post_cell_view: View of the postsynaptic cell.
        synapse_type: The synapse to append.
        p: Probability of connection.
    """
    # Get pre- and postsynaptic cell indices.
    pre_cell_inds = pre_cell_view._cells_in_view
    post_cell_inds = post_cell_view._cells_in_view
    num_pre = len(pre_cell_inds)
    num_post = len(post_cell_inds)

    # Get the indices of connections, like it's from a random connectivity matrix
    num_connections = np.random.binomial(num_pre * num_post, p)
    from_idx = np.random.choice(range(0, num_pre), size=num_connections)
    to_idx = np.random.choice(range(0, num_post), size=num_connections)

    # Remove duplicate connections
    row_inds = np.stack((from_idx, to_idx), axis=1)
    row_inds = np.unique(row_inds, axis=0)
    from_idx = row_inds[:, 0]
    to_idx = row_inds[:, 1]
    
    # Pre-synapse at the zero-eth branch and zero-eth compartment
    global_pre_comp_indices = (
        pre_cell_view.scope("local").branch(0).comp(0).nodes.index.to_numpy()
    )  # setting scope ensure that this works indep of current scope
    pre_rows = pre_cell_view.select(nodes=global_pre_comp_indices[from_idx]).nodes

    # Post-synapse also at the zero-eth branch and zero-eth compartment
    global_post_comp_indices = (
        post_cell_view.scope("local").branch(0).comp(0).nodes.index.to_numpy()
    )
    post_rows = post_cell_view.select(nodes=global_post_comp_indices[to_idx]).nodes

    if len(pre_rows) > 0:
        pre_cell_view.base._append_multiple_synapses(pre_rows, post_rows, synapse_type)


def connectivity_matrix_connect(
    pre_cell_view: "View",
    post_cell_view: "View",
    synapse_type: "Synapse",
    connectivity_matrix: np.ndarray[bool],
):
    """Appends multiple connections which build a custom connected network.

    Connects pre- and postsynaptic cells according to a custom connectivity matrix.
    Entries > 0 in the matrix indicate a connection between the corresponding cells.
    Connections are from branch 0 location 0 on the presynaptic cell to branch 0
    location 0 on the postsynaptic cell.

    Args:
        pre_cell_view: View of the presynaptic cell.
        post_cell_view: View of the postsynaptic cell.
        synapse_type: The synapse to append.
        connectivity_matrix: A boolean matrix indicating the connections between cells.
    """
    # Get pre- and postsynaptic cell indices
    global_pre_cell_inds = pre_cell_view._cells_in_view
    global_post_cell_inds = post_cell_view._cells_in_view

    assert connectivity_matrix.shape == (
        len(global_pre_cell_inds),
        len(global_post_cell_inds),
    ), "Connectivity matrix must have shape (num_pre, num_post)."
    assert connectivity_matrix.dtype == bool, "Connectivity matrix must be boolean."

    # Get pre to post connection pairs from connectivity matrix
    from_idx, to_idx = np.where(connectivity_matrix)

    # Pre-synapse at the zero-eth branch and zero-eth compartment
    global_pre_comp_indices = (
        pre_cell_view.scope("local").branch(0).comp(0).nodes.index.to_numpy()
    )  # setting scope ensure that this works indep of current scope
    pre_rows = pre_cell_view.select(nodes=global_pre_comp_indices[from_idx]).nodes

    # Post-synapse also at the zero-eth branch and zero-eth compartment
    global_post_comp_indices = (
        post_cell_view.scope("local").branch(0).comp(0).nodes.index.to_numpy()
    )
    post_rows = post_cell_view.select(nodes=global_post_comp_indices[to_idx]).nodes

    pre_cell_view.base._append_multiple_synapses(pre_rows, post_rows, synapse_type)
