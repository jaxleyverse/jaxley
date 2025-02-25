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
    random_post_comp: bool = False,
):
    """Appends multiple connections which build a fully connected layer.

    Connections are from branch 0 location 0 of the pre-synaptic cell to branch 0
    location 0 of the post-synaptic cell unless random_post_comp=True.

    Args:
        pre_cell_view: View of the presynaptic cell.
        post_cell_view: View of the postsynaptic cell.
        synapse_type: The synapse to append.
        random_post_comp: If True, randomly samples the postsynaptic compartments.
    """
    # Get pre- and postsynaptic cell indices.
    num_pre = len(pre_cell_view._cells_in_view)
    num_post = len(post_cell_view._cells_in_view)

    # Pre-synapse is at the zero-eth branch and zero-eth compartment.
    pre_rows = pre_cell_view.scope("local").branch(0).comp(0).nodes.copy()
    # Repeat rows `num_post` times. See SO 50788508.
    pre_rows = pre_rows.loc[pre_rows.index.repeat(num_post)].reset_index(drop=True)

    if random_post_comp:
        global_post_comp_indices = (
            post_cell_view.nodes.groupby("global_cell_index")
            .sample(num_pre, replace=True)
            .index.to_numpy()
        )
        # Reorder the post comp inds to tile order (pre indices are repeated so here tile needed)
        global_post_comp_indices = np.reshape(
            global_post_comp_indices, (num_pre, num_post)
        ).T.flatten()
    else:
        # Post-synapse also at the zero-eth branch and zero-eth compartment
        global_post_comp_indices = (
            post_cell_view.nodes.groupby("global_cell_index").first()[
                "global_comp_index"
            ]
        ).to_numpy()
        to_idx = np.tile(range(0, num_post), num_pre)
        global_post_comp_indices = global_post_comp_indices[to_idx]

    post_rows = post_cell_view.nodes.loc[global_post_comp_indices]

    pre_cell_view.base._append_multiple_synapses(pre_rows, post_rows, synapse_type)


def sparse_connect(
    pre_cell_view: "View",
    post_cell_view: "View",
    synapse_type: "Synapse",
    p: float,
    random_post_comp: bool = False,
):
    """Appends multiple connections which build a sparse, randomly connected layer.

    Connections are from branch 0 location 0 of the pre-synaptic cell to branch 0
    location 0 of the post-synaptic cell unless random_post_comp=True.

    NOTE: This function does not generate sparse random connectivity with random graph
    generation methodology, cells may be connected multiple times and p=1.0 does
    not fully connect.

    Args:
        pre_cell_view: View of the presynaptic cell.
        post_cell_view: View of the postsynaptic cell.
        synapse_type: The synapse to append.
        p: Probability of connection.
        random_post_comp: If True, randomly samples the postsynaptic compartments.
    """
    # Get pre- and postsynaptic cell indices.
    pre_cell_inds = pre_cell_view._cells_in_view
    post_cell_inds = post_cell_view._cells_in_view
    num_pre = len(pre_cell_view._cells_in_view)
    num_post = len(post_cell_view._cells_in_view)

    num_connections = np.random.binomial(num_pre * num_post, p)
    pre_syn_neurons = np.random.choice(pre_cell_inds, size=num_connections)
    post_syn_neurons = np.random.choice(post_cell_inds, size=num_connections)

    # Sort the synapses only for convenience of inspecting `.edges`.
    sorting = np.argsort(pre_syn_neurons)
    pre_syn_neurons = pre_syn_neurons[sorting]
    post_syn_neurons = post_syn_neurons[sorting]

    # Pre-synapse is at the zero-eth branch and zero-eth compartment.
    global_pre_indices = pre_cell_view.base._cumsum_ncomp_per_cell[pre_syn_neurons]
    pre_rows = pre_cell_view.base.nodes.loc[global_pre_indices]

    # Sample the post-synaptic compartments
    if random_post_comp:
        # Filter the post cell view to include post-synaptic neurons
        post_syn_view = post_cell_view.nodes[
            post_cell_view.nodes["global_cell_index"].isin(post_syn_neurons)
        ]
        # Determine how many comps to sample for each post-synaptic neuron
        unique_cells, counts = np.unique(post_syn_neurons, return_counts=True)
        n_samples_dict = dict(zip(unique_cells, counts))
        sampled_inds = post_syn_view.groupby("global_cell_index").apply(
            lambda x: x.sample(n=n_samples_dict[x.name], replace=True)
        )
        global_post_comp_indices = sampled_inds.global_comp_index.to_numpy()
        post_rows = post_cell_view.nodes.loc[global_post_comp_indices]
    else:
        # Post-synapse also at the zero-eth branch and zero-eth compartment
        global_post_indices = post_cell_view.base._cumsum_ncomp_per_cell[
            post_syn_neurons
        ]
        post_rows = post_cell_view.base.nodes.loc[global_post_indices]

    if len(pre_rows) > 0:
        pre_cell_view.base._append_multiple_synapses(pre_rows, post_rows, synapse_type)


def connectivity_matrix_connect(
    pre_cell_view: "View",
    post_cell_view: "View",
    synapse_type: "Synapse",
    connectivity_matrix: np.ndarray[bool],
    random_post_comp: bool = False,
):
    """Appends multiple connections according to a custom connectivity matrix.

    Entries > 0 in the matrix indicate a connection between the corresponding cells.
    Connections are from branch 0 location 0 of the pre-synaptic cell to branch 0
    location 0 of the post-synaptic cell unless random_post_comp=True.

    Args:
        pre_cell_view: View of the presynaptic cell.
        post_cell_view: View of the postsynaptic cell.
        synapse_type: The synapse to append.
        connectivity_matrix: A boolean matrix indicating the connections between cells.
        random_post_comp: If True, randomly samples the postsynaptic compartments.
    """
    # Get pre- and postsynaptic cell indices
    num_pre = len(pre_cell_view._cells_in_view)
    num_post = len(post_cell_view._cells_in_view)

    assert connectivity_matrix.shape == (
        num_pre,
        num_post,
    ), "Connectivity matrix must have shape (num_pre, num_post)."
    assert connectivity_matrix.dtype == bool, "Connectivity matrix must be boolean."

    # Get pre to post connection pairs from connectivity matrix
    from_idx, to_idx = np.where(connectivity_matrix)

    # Pre-synapse at the zero-eth branch and zero-eth compartment
    global_pre_comp_indices = (
        pre_cell_view.nodes.groupby("global_cell_index").first()["global_comp_index"]
    ).to_numpy()
    pre_rows = pre_cell_view.select(nodes=global_pre_comp_indices[from_idx]).nodes

    if random_post_comp:
        global_to_idx = post_cell_view.nodes.global_cell_index.unique()[to_idx]
        # Filter the post cell view to include post-synaptic neurons selected
        post_syn_view = post_cell_view.nodes[
            post_cell_view.nodes["global_cell_index"].isin(global_to_idx)
        ]
        # Determine how many comps to sample for each post-synaptic neuron
        unique_cells, counts = np.unique(global_to_idx, return_counts=True)
        # Sample the post-synaptic compartments
        n_samples_dict = dict(zip(unique_cells, counts))
        sampled_inds = post_syn_view.groupby("global_cell_index").apply(
            lambda x: x.sample(n=n_samples_dict[x.name], replace=True)
        )
        global_post_comp_indices = sampled_inds.global_comp_index.to_numpy()
    else:
        # Post-synapse also at the zero-eth branch and zero-eth compartment
        global_post_comp_indices = (
            post_cell_view.nodes.groupby("global_cell_index").first()[
                "global_comp_index"
            ]
        ).to_numpy()
        global_post_comp_indices = global_post_comp_indices[to_idx]
    post_rows = post_cell_view.select(nodes=global_post_comp_indices).nodes

    pre_cell_view.base._append_multiple_synapses(pre_rows, post_rows, synapse_type)
