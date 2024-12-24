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


def get_random_post_comps(post_cell_view: "View", num_post: int) -> "CompartmentView":
    """Sample global compartment indices from all postsynaptic cells."""
    global_post_comp_indices = (
        post_cell_view.nodes.groupby("global_cell_index")
        .sample(num_post, replace=True)
        .index.to_numpy()
    )
    global_post_comp_indices = global_post_comp_indices.reshape(
        (-1, num_post), order="F"
    ).ravel()
    return global_post_comp_indices


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

    # Get a view of the zeroeth compartment of each cell as the pre compartments
    pre_comps = pre_cell_view.scope("local").branch(0).comp(0).nodes.copy()
    # Repeat rows `num_post` times
    pre_rows = pre_comps.loc[pre_comps.index.repeat(num_post)].reset_index(drop=True)

    if random_post_comp:
        global_post_comp_indices = get_random_post_comps(post_cell_view, num_pre)
    else:
        # Post-synapse also at the zero-eth branch and zero-eth compartment
        to_idx = np.tile(range(0, num_post), num_pre)
        global_post_comp_indices = (
            post_cell_view.nodes.groupby("global_cell_index").first()[
                "global_comp_index"
            ]
        ).to_numpy()
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

    Args:
        pre_cell_view: View of the presynaptic cell.
        post_cell_view: View of the postsynaptic cell.
        synapse_type: The synapse to append.
        p: Probability of connection.
        random_post_comp: If True, randomly samples the postsynaptic compartments.
    """
    # Get pre- and postsynaptic cell indices.
    num_pre = len(pre_cell_view._cells_in_view)
    num_post = len(post_cell_view._cells_in_view)

    # Generate random cxns via Bernoulli trials (no duplicates), done in blocks of the
    # connectivity matrix to save memory and time (smaller cut size saves memory,
    # larger saves time)
    cut_size = 100  # --> (100, 100) dim blocks
    pre_inds, post_inds = [], []
    for i in range((num_pre + cut_size - 1) // cut_size):
        for j in range((num_post + cut_size - 1) // cut_size):
            block = np.random.binomial(1, p, size=(cut_size, cut_size))
            block_pre, block_post = np.where(block)
            block_pre += i * cut_size  # block inds --> full adj mat inds
            block_post += j * cut_size  # block inds --> full adj mat inds
            pre_inds.append(block_pre)
            post_inds.append(block_post)
    pre_post_inds = np.stack(
        (np.concatenate(pre_inds), np.concatenate(post_inds)), axis=1
    )
    # Filter out connections where either pre or post index is out of range
    pre_post_inds = pre_post_inds[
        (pre_post_inds[:, 0] < num_pre) & (pre_post_inds[:, 1] < num_post)
    ]
    from_idx, to_idx = pre_post_inds[:, 0], pre_post_inds[:, 1]

    # Pre-synapse at the zero-eth branch and zero-eth compartment
    global_pre_comp_indices = (
        pre_cell_view.nodes.groupby("global_cell_index").first()["global_comp_index"]
    ).to_numpy()
    pre_rows = pre_cell_view.select(nodes=global_pre_comp_indices[from_idx]).nodes

    if random_post_comp:
        global_post_comp_indices = get_random_post_comps(post_cell_view, num_pre)
    else:
        # Post-synapse also at the zero-eth branch and zero-eth compartment
        global_post_comp_indices = (
            post_cell_view.nodes.groupby("global_cell_index").first()[
                "global_comp_index"
            ]
        ).to_numpy()
    post_rows = post_cell_view.select(nodes=global_post_comp_indices[to_idx]).nodes

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
        global_post_comp_indices = get_random_post_comps(post_cell_view, num_pre)
    else:
        # Post-synapse also at the zero-eth branch and zero-eth compartment
        global_post_comp_indices = (
            post_cell_view.nodes.groupby("global_cell_index").first()[
                "global_comp_index"
            ]
        ).to_numpy()
    post_rows = post_cell_view.select(nodes=global_post_comp_indices[to_idx]).nodes

    pre_cell_view.base._append_multiple_synapses(pre_rows, post_rows, synapse_type)
