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
    """Connect specific compartments of a network with a synapse.

    The pre- and postsynaptic compartments must be compartments of the
    same network. If `pre` and `post` are both just a single compartment, then this
    function instantiates a single synapse. If `pre` and `post` are both `N`
    compartments, then this function instatiates `N` synapses (first-to-first,
    second-to-second,...).

    Args:
        pre: View of the presynaptic compartment.
        post: View of the postsynaptic compartment.
        synapse_type: The type of synapse to use.

    Example usage
    ^^^^^^^^^^^^^

    Example 1: Connect one compartment to another compartment with a single synapse:

    ::

        from jaxley.connect import connect
        from jaxley.synapses import IonotropicSynapse

        net = jx.Network([cell for _ in range(10)])
        connect(
            net.cell(0).branch(0).comp(0),
            net.cell(1).branch(0).comp(0),
            IonotropicSynapse(),
        )
        print(net.edges)

    Example 2: Connect `N` compartments to `N` other compartments with `N` synapses:

    ::

        from jaxley.connect import connect
        from jaxley.synapses import IonotropicSynapse

        net = jx.Network([cell for _ in range(10)])
        connect(
            net.cell(0).branch([0, 1]).comp(0),
            net.cell(1).branch([2, 3]).comp(0),
            IonotropicSynapse(),
        )
        print(net.edges)
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
    """Fully (densely) connect cells of a network with synapses.

    Connections are from branch 0 location 0 of the pre-synaptic cell to branch 0
    location 0 of the post-synaptic cell unless random_post_comp=True.

    Args:
        pre_cell_view: View of the presynaptic cell.
        post_cell_view: View of the postsynaptic cell.
        synapse_type: The synapse to append.
        random_post_comp: If True, randomly samples the postsynaptic compartments.

    Example usage
    ^^^^^^^^^^^^^

    The following example insert 12 synapses (3 x 4).

    ::

        from jaxley.connect import fully_connect

        net = jx.Network([cell for _ in range(10)])
        fully_connect(
            net.cell([0, 1, 2]),
            net.cell([3, 4, 5, 6]),
            IonotropicSynapse(),
        )
        print(net.edges)
    """
    # Get pre- and postsynaptic cell indices.
    num_pre = len(pre_cell_view._cells_in_view)
    num_post = len(post_cell_view._cells_in_view)

    # Pre-synapse is at the zero-eth branch and zero-eth compartment.
    pre_rows = pre_cell_view.scope("local").branch(0).comp(0).nodes.copy()
    # Repeat rows `num_post` times. See SO 50788508.
    pre_rows = pre_rows.loc[pre_rows.index.repeat(num_post)]

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
        post_cell_view.nodes["orig_index"] = post_cell_view.nodes.index
        global_post_comp_indices = (
            post_cell_view.nodes.groupby("global_cell_index").first()["orig_index"]
        ).to_numpy()
        to_idx = np.tile(range(0, num_post), num_pre)
        global_post_comp_indices = global_post_comp_indices[to_idx]
        post_cell_view.nodes.drop(columns="orig_index", inplace=True)

    post_rows = post_cell_view.nodes.loc[global_post_comp_indices]

    pre_cell_view.base._append_multiple_synapses(pre_rows, post_rows, synapse_type)


def sparse_connect(
    pre_cell_view: "View",
    post_cell_view: "View",
    synapse_type: "Synapse",
    p: float,
    random_post_comp: bool = False,
):
    """Sparsely (densely) connect cells of a network with synapses.

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

    Example usage
    ^^^^^^^^^^^^^

    The following example insert approximately 6 synapses (3 x 4 = 12 possible
    synapses, with connection probability 0.5).

    ::

        from jaxley.connect import sparse_connect
        from jaxley.synapses import IonotropicSynapse

        net = jx.Network([cell for _ in range(10)])
        sparse_connect(
            net.cell([0, 1, 2]),
            net.cell([3, 4, 5, 6]),
            IonotropicSynapse(),
            p=0.5,
        )
        print(net.edges)
    """
    # Get pre- and postsynaptic cell indices.
    pre_cell_inds = pre_cell_view._cells_in_view
    post_cell_inds = post_cell_view._cells_in_view
    num_pre = len(pre_cell_view._cells_in_view)
    num_post = len(post_cell_view._cells_in_view)

    num_connections = np.random.binomial(num_pre * num_post, p)

    if num_connections == 0:
        # Don't do any of the following connections if no synapse is inserted.
        return
    pre_syn_neurons = np.random.choice(pre_cell_inds, size=num_connections)
    post_syn_neurons = np.random.choice(post_cell_inds, size=num_connections)

    # Sort the synapses only for convenience of inspecting `.edges`.
    sorting = np.argsort(pre_syn_neurons)
    pre_syn_neurons = pre_syn_neurons[sorting]
    post_syn_neurons = post_syn_neurons[sorting]

    pre_syn_neurons, inverse_pre = np.unique(pre_syn_neurons, return_inverse=True)

    # Pre-synapse is at the zero-eth branch and zero-eth compartment.
    global_pre_indices = (
        pre_cell_view.scope("global")
        .cell(pre_syn_neurons)
        .scope("local")
        .branch(0)
        .comp(0)
        .nodes.index
    )
    global_pre_indices = global_pre_indices[inverse_pre]
    pre_rows = pre_cell_view.base.nodes.loc[global_pre_indices]

    # Sample the post-synaptic compartments
    if random_post_comp:
        # Filter the post cell view to include post-synaptic neurons
        post_syn_view = post_cell_view.nodes[
            post_cell_view.nodes["global_cell_index"].isin(post_syn_neurons)
        ].copy()
        # Determine how many comps to sample for each post-synaptic neuron
        unique_cells, counts = np.unique(post_syn_neurons, return_counts=True)
        n_samples_dict = dict(zip(unique_cells, counts))
        post_syn_view["orig_index"] = post_syn_view.index
        sampled_inds = post_syn_view.groupby("global_cell_index").apply(
            lambda x: x.sample(n=n_samples_dict[x.name], replace=True)
        )
        global_post_comp_indices = sampled_inds.orig_index.to_numpy()
        post_rows = post_cell_view.nodes.loc[global_post_comp_indices]
        post_syn_view.drop(columns="orig_index", inplace=True)
    else:
        post_syn_neurons, inverse_post = np.unique(
            post_syn_neurons, return_inverse=True
        )
        # Post-synapse also at the zero-eth branch and zero-eth compartment
        global_post_indices = (
            post_cell_view.scope("global")
            .cell(post_syn_neurons)
            .scope("local")
            .branch(0)
            .comp(0)
            .nodes.index
        )
        global_post_indices = global_post_indices[inverse_post]
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
    """Connect cells of a network with synapses via a boolean connectivity matrix.

    Entries > 0 in the matrix indicate a connection between the corresponding cells.
    Connections are from branch 0 location 0 of the pre-synaptic cell to branch 0
    location 0 of the post-synaptic cell unless random_post_comp=True.

    Args:
        pre_cell_view: View of the presynaptic cell.
        post_cell_view: View of the postsynaptic cell.
        synapse_type: The synapse to append.
        connectivity_matrix: A boolean matrix indicating the connections between cells.
            If floating point values are passed, they are _not_ interpreted as
            synaptic weights, but we only check if they are zero (no connection) or
            not (connection).
        random_post_comp: If True, randomly samples the postsynaptic compartments.

    Example usage
    ^^^^^^^^^^^^^

    The following generates a random 10 x 10 boolean matrix and uses it to connect the
    neurons in a network.

    ::

        from jaxley.connect import connectivity_matrix_connect
        from jaxley.synapses import IonotropicSynapse

        net = jx.Network([cell for _ in range(10)])
        connectivity_matrix = np.random.choice([False, True], size=(10, 10))
        connectivity_matrix_connect(
            net.cell("all"),
            net.cell("all"),
            IonotropicSynapse(),
            connectivity_matrix,
        )
        print(net.edges)
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
    pre_cell_view.nodes["orig_index"] = pre_cell_view.nodes.index
    post_cell_view.nodes["orig_index"] = post_cell_view.nodes.index
    global_pre_comp_indices = (
        pre_cell_view.nodes.groupby("global_cell_index").first()["orig_index"]
    ).to_numpy()
    pre_rows = pre_cell_view.select(nodes=global_pre_comp_indices[from_idx]).nodes
    pre_cell_view.nodes.drop(columns="orig_index", inplace=True)

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
        global_post_comp_indices = sampled_inds.orig_index.to_numpy()
    else:
        # Post-synapse also at the zero-eth branch and zero-eth compartment
        global_post_comp_indices = (
            post_cell_view.nodes.groupby("global_cell_index").first()["orig_index"]
        ).to_numpy()
        global_post_comp_indices = global_post_comp_indices[to_idx]
        post_cell_view.nodes.drop(columns="orig_index", inplace=True)
    post_rows = post_cell_view.select(nodes=global_post_comp_indices).nodes

    pre_cell_view.base._append_multiple_synapses(pre_rows, post_rows, synapse_type)
