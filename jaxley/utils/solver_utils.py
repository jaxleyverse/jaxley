# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd


def convert_to_csc(
    num_elements: int, row_ind: np.ndarray, col_ind: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert between two representations for sparse systems.

    This is needed because `jax.scipy.linalg.spsolve` requires the `(ind, indptr)`
    representation, but the `(row, col)` is more intuitive and easier to build.

    This function uses `np` instead of `jnp` because it only deals with indexing which
    can be dealt with only based on the branch structure (i.e. independent of any
    parameter values).

    Written by ChatGPT.
    """
    data_inds = np.arange(num_elements)
    # Step 1: Sort by (col_ind, row_ind)
    sorted_indices = np.lexsort((row_ind, col_ind))
    data_inds = data_inds[sorted_indices]
    row_ind = row_ind[sorted_indices]
    col_ind = col_ind[sorted_indices]

    # Step 2: Create indptr array
    n_cols = col_ind.max() + 1
    indptr = np.zeros(n_cols + 1, dtype=int)
    np.add.at(indptr, col_ind + 1, 1)
    np.cumsum(indptr, out=indptr)

    # Step 3: The row indices are already sorted
    indices = row_ind

    return data_inds, indices, indptr


def comp_edges_to_indices(
    comp_edges: pd.DataFrame,
) -> Tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generates sparse matrix indices from the table of node edges.

    This is only used for the `jax.sparse` voltage solver.

    Args:
        comp_edges: Dataframe with three columns (sink, source, type).

    Returns:
        n_nodes: The number of total nodes (including branchpoints).
        data_inds: The indices to reorder the data.
        indices and indptr: Indices passed to the sparse matrix solver.
    """
    # Build indices for diagonals.
    sources = np.asarray(comp_edges["source"].to_list())
    sinks = np.asarray(comp_edges["sink"].to_list())
    n_nodes = np.max(sinks) + 1 if len(sinks) > 0 else 1
    diagonal_inds = jnp.stack([jnp.arange(n_nodes), jnp.arange(n_nodes)])

    # Build indices for off-diagonals.
    off_diagonal_inds = jnp.stack([sources, sinks]).astype(int)

    # Concatenate indices of diagonals and off-diagonals.
    all_inds = jnp.concatenate([diagonal_inds, off_diagonal_inds], axis=1)

    # Cast (row, col) indices to the format required for the `jax` sparse solver.
    data_inds, indices, indptr = convert_to_csc(
        num_elements=all_inds.shape[1],
        row_ind=all_inds[0],
        col_ind=all_inds[1],
    )
    return data_inds, indices, indptr


def dhs_permutation_indices(
    lowers_and_uppers,
    offdiag_inds,
    node_order,
    mapping_dict,
):
    """Return mapping to the DHS solve order, also for lower and upper diagonal vals."""
    edge_data = {}
    for edge, v in zip(offdiag_inds.T, lowers_and_uppers):
        edge_data[tuple(edge.tolist())] = v

    ordered_comp_edges = {}
    for i, j in edge_data.keys():
        new_i = mapping_dict[i]
        new_j = mapping_dict[j]
        ordered_comp_edges[(new_i, new_j)] = edge_data[(i, j)]

    lowers = {}
    uppers = {}
    for ij in ordered_comp_edges.keys():
        if ij[0] < ij[1]:
            lowers[ij] = ordered_comp_edges[ij]
        else:
            uppers[ij] = ordered_comp_edges[ij]

    # We have to sort lowers and uppers to be in the solve order.
    children = []
    for key in lowers.keys():
        children.append(key[1])
    lower_sorting = np.argsort(children)

    parents = []
    for key in uppers.keys():
        parents.append(key[0])
    upper_sorting = np.argsort(parents)

    lowers = jnp.asarray(list(lowers.values()))[lower_sorting]
    uppers = jnp.asarray(list(uppers.values()))[upper_sorting]

    # Adapt node order.
    new_node_order = []
    for n in node_order:
        new_node_order.append(
            [mapping_dict[int(n[0])], mapping_dict[int(n[1])], int(n[2])]
        )
    new_node_order = jnp.asarray(new_node_order)

    lowers_and_uppers = jnp.concatenate([lowers, uppers])
    return lowers_and_uppers, new_node_order


def dhs_solve_index(
    solve_graph: nx.DiGraph,
    allowed_nodes_per_level: int = 1,
    root: Optional[int] = 0,
) -> nx.DiGraph:
    """Given a compartment graph, return a directed graph indicating the solve order.

    Args:
        comp_graph: Compartment graph. Must have the compartment indices such that
            they match the ones in the `jx.Module` (i.e., the compartment graph must
            have been generated with `to_graph()` or it must have run
            `_set_comp_and_branch_index()` after having run
            `build_compartment_graph()`).
        root: The root node to traverse the graph for the DHS solve order.
        allowed_nodes_per_level: How many nodes are visited before the level is
            increased, even if the number of hops did not change.

    Returns:
        - node_and_parent: A list of tuples (node, parent, level), where the node and
        parent indicate the (child and parent) node index, and the level indicates the
        number of hops it took from the root to get there. The number of hops is
        currently unused, but it will be important in the future to parallelization.
        - node_to_solve_index_mapping: An dictionary mapping from the compartment
        indices to the solve indices.
    """
    undirected_solve_graph = solve_graph.to_undirected()

    # Traverse the graph for the solve order.
    # `sort_neighbors=lambda x: sorted(x)` to first handle nodes with lower node index.
    node_and_parent = [(root, -1, 0)]
    solve_graph.nodes[root]["solve_index"] = 0
    solve_index = 1
    for i, j, level in bfs_edge_hops(
        undirected_solve_graph, root, allowed_nodes_per_level
    ):
        solve_graph.add_edge(i, j)
        # Copy comp_index and branch_index from compartment graph. We only update the
        # solve_index.
        solve_graph.nodes[j]["solve_index"] = solve_index
        node_and_parent.append((j, i, level))
        solve_index += 1

    # Create a dictionary which maps every node to its solve index. Two notes:
    # - The node name corresponds to the compartment index (and branchpoints continue
    # numerically from where the compartments have ended)
    # - The solve index specifies the order in which the node is processed during the
    # DHS solve.
    inds = {node: solve_graph.nodes[node]["solve_index"] for node in solve_graph.nodes}
    node_to_solve_index_mapping = dict(sorted(inds.items()))
    return node_and_parent, node_to_solve_index_mapping


def bfs_edge_hops(graph: nx.DiGraph, root: Any, allowed_nodes_per_level: int):
    """Yields BFS tree edges along with hop count from root.

    Hop count increases when:
    - A real hop (child is at parent + 1 distance).
    - Or after `allowed_nodes_per_level` nodes have been discovered without hop
      increase.

    Function written by ChatGPT.

    Args:
        graph: The graph to traverse.
        root: The starting node for BFS.
        allowed_nodes_per_level: How many nodes are visited before the level is
            increased, even if the number of hops did not change.

    Yields:
        Tuple[Tuple[Any, Any], int]: Edge u, v and current hop count.
    """
    true_distance = nx.single_source_shortest_path_length(graph, root)

    current_depth = 0
    nodes_since_last_depth_increase = 0
    last_true_depth = true_distance[root]

    for u, v in nx.bfs_edges(graph, root):
        # Detect real hop
        if true_distance[v] > last_true_depth:
            current_depth += 1
            nodes_since_last_depth_increase = 0
            last_true_depth = true_distance[v]
        elif nodes_since_last_depth_increase >= allowed_nodes_per_level:
            current_depth += 1
            nodes_since_last_depth_increase = 0

        nodes_since_last_depth_increase += 1
        yield u, v, current_depth


def dhs_group_comps_into_levels(new_node_order: np.ndarray) -> List[np.ndarray]:
    """Group nodes into levels, such that nodes get processed in parallel when possible.

    Args:
        new_node_order: Array of shape (N, 3). The `3` are (node, parent, level).
            `N` is the number of compartment edges to be processed.

    Returns:
        Array of shape (num_levels, allowed_nodes_per_level, 2), where 2 indicates
        (node, parent).
    """
    if len(new_node_order) == 0:
        # If len == 0, we need to ensure that there are 3 columns, otherwise
        # pd.DataFrame() will throw an error.
        new_node_order = np.zeros((0, 3)).astype(int)

    # Group the edges by their level. Each level is processed in parallel.
    nodes = pd.DataFrame(new_node_order, columns=["node", "parent", "level"])
    grouping = nodes.groupby("level")
    nodes = grouping["node"].apply(list).to_numpy()
    parents = grouping["parent"].apply(list).to_numpy()
    nodes_and_parents = [np.stack([n, p]).T for n, p in zip(nodes, parents)]

    # `nodes_and_parents` is a List of length `num_levels`. Each element has shape
    # `(num_comps_per_level, 2)`.
    return nodes_and_parents
