# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
from typing import Any, Dict, List, Optional, Tuple

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates sparse matrix indices from the table of node edges.

    This is only used for the `jax.sparse` voltage solver.

    Args:
        comp_edges: Dataframe with three columns (sink, source, type).

    Returns:
        data_inds: The indices to reorder the data.
        indices and indptr: Indices passed to the sparse matrix solver.
    """
    sources = comp_edges["source"].to_numpy()
    sinks = comp_edges["sink"].to_numpy()
    n_nodes = np.max(sinks) + 1 if len(sinks) > 0 else 1

    # (row, col) indices of the diagonal, followed by the off-diagonals.
    rows = np.concatenate([np.arange(n_nodes), sources]).astype(int)
    cols = np.concatenate([np.arange(n_nodes), sinks]).astype(int)

    # Cast (row, col) indices to the format required for the `jax` sparse solver.
    return convert_to_csc(num_elements=len(rows), row_ind=rows, col_ind=cols)


def dhs_permutation_indices(
    offdiag_inds: np.ndarray, node_order: np.ndarray, solve_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return mapping to the DHS solve order, also for lower and upper diagonal vals.

    Args:
        offdiag_inds: Array of shape `(2, num_edges)`, the (source, sink) node of every
            compartment edge.
        node_order: Array of shape `(N, 3)`, (node, parent, level) of every edge
            traversed by the solve.
        solve_index: For every node, its index in the solve order.

    Returns:
        - The positions of the off-diagonal entries (edges) in the solve order: first
          the lowers (parent -> child), then the uppers (child -> parent), each sorted
          by the child (which has exactly one parent).
        - `node_order` with nodes and parents replaced by their solve index.
    """
    # `reshape` because a single compartment has no edges.
    source, sink = solve_index[np.asarray(offdiag_inds, dtype=int).reshape(2, -1)]
    is_lower = source < sink
    lowers = np.flatnonzero(is_lower)[np.argsort(sink[is_lower])]
    uppers = np.flatnonzero(~is_lower)[np.argsort(source[~is_lower])]

    # `reshape` because `node_order` is empty for point neurons. For networks, it must
    # still have shape (0, 3) to be concatenated with that of other cells.
    node_order = np.asarray(node_order, dtype=int).reshape(-1, 3)
    new_node_order = np.stack(
        [
            solve_index[node_order[:, 0]],
            solve_index[node_order[:, 1]],
            node_order[:, 2],
        ],
        axis=1,
    )
    return np.concatenate([lowers, uppers]), new_node_order


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
        List of length `num_levels`. Each element has shape `(num_comps_per_level, 2)`,
        where 2 indicates (node, parent).
    """
    new_node_order = np.asarray(new_node_order, dtype=int).reshape(-1, 3)
    if len(new_node_order) == 0:
        return []

    # Sort by level. The sort is stable, so nodes keep their order within a level.
    by_level = new_node_order[np.argsort(new_node_order[:, 2], kind="stable")]
    _, num_per_level = np.unique(by_level[:, 2], return_counts=True)
    return np.split(by_level[:, :2], np.cumsum(num_per_level)[:-1])


def dhs_level_indexer(node_order: np.ndarray, num_nodes: int) -> Dict[str, Any]:
    """Return the parts of the DHS solve indexer that follow from the solve order.

    Args:
        node_order: Array of shape (N, 3), (node, parent, level) in the solve order.
        num_nodes: Number of nodes (compartments and branchpoints).

    Returns:
        - `node_order_grouped`: (node, parent) of every level, see
          `dhs_group_comps_into_levels()`.
        - `parent_lookup`: the parent of every node, `-1` for roots and for one extra
          entry at the end. E.g.: `parent = parent_lookup[node]`.
        - `all_children`, `all_parents`: `node_order_grouped`, flattened. Used for the
          custom JVP of the solve.
    """
    node_order = np.asarray(node_order, dtype=int).reshape(-1, 3)
    node_order_grouped = dhs_group_comps_into_levels(node_order)
    parent_lookup = np.full(num_nodes + 1, -1)
    parent_lookup[node_order[:, 0]] = node_order[:, 1]
    edges = np.concatenate([np.zeros((0, 2), dtype=int)] + node_order_grouped)
    return {
        "node_order_grouped": node_order_grouped,
        "parent_lookup": parent_lookup,
        "all_children": edges[:, 0],
        "all_parents": edges[:, 1],
    }
