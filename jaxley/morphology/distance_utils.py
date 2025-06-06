import jax.numpy as jnp

from typing import List
from jaxley.modules.base import to_graph
import networkx as nx


def distance(
    startpoint: "View", endpoints: Union["Module", "View"], kind: str = "pathwise"
) -> List[float]:
    """Return the distance (pathwise or direct) between a root and other compartments.

    Args:
        startpoint: A single compartment from which to compute the distance.
        endpoints: One or multiple compartments to which to compute the distance to.
        kind: Either of 'pathwise' or 'direct'. If 'pathwise', it computes the
            shortest path distance between two compartments. If 'direct', it computes
            the direct (line of sight) distance between two compartments. For
            'pathwise', we use Dijkstra's algorithm to get the path with the lowest
            number of compartments between start and endpoint. It then computes the
            length of that path in micrometers. Note that, for an uncyclic graph, the
            path with the lowest number of compartments between start and endpoint is
            also the path with the lowest length. For 'direct', we use
            `cell.nodes[['x', 'y', 'z']]` and compute the euclidean distance.

    Returns:
        A list of distances.

    Example usage
    ^^^^^^^^^^^^^

    Example 1: The following computes the pathwise distance between the zero-eth soma
    compartment and all other compartments. It then saves this distance in
    `cell.nodes["path_dist_from_soma"]`.

    ::

        path_dists = distance(cell.soma.branch(0).comp(0), cell)
        cell.nodes["path_dist_from_soma"] = path_dists

    Example 2: The following computes the pathwise distance between two compartments.

    ::

        dist = distance(cell.branch(8).comp(2), cell.branch(2).comp(0))

    Example 3: The following computes the direct (line of sight) distance between the
    zero-eth soma compartment and all other compartments. It then saves this distance
    in `cell.nodes["direct_dist_from_soma"]`.

    ::

        direct_dists = distance(cell.soma.branch(0).comp(0), cell, kind='direct')
        cell.nodes["direct_dist_from_soma"] = direct_dists
    """
    assert len(startpoint.nodes.index) == 1, "Cannot use multiple root nodes."

    if kind == "direct":
        start_xyz = startpoint.nodes[["x", "y", "z"]].to_numpy()[0]
        end_xyz = endpoints.nodes[["x", "y", "z"]].to_numpy()
        return jnp.sqrt(jnp.sum((start_xyz - end_xyz) ** 2, axis=1))

    elif kind == "pathwise":
        root = startpoint.nodes.index[0]
        endpoint_inds = endpoints.nodes.index
        graph = to_graph(startpoint.base)
        graph = nx.to_undirected(graph)

        # Set default for branchpoints.
        for _, data in graph.nodes(data=True):
            data.setdefault("length", 0.0)

        def custom_path_length(graph: nx.Graph, path: List):
            """Compute total path length, counting root and target node lengths as half.

            Args:
                G (nx.Graph): Graph with node attribute "length".
                path (list): List of node IDs representing the path.

            Returns:
                float: Adjusted total path length.
            """
            if not path:
                return 0
            length = sum(graph.nodes[n]["length"] for n in path)
            length -= 0.5 * graph.nodes[path[0]]["length"]  # subtract half root
            length -= 0.5 * graph.nodes[path[-1]]["length"]  # subtract half target
            return length

        # Use Dijkstra to minimum adjusted length from root to all endpoint nodes.
        lengths = []
        for endpoint in endpoint_inds:
            path = nx.shortest_path(graph, root, endpoint)
            lengths.append(custom_path_length(graph, path))
        return lengths

    else:
        raise ValueError(
            f"`kind` must be either of `pathwise` or `direct`, but you passed `{kind}`"
        )
