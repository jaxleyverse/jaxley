# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, List, Union

import jax.numpy as jnp
import networkx as nx

from jaxley.modules.base import to_graph


def distance_direct(
    startpoint: "View",
    endpoints: Union["Branch", "Cell", "View"],
) -> List[float]:
    """Returns the direct distance between a root and other compartments.

    This function uses ``cell.nodes[['x', 'y', 'z']]`` and computes the euclidean
    distance (i.e., the line of sight distance).

    Args:
        startpoint: A single compartment from which to compute the distance.
        endpoints: One or multiple compartments to which to compute the distance to.

    Returns:
        A list of distances.

    Example usage
    ^^^^^^^^^^^^^

    The following computes the direct (line of sight) distance between the
    zero-eth soma compartment and all other compartments. It then saves this distance
    in `cell.nodes["direct_dist_from_soma"]`.

    ::

        from jaxley.morphology import distance_pathwise

        cell.compute_compartment_centers()  # necessary if you modified branch length.
        direct_dists = distance_direct(cell.soma.branch(0).comp(0), cell)
        cell.nodes["direct_dist_from_soma"] = direct_dists
    """
    assert len(startpoint.nodes.index) == 1, "Cannot use multiple root nodes."

    start_xyz = startpoint.nodes[["x", "y", "z"]].to_numpy()[0]
    end_xyz = endpoints.nodes[["x", "y", "z"]].to_numpy()
    return jnp.sqrt(jnp.sum((start_xyz - end_xyz) ** 2, axis=1))


def distance_pathwise(
    startpoint: "View", endpoints: Union["Branch", "Cell", "View"]
) -> List[float]:
    """Returns the pathwise distance between a root and other compartments.

    We use Dijkstra's algorithm to get the path with the lowest
    number of compartments between start and endpoint. It then computes the
    length of that path in micrometers. Note that, for an uncyclic graph, the
    path with the lowest number of compartments between start and endpoint is
    also the path with the lowest length.

    Args:
        startpoint: A single compartment from which to compute the distance.
        endpoints: One or multiple compartments to which to compute the distance to.

    Returns:
        A list of distances.

    Example usage
    ^^^^^^^^^^^^^

    Example 1: The following computes the pathwise distance between the zero-eth soma
    compartment and all other compartments. It then saves this distance in
    `cell.nodes["path_dist_from_soma"]`.

    ::

        from jaxley.morphology import distance_pathwise

        path_dists = distance_pathwise(cell.soma.branch(0).comp(0), cell)
        cell.nodes["path_dist_from_soma"] = path_dists

    Example 2: The following computes the pathwise distance between two compartments.

    ::

        dist = distance_pathwise(cell.branch(8).comp(2), cell.branch(2).comp(0))
    """
    assert len(startpoint.nodes.index) == 1, "Cannot use multiple root nodes."

    root = startpoint.nodes.index[0]
    endpoint_inds = endpoints.nodes.index
    graph = to_graph(startpoint.base)
    graph = nx.to_undirected(graph)

    # Set default for branchpoints.
    for _, data in graph.nodes(data=True):
        data.setdefault("length", 0.0)

    def edge_weight(u: int, v: int, d: Dict) -> float:
        """
        Args:
            u: Start node of an edge.
            v: End node of an edge.
            d: Dictionary of edge attributes (unused because all our attributes are
                not attributes, not edge attributes).

        Returns:
            The pathwise distance between two nodes.
        """
        return (graph.nodes[u]["length"] + graph.nodes[v]["length"]) / 2

    return [
        nx.dijkstra_path_length(graph, root, end, edge_weight) for end in endpoint_inds
    ]
