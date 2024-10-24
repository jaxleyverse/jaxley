import math
import networkx as nx
import numpy as np

from functools import lru_cache
from jaxley.io.graph import to_graph


def graph_from_pointer(pointer: "Module") -> nx.Graph:
    """Create a graph from a pointer, and caches result by hash."""
    graph = to_graph(pointer)
    return graph


def nan_position_to_zero(graph: nx.Graph):
    """Sets the nan position to the origin. - often the first brach not sure why"""
    # This often is nan...
    for node in graph.nodes:
        if np.isnan(graph.nodes[node]["x"]):
            graph.nodes[node]["x"] = 0.0
        if np.isnan(graph.nodes[node]["y"]):
            graph.nodes[node]["y"] = 0.0
        if np.isnan(graph.nodes[node]["z"]):
            graph.nodes[node]["z"] = 0.0


def add_distance_to_edge(graph: nx.Graph):
    """Adds a distance attribute to each edge in the graph."""
    for e in graph.edges:
        n1 = graph.nodes[e[0]]
        n2 = graph.nodes[e[1]]

        x1, y1, z1 = n1["x"], n1["y"], n1["z"]
        x2, y2, z2 = n2["x"], n2["y"], n2["z"]

        # Euclidean distance between neighboring nodes
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        graph.edges[e]["distance"] = dist


def compute_shortest_path_distance(comp1: "View", comp2: "View"):
    mother = comp1.pointer
    mother.compute_xyz()  # Ensure xyz is computed (maybe one can check if it is already computed)
    assert mother is comp2.pointer, "Compartments must be from the same network."

    # This probably should be cached, if runtime is an issue
    # Currently, this is recomputed for each call...
    graph = graph_from_pointer(mother)
    graph = graph.to_undirected()
    nan_position_to_zero(graph)
    add_distance_to_edge(graph)

    # Get the indices of the compartments
    comp1_index = [int(i) for i in comp1.view.index]
    comp2_index = [int(i) for i in comp2.view.index]
    indices = comp1_index + comp2_index

    # Get all pairs shortest path lengths  maybe this shou
    distance_matrix = lru_cache(nx.floyd_warshall_numpy)(graph, weight="distance")

    return distance_matrix[np.ix_(indices, indices)]


def is_connected(comp1: "View", comp2: "View"):
    """Check if two compartments are connected."""
    mother = comp1.pointer
    mother.compute_xyz()  # Ensure xyz is computed (maybe one can check if it is already computed)
    assert mother is comp2.pointer, "Compartments must be from the same network."

    graph = graph_from_pointer(mother)
    graph = graph.to_undirected()

    # Get the indices of the compartments
    comp1_index = set([int(i) for i in comp1.view.index])
    comp2_index = set([int(i) for i in comp2.view.index])

    # Get all pairs shortest path lengths
    connected = any(
        nx.has_path(graph, source=comp1_idx, target=comp2_idx)
        for comp1_idx in comp1_index
        for comp2_idx in comp2_index
    )
    return connected
