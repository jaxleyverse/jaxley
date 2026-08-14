# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import numpy as np
import pandas as pd


def import_neuron_morph(fname, ncomp=8):
    from neuron import h

    _ = h.load_file("stdlib.hoc")
    _ = h.load_file("import3d.hoc")
    ncomp = 8

    ##################### NEURON ##################
    for sec in h.allsec():
        h.delete_section(sec=sec)

    cell = h.Import3d_SWC_read()
    cell.input(fname)
    i3d = h.Import3d_GUI(cell, False)
    i3d.instantiate(None)

    for sec in h.allsec():
        sec.nseg = ncomp
    return h, cell


def equal_both_nan_or_empty_df(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Return whether all elements of two dataframes are identical, NaN counting as equal.

    Column order is ignored. Missing values are compared with a mask.
    """
    a = a.drop(columns="xyzr", errors="ignore")
    b = b.drop(columns="xyzr", errors="ignore")
    if a.empty and b.empty:
        return True
    if set(a.columns) != set(b.columns) or not a.index.equals(b.index):
        return False
    b = b[a.columns]
    return bool(((a == b) | (a.isna() & b.isna())).fillna(False).all().all())


def neuron_seg_xyz(seg) -> np.ndarray:
    """Return the xyz coordinate of the center of a NEURON segment.

    NEURON only exposes the traced 3d points, so the center is interpolated along arc.
    """
    sec = seg.sec
    n3d = sec.n3d()
    arc = np.array([sec.arc3d(i) for i in range(n3d)])
    norm_arc = arc / arc[-1]
    return np.array(
        [
            np.interp(seg.x, norm_arc, [getattr(sec, f"{c}3d")(i) for i in range(n3d)])
            for c in "xyz"
        ]
    )


def neuron_section_graph():
    """Return a graph of NEURON's sections, each node at the section's center."""
    import networkx as nx
    from neuron import h

    name2idx = {sec.name(): n for n, sec in enumerate(h.allsec())}
    graph = nx.Graph()
    for n, sec in enumerate(h.allsec()):
        centers = np.stack([neuron_seg_xyz(seg) for seg in sec])
        graph.add_node(n, **dict(zip("xyz", centers.mean(axis=0))))
        if sec.parentseg():
            graph.add_edge(name2idx[sec.parentseg().sec.name()], n)
    return graph


def select_evenly_spaced_nodes(graph, num_nodes: int) -> list:
    """Select approximately evenly spaced nodes, by distance along the graph.

    Farthest-point sampling with euclidean edge lengths. Every node needs `x`, `y`, `z`.
    Ties break on coordinates, so the result does not depend on node insertion order.
    """
    import networkx as nx

    if num_nodes >= len(graph):
        return list(graph.nodes)

    pos = lambda n: np.array([graph.nodes[n][k] for k in "xyz"], dtype=float)
    graph = graph.to_undirected()
    for u, v in graph.edges:
        graph.edges[u, v]["length"] = float(np.linalg.norm(pos(u) - pos(v)))

    coord_key = lambda n: tuple(pos(n))
    start = min(
        graph.nodes, key=lambda n: (float(np.linalg.norm(pos(n))), coord_key(n))
    )
    selected = [start]
    min_dist = dict.fromkeys(graph.nodes, np.inf)
    min_dist.update(
        nx.single_source_dijkstra_path_length(graph, start, weight="length")
    )

    while len(selected) < num_nodes:
        remaining = (n for n in graph.nodes if n not in selected)
        candidate = max(remaining, key=lambda n: (min_dist[n], coord_key(n)))
        selected.append(candidate)
        dists = nx.single_source_dijkstra_path_length(graph, candidate, weight="length")
        for n, d in dists.items():
            min_dist[n] = min(min_dist[n], d)
    return selected
