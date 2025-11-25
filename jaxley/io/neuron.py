from collections import deque
from typing import Any, Callable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from neuron import h, rxd
from scipy.spatial.distance import cdist

import jaxley.io.tmp as graph_io_new

# Load NEURON stdlib and import3d
h.load_file("stdlib.hoc")
h.load_file("import3d.hoc")

unpack_dict = lambda d, keys: np.array([d[k] for k in keys])


def contract_similar_nodes(
    G: nx.Graph,
    attrs: List[str] = ["x", "y", "z"],
    relabel_nodes: bool = False,
    merge_rule: Optional[Callable[[nx.Graph], Tuple[Any, List[Any]]]] = None,
) -> nx.Graph:
    """Contracts similar nodes in a graph into a single node.

    Args:
        G: A NetworkX graph.
        attrs: The attributes to consider for similarity.
        relabel_nodes: Whether to relabel all nodes after contraction.
        merge_rule: A function that takes a graph that contains all similar nodes and
            returns the keep node and the remove nodes (as tuple). Defaults to keeping
            the first node in the subgraph and removing the rest.

    Returns:
        The contracted graph.
    """
    nodes = list[Any](G.nodes())
    node_attrs = [list(nx.get_node_attributes(G, attr).values()) for attr in attrs]
    node_attrs = np.array(node_attrs).T

    dists = cdist(node_attrs, node_attrs)
    merge_pairs = np.where(np.isclose(dists, 0))
    merge_pairs = [(nodes[i], nodes[j]) for i, j in zip(*merge_pairs) if i < j]
    merge_groups = nx.Graph()
    merge_groups.add_edges_from(merge_pairs)
    sets = nx.connected_components(merge_groups)  # get disjoint sets of nodes

    if merge_rule is None:
        merge_rule = lambda x: (list(x.nodes)[0], list(x.nodes())[1:])

    for set in sets:
        keep_node, rm_nodes = merge_rule(G.subgraph(set))
        for remove_node in rm_nodes:
            G = nx.contracted_nodes(
                G, keep_node, remove_node, self_loops=False, copy=False
            )
            del G.nodes[keep_node]["contraction"]

    if relabel_nodes:
        G = nx.relabel_nodes(G, {n: i for i, n in enumerate(G.nodes())})
    return G


def _load_swc_into_neuron(fname: str) -> None:
    """Load SWC file into NEURON's global h.allsec().

    This is a private helper that loads the file into NEURON's state.
    Similar to how graph backend loads the file initially.

    Args:
        fname: Path to SWC file
    """
    # Clear existing sections
    for sec in h.allsec():
        h.delete_section(sec=sec)

    # Load and instantiate SWC
    cell = h.Import3d_SWC_read()
    cell.input(fname)
    i3d = h.Import3d_GUI(cell, False)
    i3d.instantiate(None)


def h_allsec_to_nx(
    relevant_ids: Optional[List[int]] = None, merge_branchpoints: bool = True
) -> nx.Graph:
    """
    Reads from the global `h.allsec()` and writes the attributes to a nx.Graph. The
    edges are created by connecting the first point of a section to the last point of the
    parent section and connecting up each point (line in the SWC file) within a section.
    Each n3d point becomes a node in the graph.

    The graph is comparable to Jaxley's SWC-Graph.

    The node attributes are:
    - id: The SWC type ID.
    - x: The x-coordinate.
    - y: The y-coordinate.
    - z: The z-coordinate.
    - r: The radius.

    Args:
        relevant_ids: The section types to include in the graph.
            Defaults to `[1, 2, 3, 4]`.
        merge_branchpoints: NEURON duplicates branchpoints to start a new section. This
            leads to nodes with the same x,y,z coordinates. To mimic jaxley graph, we
            contract these nodes into a single branchpoint node.

    Returns:
        A nx.Graph with the node and edge attributes.
    """
    # Map section types to SWC type IDs
    relevant_ids = relevant_ids or [1, 2, 3, 4]
    type2id = {"soma": 1, "axon": 2, "dend": 3, "apic": 4}
    type2id = {k: v for k, v in type2id.items() if k in relevant_ids}

    nodes = {}
    for sec in h.allsec():
        sec_name = sec.name()
        sec_type = sec_name.split("[")[0]
        sec_id = type2id.get(sec_type, 0)

        # nodes
        for n in range(sec.n3d()):
            if sec.parentseg() and n == 0:
                parent_sec = sec.parentseg().sec
                parent_n3d = parent_sec.n3d()
                parent = parent_sec.name() + f"({parent_n3d-1})"
            elif n == 0:
                parent = "root"  # root node
            else:
                parent = f"{sec_name}({n-1})"

            nodes[f"{sec_name}({n})"] = {
                "id": sec_id,
                "x": sec.x3d(n),
                "y": sec.y3d(n),
                "z": sec.z3d(n),
                "r": sec.diam3d(n) / 2,
                "p": parent,
            }

    nodes_df = pd.DataFrame(nodes).T

    node_inds = nodes_df.index
    node2idx = {**{n: i for i, n in enumerate(node_inds, start=1)}, "root": -1}
    nodes_df.index = nodes_df.index.map(node2idx)
    nodes_df["p"] = nodes_df["p"].map(node2idx)

    graph = graph_io_new.swc_to_nx(nodes_df)

    if merge_branchpoints:
        # TODO: find out how to contract branchpoints to mimic jaxley graph
        # (i.e. which of the merged nodes is the orignal swc node? vs which have
        # already id and radius modded)
        graph = contract_similar_nodes(graph)
    return graph


def build_compartment_graph(
    ncomp: int = 1, drop_neuron_specific_attrs: bool = True
) -> nx.Graph:
    """
    Reads from the global `h.allsec()` and constructs a compartment graph from the ingested
    SWC file. Each section is divided into `ncomp` compartments and each compartment will
    become a node in the graph. Each node will contain the attributes that are used to
    simulate each compartment.

    The graph is comparable to Jaxley's Compartment-Graph and can be used to compare / debug
    Jaxley's and NEURON's SWC readers. Especially how NEURON and Jaxley handle computation
    of the compartment / segment attributes, like radius, volume, surface area, etc.

    The node attributes are:
    - comp_index: The compartment index.
    - seg_name: The name of the segment. (neuron specific)
    - sec_name: The name of the section. (neuron specific)
    - x: The x-coordinate.
    - y: The y-coordinate.
    - z: The z-coordinate.
    - radius: The radius.
    - area: The area.
    - surface_area: The surface area.
    - volume: The volume.
    - length: The length.
    - groups: The groups.

    Returns:
        A nx.Graph with the node and edge attributes.
    """
    for sec in h.allsec():
        sec.nseg = ncomp

    type2id = {"soma": 1, "axon": 2, "dend": 3, "apic": 4}

    # Dummy cytosolic region
    cyt = rxd.Region(h.allsec(), name="cyt")

    # Create a dummy species to get access to segment volumes
    ca = rxd.Species(cyt, name="ca", d=0)
    seg2ca_node = {str(node.segment): node for node in ca.nodes}

    # collect node data and assign segment indices
    graph_attrs = {"xyzr": [], "branchpoints_and_tips": []}
    segments = {}
    for sec in h.allsec():
        sec_name = sec.name()

        n3d = sec.n3d()
        arc = np.array([sec.arc3d(i) for i in range(n3d)])  # Cumulative arc lengths
        norm_arc = arc / arc[-1]
        x3d = np.array([sec.x3d(i) for i in range(n3d)])
        y3d = np.array([sec.y3d(i) for i in range(n3d)])
        z3d = np.array([sec.z3d(i) for i in range(n3d)])
        r3d = np.array([sec.diam3d(i) / 2 for i in range(n3d)])
        xyzr = np.array([x3d, y3d, z3d, r3d]).T

        for seg_idx, seg in enumerate(sec):
            seg_name = str(seg)
            node = seg2ca_node.get(seg_name)
            radius = seg.diam / 2
            length = sec.L / sec.nseg
            type_name = sec_name.split("[")[0]

            if sec.parentseg() and seg_idx == 0:
                parent_sec = sec.parentseg().sec
                parent = str(deque(iter(parent_sec), maxlen=1).pop())
            elif seg_idx == 0:
                parent = "root"
            else:
                parent = f"{seg_name}"

            segments[seg_name] = {
                "seg_name": seg_name,
                "sec_name": sec_name,
                "id": type2id.get(type_name, 0),
                "x": np.interp(seg.x, norm_arc, x3d),
                "y": np.interp(seg.x, norm_arc, y3d),
                "z": np.interp(seg.x, norm_arc, z3d),
                "r": radius,
                "l": length,
                "area": node.surface_area,  # seg.area()
                "volume": node.volume,
                "resistive_load_in": length / 2 / radius**2 / np.pi,
                "resistive_load_out": length / 2 / radius**2 / np.pi,
                "parent": parent,
            }
            graph_attrs["xyzr"].append(xyzr)

    # Create DataFrames
    seg_df = pd.DataFrame(segments).T

    seg2idx = {**{seg: i for i, seg in enumerate(seg_df.index)}, "root": -1}
    seg_df.index = seg_df.index.map(seg2idx)
    seg_df["parent"] = seg_df["parent"].map(seg2idx)
    seg_df["is_comp"] = True

    sec2idx = {sec.name(): i for i, sec in enumerate(h.allsec())}
    seg_df["branch_index"] = seg_df["sec_name"].map(sec2idx)

    if drop_neuron_specific_attrs:
        seg_df = seg_df.drop(columns=["seg_name", "sec_name"])

    graph = nx.Graph()
    graph.graph = graph_attrs
    for i, attrs in seg_df.iterrows():  # tolist: np.float64 -> float
        parent = attrs.pop("parent")
        graph.add_node(i, **attrs)
        if parent != -1:
            graph.add_edge(parent, i, comp_edge=True, synapse=False)

    branch_inds = nx.get_node_attributes(graph, "branch_index")
    for i, j in graph.edges():
        graph.edges[i, j]["branch_edge"] = branch_inds[i] != branch_inds[j]

    graph = graph_io_new._add_jaxley_meta_data(graph)
    return graph
