# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, Callable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import jaxley.io.graph as graph_io_new
from jaxley.utils.morph_attributes import compute_cone_props

try:
    from neuron import h

    # Load NEURON stdlib and import3d
    h.load_file("stdlib.hoc")
    h.load_file("import3d.hoc")
    NEURON_AVAILABLE = True
except ImportError:
    NEURON_AVAILABLE = False


def assert_NEURON():
    if not NEURON_AVAILABLE:
        raise ImportError(
            "NEURON is not installed. Install it with: pip install neuron\n"
            "Or install Jaxley with dev dependencies: pip install Jaxley[dev]"
        )


# NEURON's section name prefixes, mapped onto the SWC type ids.
TYPE_TO_ID = {"soma": 1, "axon": 2, "dend": 3, "apic": 4}


def contract_similar_nodes(
    graph: nx.Graph,
    attrs: Optional[List[str]] = None,
    relabel_nodes: bool = False,
    merge_rule: Optional[Callable[[nx.Graph], Tuple[Any, List[Any]]]] = None,
) -> nx.Graph:
    """Contracts similar nodes in a graph into a single node.

    Args:
        graph: A NetworkX graph.
        attrs: The attributes to consider for similarity. Defaults to `["x", "y", "z"]`.
        relabel_nodes: Whether to relabel all nodes after contraction.
        merge_rule: A function that takes a graph that contains all similar nodes and
            returns the keep node and the remove nodes (as tuple). Defaults to keeping
            the first node in the subgraph and removing the rest.

    Returns:
        The contracted graph.
    """
    attrs = ["x", "y", "z"] if attrs is None else attrs
    nodes = list(graph.nodes())
    node_attrs = [list(nx.get_node_attributes(graph, attr).values()) for attr in attrs]
    node_attrs = np.array(node_attrs).T

    dists = cdist(node_attrs, node_attrs)
    merge_pairs = np.where(np.isclose(dists, 0))
    merge_pairs = [(nodes[i], nodes[j]) for i, j in zip(*merge_pairs) if i < j]
    merge_groups = nx.Graph()
    merge_groups.add_edges_from(merge_pairs)
    sets = nx.connected_components(merge_groups)  # get disjoint sets of nodes

    if merge_rule is None:
        merge_rule = lambda x: (list(x.nodes)[0], list(x.nodes())[1:])

    for group in sets:
        keep_node, rm_nodes = merge_rule(graph.subgraph(group))
        for remove_node in rm_nodes:
            graph = nx.contracted_nodes(
                graph, keep_node, remove_node, self_loops=False, copy=False
            )
            del graph.nodes[keep_node]["contraction"]

    if relabel_nodes:
        graph = nx.relabel_nodes(graph, {n: i for i, n in enumerate(graph.nodes())})
    return graph


def swc_to_hoc(fname: str, min_radius: Optional[float] = None) -> None:
    """Load SWC file into NEURON's global h.allsec().

    This is a private helper that loads the file into NEURON's state.
    Similar to how graph backend loads the file initially.

    Args:
        fname: Path to SWC file
        min_radius: If passed, all traced radii below this value are clipped to it. The
            clipping is applied to NEURON's 3d points, so NEURON recomputes `seg.diam`,
            `seg.area()`, `seg.volume()` and `seg.ri()` from the clipped radii. Every
            reader below therefore sees the clipped morphology, exactly as if the SWC
            file itself had been clipped.

    Raises:
        ImportError: If NEURON is not installed
    """
    assert_NEURON()

    # Clear existing sections
    for sec in h.allsec():
        h.delete_section(sec=sec)

    # Load and instantiate SWC
    cell = h.Import3d_SWC_read()
    cell.input(fname)
    i3d = h.Import3d_GUI(cell, False)
    i3d.instantiate(None)

    if min_radius is not None:
        for sec in h.allsec():
            for i in range(sec.n3d()):
                if sec.diam3d(i) / 2 < min_radius:
                    h.pt3dchange(i, 2 * min_radius, sec=sec)


def hoc_to_nx(
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

    Raises:
        ImportError: If NEURON is not installed
    """
    assert_NEURON()

    # Map section types to SWC type IDs
    relevant_ids = relevant_ids or [1, 2, 3, 4]
    type2id = {k: v for k, v in TYPE_TO_ID.items() if v in relevant_ids}

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
                "radius": sec.diam3d(n) / 2,
                "p": parent,
            }

    nodes_df = pd.DataFrame(nodes).T

    node_inds = nodes_df.index
    node2idx = {**{n: i for i, n in enumerate(node_inds, start=1)}, "root": -1}
    nodes_df.index = nodes_df.index.map(node2idx)
    nodes_df["p"] = nodes_df["p"].map(node2idx)

    graph = graph_io_new.swc_to_nx(nodes_df)

    if merge_branchpoints:
        # NOTE: NEURON duplicates nodes from the original SWC and modifies their radius and id.
        # Which of the duplicated nodes is the original SWC node is ambiguous.
        # The backend matches NEURON though.
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

    All attributes are read from NEURON. `min_radius` can be enforced by passing it to
    `swc_to_hoc()`.

    Args:
        ncomp: How many compartments (segments) per section.
        drop_neuron_specific_attrs: Whether to drop the `seg_name` and `sec_name`
            attributes.

    Returns:
        A nx.Graph with the node and edge attributes.

    Raises:
        ImportError: If NEURON is not installed
    """
    assert_NEURON()

    for sec in h.allsec():
        sec.nseg = ncomp

    # collect node data and assign segment indices
    graph_attrs = {"xyzr": []}
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

        seg_bounds = np.linspace(0.0, arc[-1], sec.nseg + 1)
        segs = list(sec)

        # `sec(x).ri()` returns the full center-to-center resistance, whereas jaxley
        # expects the resistive load of each compartment half. So we integrate over the
        # frustums between the 3d points instead, which is what NEURON does internally.
        if arc[-1] > 0.0:
            *_, res_in, res_out = compute_cone_props(arc, r3d, seg_bounds)
        else:
            # Degenerate section (all 3d points coincide), so nothing to integrate over.
            # Fall back to NEURON's own `ri()`.
            norm_bounds = np.linspace(1e-10, 1.0, sec.nseg + 1)
            res_in = np.array([sec(x).ri() / sec.Ra * 1e2 for x in norm_bounds[:-1]])
            res_out = np.array([sec(x).ri() / sec.Ra * 1e2 for x in norm_bounds[1:]])

        for seg_idx, seg in enumerate(segs):
            seg_name = str(seg)
            radius = seg.diam / 2
            length = sec.L / sec.nseg
            type_name = sec_name.split("[")[0]

            # NEURON can attach a child anywhere along its parent (`sec.parentseg().x`, e.g. at
            # `soma(0.5)` for a single-point soma), but we assume that the preceding segment is
            # always the parent segment.
            if seg_idx > 0:
                parent = str(segs[seg_idx - 1])
            elif sec.parentseg():
                parent = str(list(sec.parentseg().sec)[-1])
            else:
                parent = "root"

            segments[seg_name] = {
                "seg_name": seg_name,
                "sec_name": sec_name,
                "id": TYPE_TO_ID.get(type_name, 0),
                "x": np.interp(seg.x, norm_arc, x3d),
                "y": np.interp(seg.x, norm_arc, y3d),
                "z": np.interp(seg.x, norm_arc, z3d),
                "radius": radius,
                "length": length,
                "area": seg.area(),
                "volume": seg.volume(),
                "resistive_load_in": res_in[seg_idx],
                "resistive_load_out": res_out[seg_idx],
                "parent": parent,
            }

        graph_attrs["xyzr"].append(xyzr)

    # Create DataFrames
    seg_df = pd.DataFrame(segments).T

    seg2idx = {**{seg: i for i, seg in enumerate(seg_df.index)}, "root": -1}
    seg_df.index = seg_df.index.map(seg2idx)
    seg_df["parent"] = seg_df["parent"].map(seg2idx)

    sec2idx = {sec.name(): i for i, sec in enumerate(h.allsec())}
    seg_df["global_branch_index"] = seg_df["sec_name"].map(sec2idx)

    if drop_neuron_specific_attrs:
        seg_df = seg_df.drop(columns=["seg_name", "sec_name"])

    graph = nx.Graph()
    graph.graph = graph_attrs
    for i, attrs in seg_df.iterrows():
        parent = attrs.pop("parent")
        graph.add_node(i, **attrs)
        if parent != -1:
            graph.add_edge(parent, i, comp_edge=True, synapse=False)

    graph = graph_io_new._add_jaxley_meta_data(graph)
    return graph
