# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from jaxley.modules import Branch, Cell, Compartment, Network
from jaxley.utils.cell_utils import v_interp
from jaxley.utils.morph_attributes import (
    morph_attrs_from_xyzr,
    split_xyzr_into_equal_length_segments,
)

########################################################################################
###################################### HELPERS #########################################
########################################################################################


def _is_leaf(graph: nx.Graph, node: Any) -> bool:
    """Return whether or not a node in an undirected graph is a leaf."""
    # For a leaf node, the degree is is 1. For a single compartment neuron it is 0.
    return graph.degree(node) < 2


def _is_branching(graph: nx.Graph, node: Any) -> bool:
    """Return whether an undirected graph is branching at a particular node."""
    return graph.degree(node) > 2


def _has_same_id(
    graph: nx.Graph,
    node_i: Any,
    node_j: Any,
    relevant_type_ids: List[int],
):
    """Return whether two nodes in a graph have the same value for the `id` attribute.

    Args:
        relevant_type_ids: All type ids that are not in this list will be ignored for
            tracing the morphology. This means that branches which have multiple type
            ids (which are not in `relevant_type_ids`) will be considered as one branch.
    """
    if (
        graph.nodes[node_i]["id"] not in relevant_type_ids
        and graph.nodes[node_j]["id"] not in relevant_type_ids
    ):
        return True
    else:
        return graph.nodes[node_i]["id"] == graph.nodes[node_j]["id"]


def _get_soma_idxs(graph: nx.Graph):
    """Return all SWC nodes which have id=1, i.e. which are labelled as soma."""
    return [i for i, n in nx.get_node_attributes(graph, "id").items() if n == 1]


def _unpack(d: Dict, keys: list[str]) -> List:
    """Return all values of a dictionary whose key is in `keys`."""
    return [d[k] for k in keys]


def _branch_e2n(branch_edges):
    """Return all nodes given the edges within a branch.

    E.g. `branch_edges = [(0, 1), (1, 2)]` -> `[0, 1, 2]`.
    """
    array = np.concatenate(branch_edges)
    return array[np.sort(np.unique(array, return_index=True)[1])]


def _branch_n2e(branch_nodes):
    """Return all edges given the nodes within a branch.

    E.g. `branch_nodes = [0, 1, 2]` -> [(0, 1), (1, 2)]`"""
    return [e for e in zip(branch_nodes[:-1], branch_nodes[1:])]


def _find_root(G: nx.Graph):
    """Return a possible root for tracing the graph.

    Roots are nodes which have degree = 0."""
    roots = [n for n in sorted(G.nodes) if _is_leaf(G, n)]
    return roots[0]


########################################################################################
################################### BUILD SWC GRAPH ####################################
########################################################################################


def to_swc_graph(fname: str, num_lines: int = None) -> nx.DiGraph:
    """Read a SWC file and return a SWC graph via networkX.

    The graph is read such that each entry in the swc file becomes a graph node
    with the column attributes (id, x, y, z, r). Then each node is connected to its
    designated parent via an edge. A "type" attribute is added to the graph to identify
    its processing stage for subsequent steps.

    Args:
        fname: Path to the swc file.
        num_lines: Number of lines to read from the file. If None, all lines are read.

    Returns:
        A networkx graph of the traced morphology in the swc file. It has attributes:
        nodes: {'id': 1, 'x': 0.0, 'y': 0.0, 'z': 0.0, 'r': 1.0, 'p': -1}
        edges: {'l': 1.0}

    Example usage
    ^^^^^^^^^^^^^

    ::

        from jaxley.io.graph to_swc_graph
        swc_graph = to_swc_graph("path_to_swc.swc")
    """
    i_id_xyzr_p = np.loadtxt(fname)[:num_lines]

    graph = nx.DiGraph()
    graph.add_nodes_from(
        (
            (int(i), {"id": int(id), "x": x, "y": y, "z": z, "r": r, "p": int(p)})
            for i, id, x, y, z, r, p in i_id_xyzr_p
        )
    )
    graph.add_edges_from([(p, i) for p, i in i_id_xyzr_p[:, [-1, 0]] if p != -1])
    return _add_missing_graph_attrs(graph)


def _add_missing_graph_attrs(graph: nx.Graph) -> nx.Graph:
    """Add missing attributes to the graph nodes and edges.

    The following attributes are added to the graph:
    - id: int (default: 0)
    - x, y, z: float (default: NaN)
    - r: float (default: 1)
    - l: float (default: 1)
    - p: int (default: 0)

    Args:
        graph: A networkx graph.

    Returns:
        The graph with the added attributes."""
    available_keys = graph.nodes[1].keys()
    defaults = {
        "id": 0,
        "x": float("nan"),
        "y": float("nan"),
        "z": float("nan"),
        "r": 1,
        "p": 0,
    }
    # add defaults if not present
    for key in set(defaults.keys()).difference(available_keys):
        nx.set_node_attributes(graph, defaults[key], key)

    graph = _add_edge_lengths(graph)
    edge_lens = nx.get_edge_attributes(graph, "l")
    if np.isnan(list(edge_lens.values())[0]):
        nx.set_edge_attributes(graph, 1, "l")

    return graph


def _add_edge_lengths(graph: nx.Graph, min_len: float = 1e-5) -> nx.DiGraph:
    """Add edge lengths to graph.edges based on the xyz coordinates of graph.nodes."""
    xyz = lambda i: np.array(_unpack(graph.nodes[i], "xyz"))
    for i, j in graph.edges:
        d_ij = (
            np.sqrt(((xyz(i) - xyz(j)) ** 2).sum())
            if i != j
            else 2 * graph.nodes[i]["r"]
        )
        # min_len ensures that 2 nodes cannot lie on top of each other
        # this is important for the compartmentalization
        graph.edges[i, j]["l"] = d_ij if d_ij >= min_len else min_len
    return graph


########################################################################################
############################## BUILD COMPARTMENT GRAPH #################################
########################################################################################


def build_compartment_graph(
    swc_graph: nx.DiGraph,
    ncomp: int,
    root: Optional[int] = None,
    min_radius: Optional[float] = None,
    max_len=None,
    ignore_swc_tracing_interruptions=True,
    relevant_type_ids: Optional[List[int]] = None,
) -> nx.DiGraph:
    """Return a networkX graph that indicates the compartment structure.

    Build a new graph made up of compartments in every branch. These compartments are
    spaced at equidistant points along the branch. Node attributes, like radius are
    linearly interpolated along its length.

    Example: 4 compartments | edges = - | nodes = o | comp_nodes = x
    o-----------o----------o---o---o---o--------o
    o-------x---o----x-----o--xo---o---ox-------o

    This function returns a _directed_ graph. The graph is directed only because every
    compartment tracks the xyzr coordinates of the associated SWC file. These xyzr
    coordinates are ordered by the order of the traversal of the swc_graph. In later
    methods (e.g. build_solve_graph), we traverse the `comp_graph` and mostly ignore
    the directionality of the edges, but we only use the directionality to reverse the
    xyzr coordinates if necessary.

    Args:
        swc_graph: Graph generated by `to_swc_graph()`.
        ncomp: How many compartments per branch to insert.
        root: The root branch from which to start tracing the nodes. This defines the
            branch indices.
        min_radius: Minimal radius for each compartment.
        max_len: Maximal length for each branch. Longer branches are split into
            separate branches.
        ignore_swc_tracing_interruptions: If `False`, it this function automatically
            starts a new branch when a section is traced with interruptions.
        relevant_type_ids: All type ids that are not in this list will be ignored for
            tracing the morphology. This means that branches which have multiple type
            ids (which are not in `relevant_type_ids`) will be considered as one branch.
            If `None`, we default to `[1, 2, 3, 4]`.

    Returns:
        Directed graph made up of compartments.

        Compartment nodes have attributes (example):
        {'x': 2.0,
        'y': 0.0,
        'z': 0.0,
        'branch_index': 2,
        'comp_index': 2,
        'type': 'comp',
        'xyzr': array([[0., 0., 0., 1.], [1., 0., 0., 1.]]),
        'groups': ['soma'],
        'radius': 1.0,
        'length': 4.0,
        'cell_index': 0}

        Between-compartment nodes have attributes:
        {'x': 0.0,
        'y': 0.0,
        'z': 0.0,
        'p': -1,
        'type': 'branchpoint',
        'groups': ['soma'],
        'radius': 1.0,
        'length': 0.0,
        'cell_index': 0}

        Edges have attributes: {}

    Example usage
    ^^^^^^^^^^^^^

    ::

        from jaxley.io.graph to_swc_graph, build_compartment_graph
        swc_graph = to_swc_graph("path_to_swc.swc")
        comp_graph = build_compartment_graph(swc_graph, ncomp=1)
    """
    swc_graph, branch_edge_indices, all_type_ids, soma_ignore_inds = _trace_branches(
        swc_graph,
        root=root,
        max_len=max_len,
        ignore_swc_tracing_interruptions=ignore_swc_tracing_interruptions,
        relevant_type_ids=relevant_type_ids,
    )
    comp_offset = 0

    # See docstring for why `comp_graph` a directed graph.
    comp_graph = nx.DiGraph()

    # Get branchpoints.
    branchpoint_inds = [
        n for n in swc_graph.nodes if swc_graph.nodes[n]["type"] == "branchpoint"
    ]
    comp_graph.add_nodes_from((n, swc_graph.nodes[n]) for n in branchpoint_inds)
    nx.set_node_attributes(comp_graph, 0.0, "l")

    for branch_index, branch_edge_inds in enumerate(branch_edge_indices):
        path_lens = np.cumsum(np.concatenate([[0], branch_edge_inds[:, 2]]))

        # [:, :2] because `branch_edge_inds` contains `(start_node, end_note, length)`.
        branch_nodes = _branch_e2n(branch_edge_inds[:, :2])

        # `branch_data` is a pd.DataFrame which contains all SWC nodes of the current
        # branch.
        branch_data = pd.DataFrame([swc_graph.nodes[i] for i in branch_nodes])
        branch_data["node_index"] = branch_nodes
        branch_data["l"] = path_lens

        # errors="ignore" because user-defined graphs might not have the `p` attribute.
        branch_data = branch_data.drop(columns=["type", "p"], errors="ignore")

        # fix id and r bleed over from neighboring neurites of a different type
        if branch_data.loc[0, "id"] != branch_data.loc[1, "id"]:
            branch_data.loc[0, ["r", "id"]] = branch_data.loc[1, ["r", "id"]]

        xyzr = branch_data[["x", "y", "z", "r"]].to_numpy()

        # `soma_ignore_inds` tracks all node indices which are part of a single-point-
        # soma or of a somatic branchpoint. In these cases, the somatic SWC node
        # is _not_ considered to be part of the dendrite. Here, we delete somatic
        # SWC node from the xyzr of the dendrite.
        if branch_edge_inds[0, 0] in soma_ignore_inds:
            xyzr = xyzr[1:]

        # Here, we split xyzr into compartments.
        xyzr_per_comp = split_xyzr_into_equal_length_segments(xyzr, ncomp)
        morph_attrs = np.asarray(
            [morph_attrs_from_xyzr(xyzr, min_radius, ncomp) for xyzr in xyzr_per_comp]
        )

        branch_len = branch_data["l"].max()
        if branch_len < 1e-8:
            warn(
                "Found a branch with length 0. To avoid NaN while integrating the "
                "ODE, we capped this length to 0.1 um. The underlying cause for the "
                "branch with length 0 is likely a strange SWC file. The "
                "most common reason for this is that the SWC contains a soma "
                "traced by a single point, and a dendrite that connects to the soma "
                "has no further child nodes."
            )
            branch_len = 0.1
        comp_len = branch_len / ncomp
        locs = np.linspace(comp_len / 2, branch_len - comp_len / 2, ncomp)

        # New branch_nodes is a pd.DataFrame which contains all ncomp compartments that
        # make up the branch.
        new_branch_nodes = v_interp(locs, branch_data["l"].values, branch_data.values)
        new_branch_nodes = pd.DataFrame(
            np.array(new_branch_nodes.T), columns=branch_data.columns
        )
        new_branch_nodes["id"] = all_type_ids[
            branch_index
        ]  # new_branch_nodes["id"].astype(int)
        new_branch_nodes["l"] = comp_len
        new_branch_nodes["branch_index"] = branch_index
        new_branch_nodes["comp_index"] = comp_offset + np.arange(ncomp)
        num_nodes = max(comp_graph.nodes) + 1 if comp_graph.nodes else 0
        new_branch_nodes["node_index"] = num_nodes + np.arange(ncomp)
        new_branch_nodes["type"] = "comp"
        comp_offset += ncomp
        new_branch_nodes["xyzr"] = xyzr_per_comp
        new_branch_nodes["r"] = morph_attrs[:, 0]
        new_branch_nodes["area"] = morph_attrs[:, 1]
        new_branch_nodes["volume"] = morph_attrs[:, 2]
        new_branch_nodes["resistive_load_in"] = morph_attrs[:, 3]
        new_branch_nodes["resistive_load_out"] = morph_attrs[:, 4]

        # Add the compartments as nodes to the new `comp_graph`.
        new_branch_nodes = new_branch_nodes.set_index("node_index")
        comp_graph.add_nodes_from(new_branch_nodes.to_dict(orient="index").items())

        # Add the edges between compartments within a node.
        new_branch_edges = _branch_n2e(new_branch_nodes.index)
        comp_graph.add_edges_from(new_branch_edges)

        # Add edge from compartment ending & beginning to branchpoint.
        pre_branch_node = branch_edge_inds[0, 0]
        post_branch_node = branch_edge_inds[-1, 1]
        if (
            pre_branch_node in comp_graph.nodes
            and comp_graph.nodes[pre_branch_node]["type"] == "branchpoint"
        ):
            comp_graph.add_edge(pre_branch_node, new_branch_nodes.index[0])
        if (
            post_branch_node in comp_graph.nodes
            and comp_graph.nodes[post_branch_node]["type"] == "branchpoint"
        ):
            comp_graph.add_edge(new_branch_nodes.index[-1], post_branch_node)

    # Delete `comp_index`,... from branchpoint nodes.
    for node, attrs in comp_graph.nodes(data=True):
        if comp_graph.nodes[node]["type"] == "branchpoint":
            for key in ["comp_index", "branch_index", "xyzr"]:
                if key in attrs:
                    del comp_graph.nodes[node][key]

    # Rename all nodes to have the compartment index as node name _if they are
    # compartments_ and otherwise to have a larger indices as node names
    # (for branchpoints).
    mapping = {}
    branchpoint_index = 0
    for n, attrs in comp_graph.nodes(data=True):
        if "comp_index" in attrs:
            mapping[n] = attrs["comp_index"]
        else:
            mapping[n] = f"n{branchpoint_index}"
            branchpoint_index += 1
    comp_graph = nx.relabel_nodes(comp_graph, mapping, copy=True)
    comp_graph = _set_branchpoint_indices(comp_graph)

    min_radius = min_radius if min_radius else 0.0

    # Rename attributes.
    # Description of SWC file format:
    # http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
    group_ids = {0: "undefined", 1: "soma", 2: "axon", 3: "basal", 4: "apical"}
    min_radius = min_radius if min_radius else 0.0
    clip_radius = lambda r: max(r, min_radius) if min_radius else r
    for n in comp_graph.nodes:
        type_id = comp_graph.nodes[n].pop("id")
        comp_graph.nodes[n]["groups"] = [group_ids.get(type_id, f"custom{type_id}")]
        comp_graph.nodes[n]["radius"] = clip_radius(comp_graph.nodes[n].pop("r"))
        comp_graph.nodes[n]["length"] = comp_graph.nodes[n].pop("l")

    # Here, we assume that the SWC file was all from a single neuron. However, this is
    # fine anyways because we assert that it is a tree.
    nx.set_node_attributes(comp_graph, 0, "cell_index")
    return comp_graph


def _trace_branches(
    swc_graph: nx.DiGraph,
    root: Optional[int] = None,
    max_len: Optional[float] = None,
    ignore_swc_tracing_interruptions: bool = True,
    relevant_type_ids: Optional[List[int]] = None,
) -> List[np.ndarray]:
    """Get all uninterrupted paths in a graph (i.e. branches).

    The graph is traversed depth-first starting from the source node, which is the only
    node with in_degree 0 (raises in case multiple are found). Note: Traversal order
    can be changed, by reversing the edge directions, i.e. to start traversal from a
    leaf node set: [source -> ... -> leaf] to [leaf -> ... -> source].

    The `graph` is modified in two ways:
    - For single-point somata we modify some things. In addition, we make the graph
    undirected.
    - Add `branchpoint: bool` as a node attribute.

    Args:
        graph: A networkx graph.
        ignore_swc_tracing_interruptions: Whether to ignore discontinuities in the swc
            tracing order. If False, this will result in split branches at these points.
        relevant_type_ids: All type ids that are not in this list will be ignored for
            tracing the morphology. This means that branches which have multiple type
            ids (which are not in `relevant_type_ids`) will be considered as one branch.
            If `None`, we default to `[1, 2, 3, 4]`.

    Returns:
        A list of linear paths in the graph. Each path is represented as an array of
        edges.
    """
    if relevant_type_ids is None:
        relevant_type_ids = [1, 2, 3, 4]

    # `soma_ignore_inds` tracks the node indices of single-point-somata or of
    # somatic branchpoints.
    soma_ignore_inds = []

    # Handle special case of a single soma node.
    soma_idxs = _get_soma_idxs(swc_graph)
    if len(soma_idxs) == 1:
        soma = soma_idxs[0]

        for i, j in (*swc_graph.in_edges(soma), *swc_graph.out_edges(soma)):
            swc_graph.edges[i, j]["l"] = 0

        # For single-point somatata, we set
        # l = 2*r ensures A_cylinder = 2*pi*r*l = 4*pi*r^2 = A_sphere.
        # Here, we add another compartment such that there exists an `edge` between
        # that new compartment and the single-point-soma of appropriate length.
        swc_graph.nodes[1]["p"] = 0
        for n in swc_graph.nodes:
            swc_graph.nodes[n]["p"] += 1
        swc_graph.add_node(0, **swc_graph.nodes[1])
        swc_graph.nodes[0]["p"] = -1
        swc_graph.add_edge(0, soma, l=2 * swc_graph.nodes[soma]["r"])
        swc_graph = nx.relabel_nodes(swc_graph, {i: i + 1 for i in swc_graph.nodes})
        soma_ignore_inds.append(1)
        soma_ignore_inds.append(soma + 1)

    undir_swc_graph = swc_graph.to_undirected()

    # Handle somatic branchpoints. A somatic branchpoint is a branchpoint at which at
    # least two connecting branches are somatic. In that case (and in the case of a
    # single-point soma), non-somatic branches are assumed to start from their first
    # traced point, not from the soma.
    for node in soma_idxs:
        somatic_neighbors = [
            n
            for n in undir_swc_graph.neighbors(node)
            if undir_swc_graph.nodes[n]["id"] == 1
        ]
        if len(somatic_neighbors) > 1:
            for i, j in undir_swc_graph.edges(node):
                if undir_swc_graph.nodes[j]["id"] != 1:
                    soma_ignore_inds.append(i)
                    undir_swc_graph.edges[i, j]["l"] = 0

    branches, current_branch, all_type_ids = [], [], []

    # Traverse the SWC graph and identify which nodes belong to one branch. This builds
    # a list `branches: List` where each elements is a `np.array` of shape (N, 3).
    # The `3` are `swc_parent, swc_node, length` of all SWC edges within a branch.
    root = root if root else _find_root(undir_swc_graph)

    # We first set the type of all SWC nodes to be "spurious". Later on, we change the
    # type of branchpoints to `branchpoint`.
    nx.set_node_attributes(undir_swc_graph, "spurious", "type")
    # `sort_neighbors=lambda x: sorted(x)` to first handle edges with a low node index.
    for i, j in nx.dfs_edges(undir_swc_graph, root, sort_neighbors=lambda x: sorted(x)):
        current_edge = (i, j)
        current_len = undir_swc_graph.edges[current_edge]["l"]
        current_type_id = undir_swc_graph.nodes[i]["id"]
        current_branch += [(i, j, current_len)]
        if _is_leaf(undir_swc_graph, j):
            # If the SWC leaf node has a different type_id than the its SWC
            # predecessor, then we explicitly must add both branches here.
            if not _has_same_id(undir_swc_graph, i, j, relevant_type_ids):
                # Add the branch that goes up until the last edge.
                branches.append(current_branch[:-1])
                all_type_ids.append(current_type_id)

                # Add the branch made up of just the last edge.
                branches.append(current_branch[-1:])
                all_type_ids.append(undir_swc_graph.nodes[j]["id"])

                current_branch = []
            else:
                branches.append(current_branch)
                all_type_ids.append(current_type_id)
                current_branch = []

        elif _is_branching(undir_swc_graph, j):
            branches.append(current_branch)
            all_type_ids.append(current_type_id)
            current_branch = []

        # Start new branch if ids differ.
        elif not _has_same_id(undir_swc_graph, i, j, relevant_type_ids):
            # Consider the SWC graph:
            # 1  1  1  2  2  2 (number indicates type_id)
            if not swc_graph.has_edge(i, j) and swc_graph.has_edge(j, i):
                branches.append(current_branch)
                all_type_ids.append(current_type_id)
                current_branch = []
            else:
                branches.append(current_branch[:-1])
                all_type_ids.append(current_type_id)
                current_branch = [current_branch[-1]]

    branch_edges = []
    type_inds = []
    for i, p in enumerate(branches):
        if len(p) > 0:
            branch_edges.append(np.array(p))
            type_inds.append(all_type_ids[i])

    if max_len:
        edge_lens = nx.get_edge_attributes(undir_swc_graph, "l")
        additional_branchpoints, branch_edges, type_inds = _split_branches(
            branch_edges, type_inds, edge_lens, max_len
        )
        for b in additional_branchpoints:
            undir_swc_graph.nodes[b]["type"] = "branchpoint"

    # Label nodes in the swc_graph as `branchpoint`.
    #
    # The very first branch will start at a tip, not a branchpoint (therefore:
    # `branch_edges[1:]`). All other branches start at a branch-point. To get all
    # branchpoints, we loop over all branches but the first one and get their first
    # traced node.
    for b in branch_edges[1:]:
        tip = b[0, 0]
        undir_swc_graph.nodes[tip]["type"] = "branchpoint"

    if not ignore_swc_tracing_interruptions:
        # Ignore added index by default; only relevant in case it was added.
        additional_branchpoints, branch_edges, type_inds = (
            _split_branches_if_swc_nodes_were_traced_with_interruption(
                undir_swc_graph, branch_edges, type_inds
            )
        )
        for b in additional_branchpoints:
            undir_swc_graph.nodes[b]["type"] = "branchpoint"

    return undir_swc_graph, branch_edges, type_inds, soma_ignore_inds


def _split_branches(
    branches: List[np.ndarray], type_inds, edge_lens: Dict, max_len: int = 1000
) -> List[np.ndarray]:
    """Split branches into approximately equally long sections <= max_len.

    Args:
        branches: List of branches represented as arrays of edges.
        edge_lens: Dict for length of each edge in the graph.
        max_len: Maximum length of a branch section. If a branch exceeds this length,
            it is split into equal parts.

    Returns:
        A list of branches, where each branch is split into sections of
        length <= max_len.
    """
    # TODO: split branches into exactly equally long sections
    edge_lens.update({(j, i): l for (i, j), l in edge_lens.items()})
    additional_branchpoints, new_branches, new_type_inds = [], [], []
    for branch, type_ind in zip(branches, type_inds):
        cum_branch_len = np.cumsum([edge_lens[i, j] for i, j, _ in branch])

        k = cum_branch_len // max_len
        split_branch = [branch[np.where(np.array(k) == kk)[0]] for kk in np.unique(k)]
        new_branches += split_branch
        new_type_inds += [type_ind] * len(split_branch)

        # Introduce additional branchpoints.
        #
        # Ignore the first one (via [1:]) because that node is anyways labeled as a
        # branchpoint.
        additional_branchpoints += [int(branch[0, 0]) for branch in split_branch[1:]]

    return additional_branchpoints, new_branches, new_type_inds


def _split_branches_if_swc_nodes_were_traced_with_interruption(
    graph: nx.Graph, branches: List[np.ndarray], type_inds: List[int]
) -> List[np.ndarray]:
    """Simulate swc trace errors in the branches.

    Both NEURON and Jaxley's hand coded swc reader introduce breaks in the trace
    if the same neurite was traced in disconnected pieces. Since `swc_to_graph` is
    agnostic to the order of the tracing, it does not suffer from this issue. Hence,
    to artificially force this behaviour (to compare to the other parsers), this
    function can be used to simulate these errors. See
    `_find_swc_tracing_interruptions` for how to identify these points in the graph.

    Args:
        graph: A networkx graph of a traced morphology.
        branches: List of branches represented as arrays of edges.

    Returns:
        A list of branches with simulated trace errors.
    """
    node_idxs = _find_swc_tracing_interruptions(graph)
    for node_idx in node_idxs:
        # Get index of the first branch in which `node_idx` appears.
        # [:, :2] to get rid of the length. `p` is of shape `(num_nodes_in_branch, 3)`,
        # where the 3 are `parent, node, length`.
        branch_idx = next(i for i, p in enumerate(branches) if node_idx in p[:, :2])
        b4, branch, after = (
            branches[:branch_idx],
            branches[branch_idx],
            branches[branch_idx + 1 :],
        )
        type_b4, type_val, type_after = (
            type_inds[:branch_idx],
            type_inds[branch_idx],
            type_inds[branch_idx + 1 :],
        )
        # [:, :2] to get rid of the length. `p` is of shape `(num_nodes_in_branch, 3)`,
        # where the 3 are `parent, node, length`.
        break_idx = np.where(branch[:, :2] == node_idx)[0][1]
        # insert artificial break into branch
        branches = b4 + [branch[:break_idx], branch[break_idx:]] + after
        type_inds = type_b4 + [type_val, type_val] + type_after
    return node_idxs, branches, type_inds


def _find_swc_tracing_interruptions(graph: nx.Graph) -> np.ndarray:
    """Identify discontinuities in the swc tracing order.

    Some swc files contain artefacts, where tracing of the same neurite was done
    in disconnected pieces. Both NEURON and Jaxley's hand coded swc reader introduce
    a break in the trace at these points, since they parse the file in order. This
    leads to split branches, which should be one. This function identifies these
    points in the graph.

    Example swc file:
    # branch 1
    1 1 0.0 0.0 0.0 1.0 -1
    2 1 1.0 0.0 0.0 1.0 1
    3 1 2.0 0.0 0.0 1.0 2
    # branch 2
    4 2 3.0 1.0 0.0 1.0 3
    5 2 4.0 2.0 0.0 1.0 4
    # branch 3
    6 3 3.0 -1.0 0.0 1.0 3
    7 3 4.0 -2.0 0.0 1.0 6
    8 3 5.0 -3.0 0.0 1.0 7
    # amend branch 2
    9 4 5.0 3.0 0.0 1.0 5

    Args:
        graph: graph tracing of swc file (from `swc_to_graph`).

    Returns:
        An array of node indices where tracing is discontinuous.
    """
    interrupted_nodes = []
    for n in graph.nodes:
        parent = graph.nodes[n]["p"]
        # Parent should be previous node.
        if parent > -1 and parent != n - 1:
            node_is_no_branchpoint = graph.nodes[n]["type"] != "branchpoint"
            parent_is_no_branchpoint = graph.nodes[parent]["type"] != "branchpoint"
            if node_is_no_branchpoint and parent_is_no_branchpoint:
                interrupted_nodes.append(parent)

    return interrupted_nodes


def _set_branchpoint_indices(jaxley_graph: nx.DiGraph) -> nx.DiGraph:
    """Return a graph whose branchpoint indices match those of a `jx.Module`.

    Here, we ensure that the branchpoints are enumerated in the same way in the
    module as they are in the graph. The ordering is by the branch_index of the
    parent branch of a branchpoint.
    """
    predecessor_branch_inds = []
    branchpoints = []
    max_comp_index = 0
    for node in jaxley_graph.nodes:
        if jaxley_graph.nodes[node]["type"] == "branchpoint":
            predecessor = list(jaxley_graph.predecessors(node))[0]
            predecessor_branch_inds.append(
                jaxley_graph.nodes[predecessor]["branch_index"]
            )
            branchpoints.append(node)
        else:
            if jaxley_graph.nodes[node]["comp_index"] > max_comp_index:
                max_comp_index = jaxley_graph.nodes[node]["comp_index"]
    sorting = np.argsort(predecessor_branch_inds)
    branchpoints_in_corrected_order = np.asarray(branchpoints)[sorting]
    mapping = {
        k: max_comp_index + i + 1 for i, k in enumerate(branchpoints_in_corrected_order)
    }
    return nx.relabel_nodes(jaxley_graph, mapping)


########################################################################################
################################ BUILD SOLVE GRAPH #####################################
########################################################################################


def _set_comp_and_branch_index(
    comp_graph: nx.DiGraph,
    root: Optional[int] = None,
) -> nx.DiGraph:
    """Given a compartment graph, return a comp_graph with new comp and branch index.

    The returned comp and branch index are the ones used also in the resulting
    jx.Cell, and they define the solve order for `jaxley.stone` solvers.

    Args:
        comp_graph: Compartment graph returned by `build_compartment_graph`.
        root: The root node to traverse the graph for the solve order.

    Returns:
        A directed graph indicating the solve order. The graph does no longer contain
        branchpoints. Nodes contain the following attributes (example):
        ```{'x': 0.0,
        'y': 3.0,
        'z': 0.0,
        'branch_index': 0,
        'comp_index': 0,
        'type': 'comp',
        'xyzr': array([[0., 4., 0., 1.], [0., 3., 0., 1.]]),
        'groups': ['axon'],
        'radius': 1.0,
        'length': 2.0,
        'cell_index': 0}```
    """
    undirected_comp_graph = comp_graph.to_undirected()
    root = root if root else _find_root(undirected_comp_graph)

    # Directed graph to store the traversal
    solve_graph = nx.DiGraph()

    # Copy all global attributes over to the solve_graph.
    for key in comp_graph.graph.keys():
        solve_graph.graph[key] = comp_graph.graph[key]
    solve_graph.add_nodes_from(undirected_comp_graph.nodes(data=True))
    solve_graph.nodes[root]["comp_index"] = 0
    solve_graph.nodes[root]["branch_index"] = 0

    comp_index = 1
    branch_index = 0
    node_inds_in_which_to_flip_xyzr = []

    # Traverse the graph for the solve order.
    # `sort_neighbors=lambda x: sorted(x)` to first handle nodes with lower node index.
    node_mapping = {root: 0}
    for i, j in nx.dfs_edges(
        undirected_comp_graph, root, sort_neighbors=lambda x: sorted(x)
    ):
        solve_graph.add_edge(i, j)
        solve_graph.nodes[j]["branch_index"] = branch_index
        if solve_graph.nodes[j]["type"] == "comp":
            solve_graph.nodes[j]["comp_index"] = comp_index
            node_mapping[j] = comp_index

        if _is_leaf(undirected_comp_graph, j):
            branch_index += 1

        # Increase the branch counter if a branchpoint is encountered.
        elif undirected_comp_graph.nodes[j]["type"] == "branchpoint":
            branch_index += 1

        # Increase the counter for the compartment index only if the node was a
        # compartment (branchpoints are skipped).
        if solve_graph.nodes[j]["type"] == "comp":
            comp_index += 1

        # The `xyzr` attribute of all compartment nodes is ordered in the order in
        # which the SWC file was traversed. If we now traverse a compartment from
        # another direction (because the solve order is not the same as the SWC trace
        # order), then we have to flip the `xyzr` coordinates.
        if not comp_graph.has_edge(i, j) and comp_graph.has_edge(j, i):
            node_inds_in_which_to_flip_xyzr.append(i)
            node_inds_in_which_to_flip_xyzr.append(j)

    unique_nodes_to_flip = list(set(node_inds_in_which_to_flip_xyzr))
    for n in unique_nodes_to_flip:
        # Branchpoint nodes do not have the xyzr property.
        if "xyzr" in solve_graph.nodes[n].keys():
            solve_graph.nodes[n]["xyzr"] = solve_graph.nodes[n]["xyzr"][::-1]

    solve_graph = nx.relabel_nodes(solve_graph, node_mapping)
    solve_graph = _set_branchpoint_indices(solve_graph)
    return solve_graph


def _remove_branch_points_at_tips(comp_graph: nx.DiGraph) -> nx.DiGraph:
    """Delete branch points at tips.

    These only occur if the user was editing the morphology."""
    nodes_to_keep = []
    for node in comp_graph.nodes:
        degree = comp_graph.in_degree(node) + comp_graph.out_degree(node)
        if degree > 1 or comp_graph.nodes[node]["type"] == "comp":
            nodes_to_keep.append(node)
    return comp_graph.subgraph(nodes_to_keep).copy()


def _remove_branch_points(solve_graph: nx.DiGraph) -> nx.DiGraph:
    """Remove branch points and label edges as `inter_branch` or `intra_branch`."""

    # Copy the graph because, otherwise, its input gets modified.
    solve_graph = solve_graph.copy()

    # All connections which either have no `type` label or which are not labelled as
    # synapses are labelled as `intra_branch` for. `inter_branch` connections are
    # handled in the loop below.
    for edge_ind in solve_graph.edges:
        edge = solve_graph.edges[edge_ind]
        if "type" not in edge or edge["type"] != "synapse":
            edge["type"] = "intra_branch"

    # Replace branch points with direct connections between the compartments.
    for node in list(solve_graph.nodes):
        if solve_graph.nodes[node].get("type") == "branchpoint":
            parents = list(solve_graph.predecessors(node))
            children = list(solve_graph.successors(node))
            for v in children:
                for u in parents:
                    solve_graph.add_edge(u, v, type="inter_branch")
            solve_graph.remove_node(node)

    return solve_graph


def _add_meta_data(solve_graph: nx.DiGraph) -> nx.DiGraph:
    """Return a graph with some attributes renamed for Jaxley compatibility.

    The returned graph is what we call the `Jaxley` graph as it is the exact graph
    that can be fully read and written by `Jaxley`.

    Args:
        solve_graph: Directed graph build by `build_solve_graph()`.
        min_radius: If not None, clips the radiuses of all compartments.

    Returns:
        Directed graph which indicates the solve order (and the order of compartments)
        within a Jaxley module. Each node has the following attributes (example, when
        it was read from an SWC file):
        ```{'x': 1.16,
        'y': 5.16,
        'z': 1.0,
        'type': 'comp',
        'xyzr': array([[2., 6., 1., 1.], [1., 5., 1., 1.]]),
        'comp_index': 0,
        'branch_index': 0,
        'cell_index': 0,
        'groups': ['axon'],
        'radius': 1.0,
        'length': 1.18}```
    """
    branch_inds = nx.get_node_attributes(solve_graph, "branch_index")
    branch_edge_df = pd.DataFrame(branch_inds.items(), columns=["node", "branch_index"])
    all_xyzr = []
    # The `.apply(list).sort_index().items()` ensures that the `branch_index` is sorted.
    # This is important because we expect all_xyzr[n] to correspond to the n-th
    # branch.
    for branch_index, nodes_in_branch in (
        branch_edge_df.groupby("branch_index")["node"].apply(list).sort_index().items()
    ):
        nodes_in_branch = sorted(nodes_in_branch)
        xyzr = [solve_graph.nodes[n]["xyzr"] for n in nodes_in_branch]
        xyzr = np.concatenate(xyzr)
        all_xyzr.append(xyzr)

    solve_graph.graph["xyzr"] = all_xyzr

    # It will typically have a `type` if the graph was exported from a module. If it
    # does not, then we make it a cell (which assumes that the read SWC was a cell, but
    # we assert for this anyways (by making sure that it is a `nx.tree`).
    if "type" not in solve_graph.graph:
        solve_graph.graph["type"] = "cell"

    return solve_graph  # This is now a `jaxley_graph`.


########################################################################################
################################ MODULE FROM GRAPH #####################################
########################################################################################


def from_graph(
    comp_graph: nx.DiGraph,
    assign_groups: bool = True,
    solve_root: Optional[int] = None,
    traverse_for_solve_order: bool = True,
):
    """Return a Jaxley module from a compartmentalized networkX graph.

    This method is currently limited to graphs that have the same number of
    compartments in each branch. If this is not the case then the method will raise
    an `AssertionError`.

    Args:
        comp_graph: The compartment graph built with `build_compartment_graph()` or
            with `to_graph()`.
        assign_groups: Whether to assign groups to the nodes.
        solve_root: The root node to traverse the graph for identifying the solve
            order.
        traverse_for_solve_order: Whether to traverse the graph for identifying the
            solve order. Should only be set to `False` if you are confident that the
            `comp_graph` is in a form in which it can be solved (i.e. its branch
            indices, compartment indices, and node names are correct). Typically, this
            is the case only if you exported a module to a `comp_graph` via `to_graph`,
            did not modify the graph, and now re-import it as a module with
            `from_graph`.

    Return:
        A `jx.Module` representing the graph.

    Example usage
    ^^^^^^^^^^^^^

    ::

        from jaxley.io.graph import to_swc_graph, build_compartment_graph, from_graph
        swc_graph = to_swc_graph("path_to_swc.swc")
        comp_graph = build_compartment_graph(swc_graph, ncomp=1)
        cell = from_graph(comp_graph)
    """
    comp_graph = _remove_branch_points_at_tips(comp_graph)

    # If the graph is based on a custom-built `jx.Module` (e.g., parents=[-1, 0, 0, 1]),
    # and we did not modify the exported graph, then we might not want to traverse
    # the graph again because this would change the ordering of the branches.
    if traverse_for_solve_order:
        comp_graph = _set_comp_and_branch_index(comp_graph, root=solve_root)

    solve_graph = _remove_branch_points(comp_graph)
    solve_graph = _add_meta_data(solve_graph)
    module = _build_module(solve_graph, assign_groups=assign_groups)
    return module


def _build_module(
    solve_graph: nx.DiGraph,
    assign_groups: bool = True,
):
    # nodes and edges
    node_df = pd.DataFrame(
        [d for i, d in solve_graph.nodes(data=True)], index=solve_graph.nodes
    ).sort_index()

    node_df = node_df.drop(columns=["xyzr", "type"])
    edge_type = nx.get_edge_attributes(solve_graph, "type")
    synapse_edges = pd.DataFrame(
        [
            {
                "pre_index": i,
                "post_index": j,
                **solve_graph.edges[i, j],
            }
            for (i, j), t in edge_type.items()
            if t == "synapse"
        ]
    )

    # branches
    branch_of_node = lambda i: solve_graph.nodes[i]["branch_index"]
    branch_edges_df = pd.DataFrame(
        [
            (branch_of_node(i), branch_of_node(j))
            for (i, j), t in edge_type.items()
            if t == "inter_branch"
        ],
        columns=["parent_branch_index", "child_branch_index"],
    )

    # drop special attrs from nodes and ignore error if col does not exist
    # x,y,z can be re-computed from xyzr if needed
    optional_attrs = [
        "recordings",
        "externals",
        "external_inds",
        "trainable",
    ]
    node_df = node_df.drop(columns=optional_attrs, errors="ignore")

    # synapses
    synapse_edges = synapse_edges.drop(["l", "type"], axis=1, errors="ignore")
    synapse_edges = synapse_edges.rename({"syn_type": "type"}, axis=1)
    synapse_edges.rename({"edge_index": "global_edge_index"}, axis=1, inplace=True)

    # build module
    acc_parents = []
    parent_branch_inds = branch_edges_df.set_index("child_branch_index").sort_index()[
        "parent_branch_index"
    ]
    assert np.std(node_df.groupby("branch_index").size().to_numpy()) < 1e-8, (
        "`from_graph()` does not support a varying number of compartments in each "
        "branch."
    )
    for branch_inds in node_df.groupby("cell_index")["branch_index"].unique():
        root_branch_idx = branch_inds[0]
        parents = parent_branch_inds.loc[branch_inds[1:]] - root_branch_idx
        acc_parents.append([-1] + parents.tolist())

    # TODO: support inhom ncomps
    module = _build_module_scaffold(
        node_df, solve_graph.graph["type"], acc_parents, solve_graph.graph["xyzr"]
    )

    # set global attributes of module. `xyzr` is passed here again, although it has
    # already been passed to `_build_module_scaffold`. `jx.Cell` requires xyzr at
    # __init__()`, all other modules do not.
    for k, v in solve_graph.graph.items():
        if k not in ["type"]:
            setattr(module, k, v)

    if assign_groups and "groups" in node_df.columns:
        groups = node_df.pop("groups").explode()
        groups = (
            pd.DataFrame(groups)
            .groupby("groups")
            .apply(lambda x: x.index.values, include_groups=False)
            .to_dict()
        )
        for group_name, group_inds in groups.items():
            module.select(nodes=group_inds).add_to_group(group_name)

    node_df.columns = [
        "global_" + col if "local" not in col and "index" in col else col
        for col in node_df.columns
    ]
    # set column-wise. preserves cols not in df.
    module.nodes[node_df.columns] = node_df
    module.edges = synapse_edges if not synapse_edges.empty else module.edges

    # add all the extra attrs
    module.membrane_current_names = [c.current_name for c in module.channels]
    module.synapse_names = [s._name for s in module.synapses]

    return module


def _build_module_scaffold(
    idxs: pd.DataFrame,
    return_type: Optional[str] = None,
    parent_branches: Optional[List[np.ndarray]] = None,
    xyzr: List[np.ndarray] = [],
) -> Union[Network, Cell, Branch, Compartment]:
    """Builds a skeleton module from a DataFrame of indices.

    This is useful for instantiating a module that can be filled with data later.

    Args:
        idxs: DataFrame containing cell_index, branch_index, comp_index, i.e.
            Module.nodes or View.view.
        return_type: Type of module to return. If None, the type is inferred from the
            number of unique values in the indices. I.e. only 1 unique cell_index
                and 1 unique branch_index -> return_type = "jx.Branch".

    Returns:
        A skeleton module with the correct number of compartments, branches, cells, or
        networks."""
    return_types = ["compartment", "branch", "cell", "network"]
    build_cache = {k: [] for k in return_types}

    if return_type is None:  # infer return type from idxs
        return_type = _infer_module_type_from_inds(idxs)

    comp = Compartment()
    build_cache["compartment"] = [comp]

    if return_type in return_types[1:]:
        ncomps = idxs["branch_index"].value_counts().iloc[0]
        branch = Branch([comp for _ in range(ncomps)])
        build_cache["branch"] = [branch]

    if return_type in return_types[2:]:
        branch_counter = 0
        for cell_id, cell_groups in idxs.groupby("cell_index"):
            num_branches = cell_groups["branch_index"].nunique()
            default_parents = np.arange(num_branches) - 1  # ignores morphology
            parents = (
                default_parents if parent_branches is None else parent_branches[cell_id]
            )
            cell = Cell(
                [branch] * num_branches,
                parents,
                xyzr=xyzr[branch_counter : branch_counter + num_branches],
            )
            build_cache["cell"].append(cell)
            branch_counter += num_branches

    if return_type in return_types[3:]:
        build_cache["network"] = [Network(build_cache["cell"])]

    module = build_cache[return_type][0]
    build_cache.clear()
    return module


def _infer_module_type_from_inds(idxs: pd.DataFrame) -> str:
    """Return type (comp, branch, cell, ...) given dataframe of indices."""
    nuniques = idxs[["cell_index", "branch_index", "comp_index"]].nunique()
    nuniques.index = ["cell", "branch", "compartment"]
    nuniques = pd.concat([pd.Series({"network": 1}), nuniques])
    return_type = nuniques.loc[nuniques == 1].index[-1]
    return return_type


########################################################################################
#################################### VISUALIZATION #####################################
########################################################################################


def vis_compartment_graph(
    comp_graph: nx.DiGraph,
    ax=None,
    font_size: float = 7.0,
    node_size: float = 150.0,
    arrowsize: float = 10.0,
    comp_color: str = "r",
    branchpoint_color: str = "orange",
):
    """Visualize a compartment graph.

    Args:
        comp_graph: A compartment graph generated with `build_compartment_graph()` or
            with `to_graph()`.
        ax: Matplotlib axis.
        font_size: The fontsize for the node names.
        node_size: The size of each node.
        arrowsize: The size of the arrow.
        comp_color: The color of the compartments.
        branchpoint_color: The color of the compartments.

    Example usage
    ^^^^^^^^^^^^^

    ::

        from jaxley.io.graph import to_graph, vis_compartment_graph
        cell = jx.read_swc("path_to_swc.swc", ncomp=1)
        comp_graph = to_graph(cell)
        vis_compartment_graph(comp_graph)
    """
    color_map = []
    for n in comp_graph.nodes:
        if comp_graph.nodes[n].get("type") == "comp":
            new_col = comp_color
        elif comp_graph.nodes[n].get("type") == "branchpoint":
            new_col = branchpoint_color
        color_map.append(new_col)

    pos = {k: (v["x"], v["y"]) for k, v in comp_graph.nodes.items()}
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 4))

    nx.draw(
        comp_graph,
        pos=pos,
        with_labels=True,
        font_size=font_size,
        node_size=node_size,
        ax=ax,
        node_color=color_map,
        arrowsize=arrowsize,
    )


########################################################################################
########################### UTILITIES FOR MODIFYING GRAPHS #############################
########################################################################################


def connect_graphs(
    graph1: nx.DiGraph,
    graph2: nx.DiGraph,
    node1: Union[str, int],
    node2: Union[str, int],
) -> nx.DiGraph:
    """Return a new graph that connects two comp_graphs at particular nodes."""
    # For each `group` and `channel` in `graph1`, ensure that it is `False` in `graph2`
    # (if it does not exist).
    graph2 = _assign_false_for_group_and_channel(graph1, graph2)
    graph1 = _assign_false_for_group_and_channel(graph2, graph1)

    # Move graph2 such that it smoothly connects to graph1.
    for i, key in enumerate(["x", "y", "z"]):
        coord1 = _infer_coord(graph1, node1, key)
        coord2 = _infer_coord(graph2, node2, key)
        offset = coord1 - coord2
        for node in graph2.nodes:
            graph2.nodes[node][key] += offset
            if graph2.nodes[node]["type"] == "comp":
                graph2.nodes[node]["xyzr"][:, i] += offset

    # Rename the nodes of graph2.
    offset_comps = max([n for n in graph1.nodes])
    mapping = {}
    for n in sorted(graph2.nodes):
        new_index = n + offset_comps + 1
        mapping[n] = new_index

    graph2 = nx.relabel_nodes(graph2, mapping)
    node2 = mapping[node2]

    # Combine the graph1 and graph2 into one graph.
    combined_graph = nx.compose(graph1, graph2)

    # By default, nx.compose uses the graph-level attributes from graph2. We want that
    # if a graph level attribute is a list, then the graph level attribute of the
    # combined_graph should be a concatenation. This is done below. It ensures that,
    # e.g., channels from both modules are concatenated.
    for key in set(graph1.graph) | set(graph2.graph):
        val1 = graph1.graph.get(key)
        val2 = graph2.graph.get(key)

        if isinstance(val1, list) and isinstance(val2, list):
            combined_graph.graph[key] = val1 + val2
        elif key in graph2.graph:
            combined_graph.graph[key] = val2
        else:
            combined_graph.graph[key] = val1

    # Add edges between graph1 and graph2 to connect them. The code below differentiates
    # three cases: comp->comp, branchpoint->comp (or comp->branchpoint) and
    # branchpoint->branchpoint.
    type1 = combined_graph.nodes[node1]["type"]
    type2 = combined_graph.nodes[node2]["type"]
    offset_branchpoints = max([n for n in combined_graph.nodes])
    if type1 == "comp" and type2 == "comp":
        # If both nodes are compartments, then we insert a new branchpoint.
        #
        # Search for the first node labelled as `type=branchpoint`. Once we have found
        # such a node, we `break`.
        for node in combined_graph.nodes:
            if combined_graph.nodes[node]["type"] == "branchpoint":
                new_attrs = combined_graph.nodes(data=True)[node].copy()
                break
        # Set the xyz coordinates of the new node.
        for key in ["x", "y", "z"]:
            comp1_xyz = combined_graph.nodes[node1][key]
            comp2_xyz = combined_graph.nodes[node2][key]
            new_attrs[key] = 0.5 * (comp1_xyz + comp2_xyz)
        new_node_index = offset_branchpoints + 1
        combined_graph.add_node(new_node_index, **new_attrs)
        combined_graph.add_edge(node1, new_node_index)
        combined_graph.add_edge(new_node_index, node2)
    elif type1 == "comp" or type2 == "comp":
        # If one of the nodes is a compartment and the other one is not, then
        # we just connect.
        combined_graph.add_edge(node1, node2)
    else:
        # Delete branchpoint in second graph. Connect all nodes that it connected to
        # to the first branchpoint.
        for i in combined_graph.predecessors(node2):
            combined_graph.add_edge(i, node1)
        for i in combined_graph.successors(node2):
            combined_graph.add_edge(node1, i)
        combined_graph.remove_node(node2)

    # Add the graph attributes (which are not carried over when doing `compose`)
    group_names = set()
    if "group_names" in graph1.graph.keys():
        group_names = group_names | set(graph1.graph["group_names"])
    if "group_names" in graph2.graph.keys():
        group_names = group_names | set(graph2.graph["group_names"])
    combined_graph.graph["group_names"] = list(group_names)

    # Relabel the compartments. Before the code below, compartments induced by `graph2`
    # have a higher node index than branchpoints of `graph1`. Below, we fix this.
    mapping = {}
    counter = 0
    for n in sorted(combined_graph.nodes):
        if combined_graph.nodes[n]["type"] == "comp":
            mapping[n] = counter
            counter += 1
    for n in sorted(combined_graph.nodes):
        if combined_graph.nodes[n]["type"] == "branchpoint":
            mapping[n] = counter
            counter += 1
    return nx.relabel_nodes(combined_graph, mapping)


def _assign_false_for_group_and_channel(
    graph1: nx.DiGraph, graph2: nx.DiGraph
) -> nx.DiGraph:
    """For any group and channel in graph1.graph, set False in nodes of graph2.

    Args:
        graph1: The graph in which to check for `group_names`.
        graph2: The graph whose `group_names` to update.
    """
    channel_names = [channel.__class__.__name__ for channel in graph1.graph["channels"]]
    if "group_names" in graph1.graph.keys():
        for group in graph1.graph["group_names"] + channel_names:
            for node in graph2.nodes:
                if group not in graph2.nodes[node].keys():
                    graph2.nodes[node][group] = False
    return graph2


def _infer_coord(comp_graph: nx.DiGraph, node: Union[str, int], key: str) -> float:
    """Return the coordinate of a node in a comp_graph.

    Args:
        key: Either of x, y, z.
    """
    i = {"x": 0, "y": 1, "z": 2}[key]
    if comp_graph.nodes[node]["type"] == "branchpoint":
        coordinate = comp_graph.nodes[node][key]
    else:
        if comp_graph.in_degree(node) == 1:
            coordinate = comp_graph.nodes[node]["xyzr"][-1, i]
        else:
            coordinate = comp_graph.nodes[node]["xyzr"][0, i]
    return coordinate
