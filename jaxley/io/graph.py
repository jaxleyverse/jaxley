# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, List, Optional, Tuple, Union
from warnings import warn

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd
from jax import vmap

from jaxley.modules import Branch, Cell, Compartment, Network
from jaxley.utils.cell_utils import v_interp

# helper functions
_is_leaf = lambda G, n: G.out_degree(n) == 0 and G.in_degree(n) == 1
_is_root = lambda G, n: G.in_degree(n) == 0
_is_branching = lambda G, n: G.out_degree(n) > 1
_has_same_id = lambda G, i, j: G.nodes[i]["id"] == G.nodes[j]["id"]
_get_soma_idxs = lambda G: [
    i for i, n in nx.get_node_attributes(G, "id").items() if n == 1
]
_unpack = lambda d, keys: [d[k] for k in keys]
_branch_e2n = lambda b: np.unique(np.concatenate(b)).tolist()
_branch_n2e = lambda b: [e for e in zip(b[:-1], b[1:])]


def _find_root(G):
    roots = [n for n in G.nodes if _is_root(G, n)]
    if len(roots) > 1:
        raise ValueError("Multiple roots found in graph.")
    return roots[0]


def swc_to_graph(fname: str, num_lines: int = None) -> nx.DiGraph:
    """Read a swc file and convert it to a networkx graph.

    The graph is read such that each entry in the swc file becomes a graph node
    with the column attributes (id, x, y, z, r). Then each node is connected to its
    designated parent via an edge. A "type" attribute is added to the graph to identify
    its processing stage for subsequent steps.

    Args:
        fname: Path to the swc file.
        num_lines: Number of lines to read from the file. If None, all lines are read.

    Returns:
        A networkx graph of the traced morphology in the swc file.
    """
    i_id_xyzr_p = np.loadtxt(fname)[:num_lines]

    graph = nx.DiGraph()
    graph.add_nodes_from(
        (
            (int(i), {"id": int(id), "x": x, "y": y, "z": z, "r": r})
            for i, id, x, y, z, r, p in i_id_xyzr_p
        )
    )
    graph.add_edges_from([(p, i) for p, i in i_id_xyzr_p[:, [-1, 0]] if p != -1])
    graph = nx.relabel_nodes(graph, {i: i - 1 for i in graph.nodes})
    return graph


def _get_nodes_and_parents(graph: nx.DiGraph) -> np.ndarray:
    """List (node, parent) pairs for a graph."""
    edges = []
    for node in graph.nodes():
        parents = list(graph.predecessors(node))
        edges += [(node, parent) for parent in parents] if parents else [(node, -1)]
    return np.array(edges)


def find_swc_trace_errors(graph: nx.DiGraph, ignore: Optional[List] = []) -> np.ndarray:
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
    # ammend branch 2
    9 4 5.0 3.0 0.0 1.0 5

    Args:
        graph: graph tracing of swc file (from `swc_to_graph`).
        ignore: Nodes to ignore.

    Returns:
        An array of node indices where tracing is discontinous.
    """
    edges = _get_nodes_and_parents(graph)[:, [-1, 0]]
    s, t = edges[0].T
    branch_ends = []

    # find first and last node of each connected segment, i.e.
    # [(1,2),(2,3),(3,4),(5,6),...] -> [(1,4),(5,...),...]
    for i, j in edges[1:]:
        if i != t:
            branch_ends += [(s, t)]
            s = i
        t = j
    branch_ends += [(s, t)]
    branch_ends = np.array(branch_ends)
    # nodes that end one and start another segment indicate a breakage during tracing
    break_idxs = branch_ends[:, 1][np.isin(branch_ends[:, 1], branch_ends[:, 0])]
    break_idxs = np.setdiff1d(break_idxs, ignore)
    return break_idxs


def simulate_swc_trace_errors(
    graph: nx.DiGraph, branches: List[np.ndarray], ignore: Optional[List] = []
) -> List[np.ndarray]:
    """Simulate swc trace errors in the branches.

    Both NEURON and Jaxley's hand coded swc reader introduce breaks in the trace
    if the same neurite was traced in disconnected pieces. Since `swc_to_graph` is
    agnostic to the order of the tracing, it does not suffer from this issue. Hence,
    to artifically force this behaviour (to compare to the other parsers), this
    function can be used to simulates these errors. See `find_swc_trace_errors` for
    how to identify these points in the graph.

    Args:
        graph: A networkx graph of a traced morphology.
        branches: List of branches represented as arrays of edges.
        ignore: Nodes to ignore when splitting branches.

    Returns:
        A list of branches with simulated trace errors.
    """
    node_idxs = find_swc_trace_errors(graph, ignore=ignore)

    for node_idx in node_idxs:
        branch_idx = next(i for i, p in enumerate(branches) if node_idx in p)
        b4, branch, after = (
            branches[:branch_idx],
            branches[branch_idx],
            branches[branch_idx + 1 :],
        )
        break_idx = np.where(branch == node_idx)[0][1]
        # insert artifical break into branch
        branches = b4 + [branch[:break_idx], branch[break_idx:]] + after
    return branches


def trace_branches(
    graph: nx.DiGraph, max_len=None, ignore_swc_trace_errors=True
) -> List[np.ndarray]:
    """Get all linearly connected paths in a graph aka. branches.

    The graph is traversed depth-first starting from the source node, which is the only
    node with in_degree 0 (raises in case multiple are found). Note: Traversal order can
    be changed, by reversing the edge directions, i.e. to start traversal from a leaf node
    set: [source -> ... -> leaf] to [leaf -> ... -> source].

    Args:
        graph: A networkx graph.
        ignore_swc_trace_errors: Whether to ignore discontinuities in the swc tracing
            order. If False, this will result in split branches at these points.
    Returns:
        A list of linear paths in the graph. Each path is represented as an array of
        edges."""

    # handles special case of a single soma node
    if len(soma_idxs := _get_soma_idxs(graph)) == 1:
        soma = soma_idxs[0]
        # edges connecting nodes to soma are considered part of the soma -> l = 0.
        for i, j in (*graph.in_edges(soma), *graph.out_edges(soma)):
            graph.edges[i, j]["l"] = 0

        # Setting l = 2*r ensures A_cylinder = 2*pi*r*l = 4*pi*r^2 = A_sphere
        graph.add_node(-1, **graph.nodes[0])
        graph.add_edge(-1, soma, l=2 * graph.nodes[soma]["r"])
        graph = nx.relabel_nodes(graph, {i: i + 1 for i in graph.nodes})

    # Ensure root segment is linear. Needed to create root branch.
    if graph.out_degree(0) > 1:
        # The root segment should be of type `custom` (=5).
        parent = graph.nodes[0]
        parent["id"] = 5
        graph.add_node(-1, **parent)
        graph.add_edge(-1, 0, l=0.1)
        graph = nx.relabel_nodes(graph, {i: i + 1 for i in graph.nodes})

    branches, current_branch = [], []

    root = _find_root(graph)
    for i, j in nx.dfs_edges(graph, root):
        current_branch += [(i, j)]
        if _is_leaf(graph, j) or _is_branching(graph, j):
            branches.append(current_branch)
            current_branch = []
        elif not _has_same_id(graph, i, j):  # start new branch if ids differ
            branches.append(current_branch[:-1])
            current_branch = [current_branch[-1]]

    branch_edges = [np.array(p) for p in branches if len(p) > 0]

    if max_len:
        edge_lens = nx.get_edge_attributes(graph, "l")
        branch_edges = split_branches(branch_edges, edge_lens, max_len)

    if not ignore_swc_trace_errors:
        # ignore added index by default; only relevant in case it was added
        branch_edges = simulate_swc_trace_errors(graph, branch_edges, ignore=[root])

    for br_idx, br_edges in enumerate(branch_edges):
        graph.add_edges_from(br_edges, branch_index=br_idx)
    return graph


def _add_edge_lens(graph: nx.DiGraph, min_len: float = 1e-5) -> nx.DiGraph:
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


def add_missing_graph_attrs(graph: nx.DiGraph) -> nx.DiGraph:
    """Add missing attributes to the graph nodes and edges.

    The following attributes are added to the graph:
    - id: int (default: 0)
    - x, y, z: float (default: NaN)
    - r: float (default: 1)
    - l: float (default: 1)

    Args:
        graph: A networkx graph.

    Returns:
        The graph with the added attributes."""
    available_keys = graph.nodes[0].keys()
    defaults = {
        "id": 0,
        "x": float("nan"),
        "y": float("nan"),
        "z": float("nan"),
        "r": 1,
    }
    # add defaults if not present
    for key in set(defaults.keys()).difference(available_keys):
        nx.set_node_attributes(graph, defaults[key], key)

    graph = _add_edge_lens(graph)
    edge_lens = nx.get_edge_attributes(graph, "l")
    if np.isnan(list(edge_lens.values())[0]):
        nx.set_edge_attributes(graph, 1, "l")

    return graph


def split_branches(
    branches: List[np.ndarray], edge_lens: Dict, max_len: int = 1000
) -> List[np.ndarray]:
    """Split branches into approximately equally long sections <= max_len.

    Args:
        branches: List of branches represented as arrays of edges.
        edge_lens: Dict for length of each edge in the graph.
        max_len: Maximum length of a branch section. If a branch exceeds this length,
            it is split into equal parts.

    Returns:
        A list of branches, where each branch is split into sections of length <= max_len.
    """
    # TODO: split branches into exactly equally long sections
    edge_lens.update({(j, i): l for (i, j), l in edge_lens.items()})
    new_branches = []
    for branch in branches:
        cum_branch_len = np.cumsum([edge_lens[i, j] for i, j in branch])
        k = cum_branch_len // max_len
        split_branch = [branch[np.where(np.array(k) == kk)[0]] for kk in np.unique(k)]
        new_branches += split_branch
    return new_branches


def insert_compartments(graph: nx.DiGraph, ncomp_per_branch: int) -> nx.DiGraph:
    """Insert compartment nodes into the graph.

    Inserts new nodes in every branch (edges with "branch_index" attribute) at equidistant
    points along it. Node attributes, like radius are linearly interpolated along its
    length.

    Example: 4 compartments | edges = - | nodes = o | comp_nodes = x
    o-----------o----------o---o---o---o--------o
    o-------x---o----x-----o--xo---o---ox-------o

    Args:
        graph: Mmorphology where edges are already labelled with "branch_index"
        ncomp_per_branch: How many compartments per branch to insert

    Returns:
        Graph with additional nodes that are labelled with "comp_index"
    """
    comp_offset = 0

    branch_inds = nx.get_edge_attributes(graph, "branch_index")
    branch_edge_df = pd.DataFrame(branch_inds.items(), columns=["edge", "branch_index"])
    for branch_index, branch_edges in branch_edge_df.groupby("branch_index")["edge"]:
        path_lens = np.cumsum([0] + [graph.edges[u, v]["l"] for u, v in branch_edges])
        branch_nodes = _branch_e2n(branch_edges.to_numpy())
        branch_data = pd.DataFrame([graph.nodes[i] for i in branch_nodes])
        branch_data["node_index"] = branch_nodes
        branch_data["l"] = path_lens

        # fix id and r bleed over from neighboring neurites of a different type
        if branch_data.loc[0, "id"] != branch_data.loc[1, "id"]:
            branch_data.loc[0, ["r", "id"]] = branch_data.loc[1, ["r", "id"]]

        branch_len = branch_data["l"].max()
        comp_len = branch_len / ncomp_per_branch
        locs = np.linspace(comp_len / 2, branch_len - comp_len / 2, ncomp_per_branch)

        new_branch_nodes = v_interp(locs, branch_data["l"].values, branch_data.values)
        new_branch_nodes = pd.DataFrame(
            np.array(new_branch_nodes.T), columns=branch_data.columns
        )
        new_branch_nodes["id"] = new_branch_nodes["id"].astype(int)
        new_branch_nodes["comp_length"] = comp_len
        new_branch_nodes["branch_index"] = branch_index
        new_branch_nodes["comp_index"] = comp_offset + np.arange(ncomp_per_branch)
        new_branch_nodes["node_index"] = (
            max(graph.nodes) + 1 + np.arange(ncomp_per_branch)
        )
        comp_offset += ncomp_per_branch

        # splice comps into morphology
        branch_data = pd.concat([branch_data, new_branch_nodes]).sort_values(
            "l", ignore_index=True
        )
        new_branch_nodes = new_branch_nodes.set_index("node_index")
        graph.add_nodes_from(new_branch_nodes.to_dict(orient="index").items())

        graph.remove_edges_from(branch_edges.to_numpy())
        new_branch_edges = _branch_n2e(branch_data["node_index"])
        graph.add_edges_from(new_branch_edges, branch_index=branch_index)

    # add missing edge lengths
    graph = _add_edge_lens(graph)

    # re-enumerate in dfs from root
    root = _find_root(graph)
    mapping = {old: new for new, old in enumerate(nx.dfs_preorder_nodes(graph, root))}
    graph = nx.relabel_nodes(graph, mapping)
    return graph


def _get_comp_edges_dfs(
    graph: nx.DiGraph, node: int, parent_comp: int = None, visited: set = None
) -> List[Tuple[int]]:
    """List edges between compartment nodes, ignoring non-compartment nodes.

    Traverses a graph depth first and only records nodes and their successors if they
    have a "compartment_index".

    Args:
        graph: Morphology with inserted compartment nodes.
        node: node_index from which to start depth first traversal
        parent_comp: node_index of parent compartment
        visited: Keeps track of visited nodes during traversal

    Returns:
        List of edges (parent_node_index, child_node_index) that directly connect
        compartments, while skipping indermediate hops via nodes w.o. "comp_index" attr.
    """
    if visited is None:
        visited = set()
    edges = []
    visited.add(node)

    current_comp = node if "comp_index" in graph.nodes[node] else None

    # If we have a parent with comp_index and current node has comp_index,
    # add the edge between them
    if parent_comp is not None and current_comp is not None:
        edges.append((parent_comp, current_comp))

    # Process all children depth first
    for child in graph.successors(node):
        if child not in visited:
            # Pass current_comp as parent if it exists, otherwise pass through the parent_comp
            prev_node = current_comp if current_comp is not None else parent_comp
            child_edges = _get_comp_edges_dfs(graph, child, prev_node, visited)
            edges.extend(child_edges)

    return edges


def extract_comp_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """Get subgraph that only includes compartment nodes and their direct edges.

    Example: 4 compartments | edges = - | nodes = o | comp_nodes = x
    o-------x---o----x-----o--xo---o---ox-------o
            x--------x--------x---------x

    Args:
        graph: Morphology with compartment nodes

    Returns:
        Morphology with branches and compartments.
    """
    # create subgraph w.o. edges
    comp_nodes = list(nx.get_node_attributes(graph, "comp_index"))
    comp_graph = nx.subgraph(graph, comp_nodes).copy()

    # remove all edges
    comp_graph.remove_edges_from(list(comp_graph.edges))

    # find all edges between compartments and connect comp_graph
    root = _find_root(graph)
    comp_edges = _get_comp_edges_dfs(graph, root)
    comp_graph.add_edges_from(comp_edges)
    comp_graph = _add_edge_lens(comp_graph)

    for i, j in comp_graph.edges:
        branch_idx_i = graph.nodes[i]["branch_index"]
        branch_idx_j = graph.nodes[j]["branch_index"]
        if branch_idx_i != branch_idx_j:
            comp_graph.edges[i, j]["type"] = "inter_branch"
        else:
            comp_graph.edges[i, j]["type"] = "intra_branch"

    # fix 0 length edges (usually due adding root branch)
    min_len = 1e-3
    for i, j in comp_graph.edges:
        if comp_graph.edges[i, j]["l"] < min_len:
            comp_graph.edges[i, j]["l"] = min_len

    # relabel nodes using comp_index
    mapping = {n: attrs["comp_index"] for n, attrs in comp_graph.nodes(data=True)}

    comp_graph = nx.relabel_nodes(comp_graph, mapping, copy=True)
    return comp_graph


def make_jaxley_compatible(
    graph: nx.DiGraph,
    ncomp: int = 4,
    max_branch_len: float = None,
    min_radius: Optional[float] = None,
    ignore_swc_trace_errors: bool = True,
) -> nx.DiGraph:
    """Make a swc traced graph compatible with jaxley.

    In order to simulate a morphology jaxley imposes that it has to consist of
    branches, each of which consists of compartments with equal length. This function
    first traces the graph to compute the length of each edge and to identify its branch
    structure. It then splits these branches into equally long compartments. The
    compartment centers and radii are obtained by linearly interpolating the traced xyz
    coordinates and radii alongeach branch.

    This means initially, each node represents a traced point (an entry in the swc
    file) and each edge represents the direction along which the neurites were traced.
    Edges along every non-branching linear section of the neurite (smaller than the
    max_branch_len) are then assigned a branch index. Finally, each branch is split
    into pieces of equal length. This results in a new graph, where each node
    represents a compartment (at its center) and each edge represents a connection
    between these compartments. The graph is then ready to be imported into jaxley.

    Attributes that are added to the returned graph:
    - xyzr: list of the originally traced points.
    - type: the type of module is inferred depending on the number of compartments,
        i.e. if there is only one compartment, it will be "compartment". This let's
        `from_graph` know what type of module to return.
    - groups: the group of each compartment is inferred from the id attribute of the
        nodes.
    - cell_index: enumerates cells globally.
    - branch_index: enumerates branches globally.
    - comp_index: enumerates compartments globally.
    - radius: the radius of each compartment.
    - length: the length of each compartment.
    - x, y, z: the coordinates of each compartment.

    Args:
        graph: A networkx graph of a traced morphology.
        ncomp: The number of segments per compartment.
        max_branch_len: Maximal length of one branch. If a branch exceeds this length,
            it is split into equal parts.
        min_radius: If the radius of a reconstruction is below this value it is clipped.
        ignore_swc_trace_errors: Whether to ignore discontinuities in the swc tracing
            order. If False, this will result in split branches at these points.

    Returns:
        A networkx graph of the traced morphology that is compatible with jaxley
        and can be imported using `from_graph`. This is also the same format that
        is returned by `to_graph` when exporting a module.
    """
    # TODO: assert DAG and connected

    # pre-processing
    graph = add_missing_graph_attrs(graph)

    graph = trace_branches(graph, max_branch_len, ignore_swc_trace_errors)

    graph = insert_compartments(graph, ncomp)

    comp_graph = extract_comp_graph(graph)

    # post-processing

    # rename attrs
    # Description of SWC file format:
    # http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
    group_ids = {0: "undefined", 1: "soma", 2: "axon", 3: "basal", 4: "apical"}
    min_radius = min_radius if min_radius else 0.0
    clip_radius = lambda r: max(r, min_radius) if min_radius else r
    for n in comp_graph.nodes:
        # `dictionary.get(key, "custom")` returns `custom` by default. This is useful
        # because some SWC files might have really strange type identifier.
        type_id = comp_graph.nodes[n].pop("id")
        comp_graph.nodes[n]["groups"] = [group_ids.get(type_id, f"custom{type_id}")]
        comp_graph.nodes[n]["radius"] = clip_radius(comp_graph.nodes[n].pop("r"))
        comp_graph.nodes[n]["length"] = comp_graph.nodes[n].pop("comp_length")
        comp_graph.nodes[n].pop("l")

    nx.set_node_attributes(comp_graph, 0, "cell_index")

    # compute additional attributes
    edge_df = pd.DataFrame(
        [
            d["branch_index"] for i, j, d in graph.edges(data=True)
        ],  # TODO: filter out added comp indices to get original morphology
        index=graph.edges,
        columns=["branch_index"],
    )

    # xyzr
    edges_in_branches = edge_df.groupby("branch_index")
    nodes_in_branches = edges_in_branches.apply(
        lambda x: [
            n for n in _branch_e2n(x.index.values) if "comp_index" not in graph.nodes[n]
        ]
    )
    stack_branch_xyzr = lambda x: np.stack([_unpack(graph.nodes[n], "xyzr") for n in x])
    xyzr = nodes_in_branches.apply(stack_branch_xyzr).to_list()
    same_rows = lambda x: np.all(np.nan_to_num(x[0]) == np.nan_to_num(x))
    xyzr = [xyzr_i[[0, -1]] if same_rows(xyzr_i) else xyzr_i for xyzr_i in xyzr]

    comp_graph.graph["xyzr"] = xyzr
    comp_graph.graph["type"] = "cell"

    return comp_graph


def _infer_module_type_from_inds(idxs: pd.DataFrame) -> str:
    nuniques = idxs[["cell_index", "branch_index", "comp_index"]].nunique()
    nuniques.index = ["cell", "branch", "compartment"]
    nuniques = pd.concat([pd.Series({"network": 1}), nuniques])
    return_type = nuniques.loc[nuniques == 1].index[-1]
    return return_type


def build_module_scaffold(
    idxs: pd.DataFrame,
    return_type: Optional[str] = None,
    parent_branches: Optional[List[np.ndarray]] = None,
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
        for cell_id, cell_groups in idxs.groupby("cell_index"):
            num_branches = cell_groups["branch_index"].nunique()
            default_parents = np.arange(num_branches) - 1  # ignores morphology
            parents = (
                default_parents if parent_branches is None else parent_branches[cell_id]
            )
            cell = Cell([branch] * num_branches, parents)
            build_cache["cell"].append(cell)

    if return_type in return_types[3:]:
        build_cache["network"] = [Network(build_cache["cell"])]

    module = build_cache[return_type][0]
    build_cache.clear()
    return module


def from_graph(
    graph: nx.DiGraph,
    ncomp: int = 4,
    max_branch_len: float = 2000.0,
    min_radius: Optional[float] = None,
    assign_groups: bool = True,
    ignore_swc_trace_errors: bool = True,
) -> Union[Network, Cell, Branch, Compartment]:
    """Build a module from a networkx graph.

    All information stored in the nodes of the graph is imported into the Module.nodes
    attribute. Edges of type "inter_branch" connect two different branches, while
    edges of type "intra_branch" connect compartments within the same branch. All other
    edges are considered synapse edges. These are added to the Module.branch_edges and
    Module.edges attributes, respectively. Additionally, the graph can contain
    global attributes, which are added as attrs, i.e. to the module instance and
    optionally can store recordings, externals, and trainables. See `to_graph` for
    how they are formatted.

    All modules that are exported to a graph from jaxley using the `to_graph` method
    can be imported back using this method. If the graph was not exported with jaxley,
    the only requirement to construct a module from a graph is that the graph contains
    edges, i.e.

    ```python
    edges = [(0,1), (1,2), (1,3)]
    graph = nx.DiGraph(edges)
    module = from_graph(graph)
    ```

    In this case, all requrired parameters are set to their default values
    and the branch structure is computed naively. However, note that in this case
    the graph has to be connected and cannot contain loops. Currently networks can
    only be imported if they have been previously exported with jaxley!

    Returns nx.Network, nx.Cell, nx.Branch, or nx.Compartment depending on
    num_cell_idxs > 1, num_branch_idxs > 1 etc.

    Possible attributes that can be read off of the graph include:
    - graph
        - ncomp: int
        - total_nbranches: int
        - cumsum_nbranches: np.ndarray
        - channels
        - synapses
        - xyzr: list[np.ndarray]
        - recordings: list[str]
        - externals: list[float]
        - trainable: dict[str, float]
    - nodes:
        - id: int (used to define groups, according to NEURON's SWC convention)
        - groups: list[str] (can also be defined directly)
        - cell_index: int
        - branch_index: int
        - comp_index: int
        - radius: float
        - length: float
        - x: float
        - y: float
        - z: float
    - edges:
        - type: str ("synapse" or "inter_branch" / "intra_branch" or None)
        - parameters: list[dict] (stores synapse parameters)

    Args:
        graph: A networkx graph representing a module.
        ncomp: The default number of segments per compartment.
            Will only be selected if the graph has not been compartmentalized yet.
        max_branch_len: Maximal length of one branch. If a branch exceeds this length,
            it is split into equal parts such that each subbranch is below
            `max_branch_len`. Will only be used if no branch structure is has been
            assigned yet.
        min_radius: If the radius of a reconstruction is below this value it is clipped.
        assign_groups: Wether to assign groups to nodes based on the the id or groups
            attribute.
        ignore_swc_trace_errors: Whether to ignore discontinuities in the swc tracing
            order. If False, this will result in split branches at these points.

    Returns:
        A module instance that is populated with the node and egde attributes of
        the nx.DiGraph."""

    ########################################
    ### Make the graph jaxley compatible ###
    ########################################

    if "type" not in graph.graph:
        try:
            graph = make_jaxley_compatible(
                graph,
                ncomp=ncomp,
                max_branch_len=max_branch_len,
                min_radius=min_radius,
                ignore_swc_trace_errors=ignore_swc_trace_errors,
            )
        except:
            raise Exception("Graph appears to be incompatible with jaxley.")

    #################################
    ### Import graph as jx.Module ###
    #################################

    # nodes and edges
    node_df = pd.DataFrame(
        [d for i, d in graph.nodes(data=True)], index=graph.nodes
    ).sort_index()
    edge_type = nx.get_edge_attributes(graph, "type")
    synapse_edges = pd.DataFrame(
        [
            {
                "pre_global_comp_index": i,
                "post_global_comp_index": j,
                **graph.edges[i, j],
            }
            for (i, j), t in edge_type.items()
            if t == "synapse"
        ]
    )

    # branches
    branch_of_node = lambda i: graph.nodes[i]["branch_index"]
    branch_edges = pd.DataFrame(
        [
            (branch_of_node(i), branch_of_node(j))
            for (i, j), t in edge_type.items()
            if t == "inter_branch"
        ],
        columns=["parent_branch_index", "child_branch_index"],
    )

    # drop special attrs from nodes and ignore error if col does not exist
    # x,y,z can be re-computed from xyzr if needed
    optional_attrs = ["recordings", "externals", "trainable", "x", "y", "z"]
    node_df = node_df.drop(columns=optional_attrs, errors="ignore")

    # synapses
    synapse_edges = synapse_edges.drop(["l", "type"], axis=1, errors="ignore")
    synapse_edges = synapse_edges.rename({"syn_type": "type"}, axis=1)
    synapse_edges.rename({"edge_index": "global_edge_index"}, axis=1, inplace=True)

    # build module
    acc_parents = []
    parent_branch_inds = branch_edges.set_index("child_branch_index").sort_index()[
        "parent_branch_index"
    ]
    for branch_inds in node_df.groupby("cell_index")["branch_index"].unique():
        root_branch_idx = branch_inds[0]
        parents = parent_branch_inds.loc[branch_inds[1:]] - root_branch_idx
        acc_parents.append([-1] + parents.tolist())

    # TODO: support inhom ncomps
    module = build_module_scaffold(node_df, graph.graph["type"], acc_parents)

    # set global attributes of module
    for k, v in graph.graph.items():
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
    module.nodes[node_df.columns] = (
        node_df  # set column-wise. preserves cols not in df.
    )

    module.edges = synapse_edges if not synapse_edges.empty else module.edges

    # add all the extra attrs
    module.membrane_current_names = [c.current_name for c in module.channels]
    module.synapse_names = [s._name for s in module.synapses]

    return module


def to_graph(
    module: "jx.Module", synapses: bool = False, channels: bool = False
) -> nx.DiGraph:
    """Export the module as a networkx graph.

    Constructs a nx.DiGraph from the module. Each compartment in the module
    is represented by a node in the graph. The edges between the nodes represent
    the connections between the compartments. These edges can either be connections
    between compartments within the same branch, between different branches or
    even between different cells. In the latter case the synapse parameters
    are stored as edge attributes. Only allows one synapse per edge however!
    Additionally, global attributes of the module, for example `ncomp`, are stored as
    graph attributes.

    Exported graphs can be imported again to `jaxley` using the `from_graph` method.

    Args:
        module: A jaxley module or view instance.
        synapses: Whether to export synapses to the graph.
        channels: Whether to export ion channels to the graph.

    Returns:
        A networkx graph of the module.
    """
    module_graph = nx.DiGraph()
    module.compute_compartment_centers()  # make xyz coords attr of nodes

    # add global attrs
    module_graph.graph["type"] = module.__class__.__name__.lower()
    for attr in [
        "ncomp",
        "xyzr",
        "externals",
        "recordings",
        "trainable_params",
        "indices_set_by_trainables",
    ]:
        module_graph.graph[attr] = getattr(module, attr)

    # add nodes
    nodes = module.nodes.copy()
    nodes = nodes.drop([col for col in nodes.columns if "local" in col], axis=1)
    nodes.columns = [col.replace("global_", "") for col in nodes.columns]

    if channels:
        module_graph.graph["channels"] = module.channels
        module_graph.graph["membrane_current_names"] = [
            c.current_name for c in module.channels
        ]
    else:
        for c in module.channels:
            nodes = nodes.drop(c.name, axis=1)
            nodes = nodes.drop(list(c.channel_params), axis=1)
            nodes = nodes.drop(list(c.channel_states), axis=1)

    for col in nodes.columns:  # col wise adding preserves dtypes
        module_graph.add_nodes_from(nodes[[col]].to_dict(orient="index").items())

    module_graph.graph["group_names"] = module.group_names

    inter_branch_edges = module.branch_edges.copy()
    intra_branch_edges = []
    for i, branch_data in nodes.groupby("branch_index"):
        inds = branch_data.index.values
        intra_branch_edges += _branch_n2e(inds)

        parents = module.branch_edges["parent_branch_index"]
        children = module.branch_edges["child_branch_index"]
        inter_branch_edges.loc[parents == i, "parent_branch_index"] = inds[-1]
        inter_branch_edges.loc[children == i, "child_branch_index"] = inds[0]

    inter_branch_edges = inter_branch_edges.to_numpy()
    module_graph.add_edges_from(inter_branch_edges, type="inter_branch")
    module_graph.add_edges_from(intra_branch_edges, type="intra_branch")

    if synapses:
        syn_edges = module.edges.copy()
        multiple_syn_per_edge = syn_edges[
            ["pre_global_comp_index", "post_global_comp_index"]
        ].duplicated(keep=False)
        dupl_inds = multiple_syn_per_edge.index[multiple_syn_per_edge].values
        if multiple_syn_per_edge.any():
            warn(
                f"CAUTION: Synapses {dupl_inds} are connecting the same compartments. Exporting synapses to the graph only works if the same two compartments are connected by at most one synapse."
            )
        module_graph.graph["synapses"] = module.synapses
        module_graph.graph["synapse_param_names"] = module.synapse_param_names
        module_graph.graph["synapse_state_names"] = module.synapse_state_names
        module_graph.graph["synapse_names"] = module.synapse_names
        module_graph.graph["synapse_current_names"] = module.synapse_current_names

        syn_edges.columns = [col.replace("global_", "") for col in syn_edges.columns]
        syn_edges["syn_type"] = syn_edges["type"]
        syn_edges["type"] = "synapse"
        syn_edges = syn_edges.set_index(["pre_comp_index", "post_comp_index"])
        if not syn_edges.empty:
            for (i, j), edge_data in syn_edges.iterrows():
                module_graph.add_edge(i, j, **edge_data.to_dict())

    module_graph.graph["type"] = module.__class__.__name__.lower()

    return module_graph
