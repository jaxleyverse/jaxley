# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd
from jax import vmap

import jaxley as jx
from jaxley.utils.cell_utils import v_interp

# helper functions
is_leaf = lambda G, n: G.out_degree(n) == 0 and G.in_degree(n) == 1
is_root = lambda G, n: G.to_undirected().degree(n) == 1
is_branching = lambda G, n: G.out_degree(n) > 1
has_same_id = lambda G, i, j: G.nodes[i]["id"] == G.nodes[j]["id"]
get_soma_idxs = lambda G: [
    i for i, n in nx.get_node_attributes(G, "id").items() if n == 1
]

unpack = lambda d, keys: [d[k] for k in keys]
branch_e2n = lambda b: np.unique(np.concatenate(b)).tolist()
branch_n2e = lambda b: [e for e in zip(b[:-1], b[1:])]
v_interp = vmap(jnp.interp, in_axes=(None, None, 1))


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


def get_nodes_and_parents(graph: nx.DiGraph) -> np.ndarray:
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
    edges = get_nodes_and_parents(graph)[:, [-1, 0]]
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
    graph: nx.DiGraph, max_len=1000, ignore_swc_trace_errors=True
) -> List[np.ndarray]:
    """Get all linearly connected paths in a graph aka. branches.

    The graph is traversed depth-first starting from the source node.

    Args:
        graph: A networkx graph.
        source_node: node at which to start graph traversal. If "leaf", the traversal
            starts at the first identified leaf node.
        ignore_swc_trace_errors: Whether to ignore discontinuities in the swc tracing
            order. If False, this will result in split branches at these points.
    Returns:
        A list of linear paths in the graph. Each path is represented as an array of
        edges."""
    branches, current_branch = [], []

    for i, j in nx.dfs_edges(graph, 0):
        current_branch += [(i, j)]
        if is_leaf(graph, j) or is_branching(graph, j):
            branches.append(current_branch)
            current_branch = []
        elif not has_same_id(graph, i, j):  # start new branch if ids differ
            branches.append(current_branch[:-1])
            current_branch = [current_branch[-1]]

    branch_edges = [np.array(p) for p in branches if len(p) > 0]

    edge_lens = nx.get_edge_attributes(graph, "l")
    branch_edges = split_branches(branch_edges, edge_lens, max_len)

    if not ignore_swc_trace_errors:
        # ignore added index by default; only relevant in case it was added
        branch_edges = simulate_swc_trace_errors(graph, branch_edges, ignore=[0])

    for br_idx, br_edges in enumerate(branch_edges):
        graph.add_edges_from(br_edges, branch_index=br_idx)
    return graph


def add_edge_lens(graph: nx.DiGraph) -> nx.DiGraph:
    """Add edge lengths to graph.edges based on the xyz coordinates of graph.nodes."""
    xyz = lambda i: np.array(unpack(graph.nodes[i], "xyz"))
    for i, j in graph.edges:
        d_ij = (
            np.sqrt(((xyz(i) - xyz(j)) ** 2).sum())
            if i != j
            else 2 * graph.nodes[i]["r"]
        )
        graph.edges[i, j]["l"] = d_ij
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

    graph = add_edge_lens(graph)
    edge_lens = nx.get_edge_attributes(graph, "l")
    if np.isnan(list(edge_lens.values())[0]):
        nx.set_edge_attributes(graph, 1, "l")

    return graph


def split_branches(
    branches: List[np.ndarray], edge_lens: Dict, max_len: int = 100
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
    comp_offset = 0

    branch_inds = nx.get_edge_attributes(graph, "branch_index")
    branch_edge_df = pd.DataFrame(branch_inds.items(), columns=["edge", "branch_index"])
    for branch_index, branch_edges in branch_edge_df.groupby("branch_index")["edge"]:
        path_lens = np.cumsum([0] + [graph.edges[u, v]["l"] for u, v in branch_edges])
        branch_nodes = branch_e2n(branch_edges.to_numpy())
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

        # splice comps into morphology graph func could be reused in to_graph
        branch_data = pd.concat([branch_data, new_branch_nodes]).sort_values(
            "l", ignore_index=True
        )
        new_branch_nodes = new_branch_nodes.set_index("node_index")
        graph.add_nodes_from(new_branch_nodes.to_dict(orient="index").items())

        graph.remove_edges_from(branch_edges.to_numpy())
        new_branch_edges = branch_n2e(branch_data["node_index"])
        graph.add_edges_from(new_branch_edges, branch_index=branch_index)

    # add missing edge lengths
    graph = add_edge_lens(graph)

    # re-enumerate in dfs from 0
    mapping = {old: new for new, old in enumerate(nx.dfs_preorder_nodes(graph, 0))}
    graph = nx.relabel_nodes(graph, mapping)
    return graph


def get_comp_edges_dfs(graph, node, parent_comp=None, visited=None):
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
            child_edges = get_comp_edges_dfs(graph, child, prev_node, visited)
            edges.extend(child_edges)

    return edges


def extract_comp_graph(graph):
    # create subgraph w.o. edges
    comp_nodes = list(nx.get_node_attributes(graph, "comp_index"))
    comp_graph = nx.subgraph(graph, comp_nodes).copy()

    # remove all edges
    comp_graph.remove_edges_from(list(comp_graph.edges))

    # find all edges between compartments and connect comp_graph
    comp_edges = get_comp_edges_dfs(graph, 0)
    comp_graph.add_edges_from(comp_edges)
    comp_graph = add_edge_lens(comp_graph)

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
    max_branch_len: float = 2000.0,
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
        nseg: The number of segments per compartment.
        max_branch_len: Maximal length of one branch. If a branch exceeds this length,
            it is split into equal parts.
        ignore_swc_trace_errors: Whether to ignore discontinuities in the swc tracing
            order. If False, this will result in split branches at these points.

    Returns:
        A networkx graph of the traced morphology that is compatible with jaxley
        and can be imported using `from_graph`. This is also the same format that
        is returned by `to_graph` when exporting a module.
    """
    # TODO:
    # assert root node is 0
    # assert DAG and connected

    # pre-processing
    graph = add_missing_graph_attrs(graph)

    # handles special case of a single soma node
    if len(soma_idxs := get_soma_idxs(graph)) == 1:
        soma = soma_idxs[0]
        # Setting l = 2*r ensures A_cylinder = 2*pi*r*l = 4*pi*r^2 = A_sphere
        graph.add_node(-1, **graph.nodes[0])
        graph.add_edge(-1, soma, l=2 * graph.nodes[soma]["r"])
        graph = nx.relabel_nodes(graph, {i: i + 1 for i in graph.nodes})

        # edges connecting nodes to soma are considered part of the soma -> l = 0.
        for i, j in (*graph.in_edges(soma), *graph.out_edges(soma)):
            graph.edges[i, j]["l"] = 0

    # ensure linear root segment to ensure root branch can be created.
    if graph.out_degree(0) > 1:
        graph.add_node(-1, **graph.nodes[0])
        graph.add_edge(-1, 0, l=0.1)
        graph = nx.relabel_nodes(graph, {i: i + 1 for i in graph.nodes})

    graph = trace_branches(graph, max_branch_len, ignore_swc_trace_errors)

    graph = insert_compartments(graph, ncomp)

    comp_graph = extract_comp_graph(graph)

    # post-processing

    # rename attrs
    # Description of SWC file format:
    # http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
    group_ids = {0: "undefined", 1: "soma", 2: "axon", 3: "basal", 4: "apical"}
    for n in comp_graph.nodes:
        comp_graph.nodes[n]["group"] = [group_ids[comp_graph.nodes[n].pop("id")]]
        comp_graph.nodes[n]["radius"] = comp_graph.nodes[n].pop("r")
        comp_graph.nodes[n]["length"] = comp_graph.nodes[n].pop("comp_length")

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
    nodes_in_branches = edges_in_branches.apply(lambda x: branch_e2n(x.index.values))
    stack_branch_xyzr = lambda x: np.stack([unpack(graph.nodes[n], "xyzr") for n in x])
    xyzr = nodes_in_branches.apply(stack_branch_xyzr).to_list()
    same_rows = lambda x: np.all(np.nan_to_num(x[0]) == np.nan_to_num(x))
    xyzr = [xyzr_i[[0, -1]] if same_rows(xyzr_i) else xyzr_i for xyzr_i in xyzr]

    comp_graph.graph["xyzr"] = xyzr
    comp_graph.graph["type"] = "cell"

    return comp_graph


def infer_module_type_from_inds(idxs: pd.DataFrame) -> str:
    nuniques = idxs[["cell_index", "branch_index", "comp_index"]].nunique()
    nuniques.index = ["cell", "branch", "compartment"]
    nuniques = pd.concat([pd.Series({"network": 1}), nuniques])
    return_type = nuniques.loc[nuniques == 1].index[-1]
    return return_type


def build_module_scaffold(
    idxs: pd.DataFrame,
    return_type: Optional[str] = None,
    parent_branches: Optional[List[np.ndarray]] = None,
) -> Union[jx.Network, jx.Cell, jx.Branch, jx.Compartment]:
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
        return_type = infer_module_type_from_inds(idxs)

    comp = jx.Compartment()
    build_cache["compartment"] = [comp]

    if return_type in return_types[1:]:
        nsegs = idxs["branch_index"].value_counts().iloc[0]
        branch = jx.Branch([comp for _ in range(nsegs)])
        build_cache["branch"] = [branch]

    if return_type in return_types[2:]:
        for cell_id, cell_groups in idxs.groupby("cell_index"):
            num_branches = cell_groups["branch_index"].nunique()
            default_parents = np.arange(num_branches) - 1  # ignores morphology
            parents = (
                default_parents if parent_branches is None else parent_branches[cell_id]
            )
            cell = jx.Cell([branch] * num_branches, parents)
            build_cache["cell"].append(cell)

    if return_type in return_types[3:]:
        build_cache["network"] = [jx.Network(build_cache["cell"])]

    module = build_cache[return_type][0]
    build_cache.clear()
    return module


def from_graph(
    graph: nx.DiGraph,
    ncomp: int = 4,
    max_branch_len: float = 2000.0,
    assign_groups: bool = True,
    ignore_swc_trace_errors: bool = True,
) -> Union[jx.Network, jx.Cell, jx.Branch, jx.Compartment]:
    """Build a module from a networkx graph.

    All information stored in the nodes of the graph is imported into the Module.nodes
    attribute. Edges of type "inter_branch" connect two different branches, while
    edges of type "intra_branch" connect compartments within the same branch. All other
    edges are considered synapse edges. These are added to the Module.branch_edges and
    Module.edges attributes, respectively. Additionally, the graph can contain
    global attributes, which are added as attrs, i.e. to the module instance and
    optionally can store recordings, externals, groups, and trainables. These are
    imported from the node attributes of the graph. See `to_graph` for how they
    are formatted.

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
    - nodes:
        - id: int (used to define groups, according to NEURON's SWC convention)
        - groups: list[str] (can also be defined direclty)
        - cell_index: int
        - branch_index: int
        - comp_index: int
        - radius: float
        - length: float
        - x: float
        - y: float
        - z: float
        - recordings: list[str]
        - externals: list[float]
        - trainable: dict[str, float]
    - edges:
        - type: str ("synapse" or "inter_branch" / "intra_branch" or None)
        - parameters: list[dict] (stores synapse parameters)

    Args:
        graph: A networkx graph representing a module.
        nseg: The default number of segments per compartment.
            Will only be selected if the graph has not been compartmentalized yet.
        max_branch_len: Maximal length of one branch. If a branch exceeds this length,
            it is split into equal parts such that each subbranch is below
            `max_branch_len`. Will only be used if no branch structure is has been
            assigned yet.
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
    edge_df = pd.DataFrame([d for i, j, d in graph.edges(data=True)], index=graph.edges)

    # drop special attrs from nodes and ignore error if col does not exist
    # x,y,z can be re-computed from xyzr if needed
    optional_attrs = ["recordings", "externals", "trainable", "x", "y", "z"]
    node_df = node_df.drop(columns=optional_attrs, errors="ignore")

    # synapses
    synapse_edges = edge_df[edge_df["type"] == "synapse"]
    synapse_edges = synapse_edges.drop(["l", "type"], axis=1)
    synapse_edges = synapse_edges.rename({"syn_type": "type"}, axis=1)

    # branches
    branch_edges = edge_df[edge_df["type"] == "inter_branch"]
    branch_edges = branch_edges.drop(["l", "type"], axis=1).reset_index(
        names=["parent_branch_index", "child_branch_index"]
    )
    branch_edges = branch_edges.map(lambda x: node_df["branch_index"].loc[x])

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
    graph.graph.pop("type")
    for k, v in graph.graph.items():
        setattr(module, k, v)

    if assign_groups:
        groups = node_df.pop("group").explode()
        groups = (
            pd.DataFrame(groups)
            .groupby("group")
            .apply(lambda x: x.index.values, include_groups=False)
            .to_dict()
        )
        module.groups = groups

    node_df = node_df.rename(
        {k: f"global_{k}" for k in ["comp_index", "branch_index", "cell_index"]}, axis=1
    )
    module.nodes[node_df.columns] = (
        node_df  # set column-wise. preserves cols not in nodes.
    )
    module.edges = synapse_edges.T if not synapse_edges.empty else module.edges

    # add all the extra attrs
    module.membrane_current_names = [c.current_name for c in module.channels]
    module.synapse_names = [s._name for s in module.synapses]

    # if not recordings.empty:
    #     recordings = recordings.T.unstack().reset_index().set_index("level_0")
    #     recordings = recordings.rename(columns={"level_1": "rec_index", 0: "state"})
    #     module.recordings = recordings

    # if not externals.empty:
    #     cached_external_inds = {}
    #     cached_externals = {}
    #     for key, data in externals.items():
    #         cached_externals[key] = jnp.array(
    #             np.stack(data[~data.isna()].explode().values)
    #         )
    #         cached_external_inds[key] = jnp.array(
    #             data[~data.isna()].explode().index.to_numpy()
    #         )
    #     module.externals = cached_externals
    #     module.external_inds = cached_external_inds

    # if not trainables.empty:
    #     # trainables require special handling, since some of them are shared
    #     # and some are set individually
    #     trainables = pd.DataFrame(
    #         nx.get_node_attributes(graph, "trainable"), dtype=float
    #     )
    #     # prepare trainables dataframe
    #     trainables = trainables.T.unstack().reset_index().dropna()
    #     trainables = trainables.rename(
    #         columns={"level_0": "param", "level_1": "index", 0: "value"}
    #     )
    #     # 1. merge indices with same trainables into lists
    #     grouped_trainables = trainables.groupby(["param", "value"])
    #     merged_trainables = grouped_trainables.agg(
    #         {"index": lambda x: jnp.array(x.values)}
    #     ).reset_index()
    #     concat_here = merged_trainables["index"].apply(lambda x: len(x) == 1)

    #     # 2. split into shared and seperate trainables
    #     shared_trainables = merged_trainables.loc[~concat_here]
    #     sep_trainables = (
    #         merged_trainables.loc[concat_here].groupby("param").agg(list).reset_index()
    #     )
    #     # 3. convert lists to jnp arrays and stack indices
    #     sep_trainables.loc[:, "index"] = sep_trainables["index"].apply(jnp.stack)
    #     sep_trainables.loc[:, "value"] = sep_trainables["value"].apply(jnp.array)
    #     shared_trainables.loc[:, "value"] = shared_trainables["value"].apply(np.array)

    #     # 4. format to match module.trainable_params and module.indices_set_by_trainables
    #     trainable_params = pd.concat([shared_trainables, sep_trainables])
    #     indices_set_by_trainables = trainable_params["index"].values.tolist()
    #     trainable_params = [
    #         {k: jnp.array(v).reshape(-1)}
    #         for k, v in trainable_params[["param", "value"]].values
    #     ]
    #     module.trainable_params = trainable_params
    #     module.indices_set_by_trainables = indices_set_by_trainables

    return module


def to_graph(module: jx.Module) -> nx.DiGraph:
    """Export the module as a networkx graph.

    Constructs a nx.DiGraph from the module. Each compartment in the module
    is represented by a node in the graph. The edges between the nodes represent
    the connections between the compartments. These edges can either be connections
    between compartments within the same branch, between different branches or
    even between different cells. In this case the latter the synapse parameters
    are stored as edge attributes. Additionally, global attributes of the module,
    for example `nseg`, are stored as graph attributes.

    Exported graphs can be imported again to `jaxley` using the `from_graph` method.

    Args:
        module: A jaxley module or view instance.

    Returns:
        A networkx graph of the module.
    """
    module_graph = nx.DiGraph()
    module.compute_compartment_centers()  # make xyz coords attr of nodes

    # add global attrs
    module_graph.graph["type"] = module.__class__.__name__.lower()
    for attr in [
        "ncomp",
        "initialized_morph",
        "initialized_syns",
        "synapses",
        "channels",
        "allow_make_trainable",
        "num_trainable_params",
        "xyzr",
    ]:
        module_graph.graph[attr] = getattr(module, attr)

    # add nodes
    nodes = module.nodes
    nodes = nodes.drop([col for col in nodes.columns if "local" in col], axis=1)
    nodes = nodes.rename(
        {f"global_{k}": k for k in ["comp_index", "branch_index", "cell_index"]}, axis=1
    )

    group_inds = pd.DataFrame(
        [(k, v) for k, vals in module.groups.items() for v in vals],
        columns=["group", "index"],
    )
    nodes = pd.concat([nodes, group_inds.groupby("index")["group"].agg(list)], axis=1)

    module_graph = nx.DiGraph()
    module_graph.add_nodes_from(nodes.T.to_dict().items())

    syn_edges = module.edges
    # syn_edges["syn_type"] = syn_edges["type"]
    # syn_edges["type"] = "synapse"
    # syn_edges = syn_edges.set_index(["pre_global_comp_index", "post_global_comp_index"])
    # module_graph.add_edges_from(syn_edges)

    inter_branch_edges = module.branch_edges.copy()
    intra_branch_edges = []
    for i, branch_data in nodes.groupby("branch_index"):
        inds = branch_data.index.values
        intra_branch_edges += branch_n2e(inds)

        parents = module.branch_edges["parent_branch_index"]
        children = module.branch_edges["child_branch_index"]
        inter_branch_edges.loc[parents == i, "parent_branch_index"] = inds[-1]
        inter_branch_edges.loc[children == i, "child_branch_index"] = inds[0]

    inter_branch_edges = inter_branch_edges.to_numpy()
    module_graph.add_edges_from(inter_branch_edges, type="inter_branch")
    module_graph.add_edges_from(intra_branch_edges, type="intra_branch")

    # # add recordings to nodes
    # if not module.recordings.empty:
    #     for index, group in module.recordings.groupby("rec_index"):
    #         rec_index = group["rec_index"].loc[0]
    #         rec_states = group["state"].values
    #         module_graph.add_node(rec_index, **{"recordings": rec_states})

    # # add externals to nodes
    # if module.externals is not None:
    #     for key, inds in module.external_inds.items():
    #         unique_inds = np.unique(inds.flatten())
    #         for i in unique_inds:
    #             which = np.where(inds == i)[0]
    #             if "externals" not in module_graph.nodes[i]:
    #                 module_graph.nodes[i]["externals"] = {}
    #             module_graph.nodes[i]["externals"].update(
    #                 {key: module.externals[key][which]}
    #             )

    # # add trainable params to nodes
    # if module.trainable_params:
    #     d = {"index": [], "param": [], "value": []}
    #     for params, inds in zip(
    #         module.trainable_params, module.indices_set_by_trainables
    #     ):
    #         inds = inds.flatten()
    #         key, values = next(iter(params.items()))
    #         d["index"] += inds.tolist()
    #         d["param"] += np.broadcast_to([key], inds.shape).tolist()
    #         d["value"] += np.broadcast_to(values.flatten(), inds.shape).tolist()
    #     df = pd.DataFrame(d)
    #     to_dicts = lambda x: x.set_index("param").to_dict()["value"]
    #     trainable_iter = df.groupby("index").apply(to_dicts).to_dict()
    #     trainable_iter = {k: {"trainable": v} for k, v in trainable_iter.items()}
    #     module_graph.add_nodes_from(trainable_iter.items())

    return module_graph
