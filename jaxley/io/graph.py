from math import pi
from typing import Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
from jax import vmap
import networkx as nx
import numpy as np
import pandas as pd

import jaxley as jx


def build_skeleton_module(
    idxs: pd.DataFrame, return_type: Optional[str] = None
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
    build_cache = {
        "compartment": [jx.Compartment()],
        "branch": [],
        "cell": [],
        "network": [],
    }
    for cell_id, cell_groups in idxs.groupby("cell_index"):
        build_cache["branch"].clear()  # clear build cache
        for branch_id, branch_groups in cell_groups.groupby("branch_index"):
            comps = [jx.Compartment() for _ in range(len(branch_groups))]
            build_cache["branch"].append(jx.Branch(comps))

        parents = np.arange(len(build_cache["branch"])) - 1  # ignores morphology
        build_cache["cell"].append(jx.Cell(build_cache["branch"], parents))
    build_cache["network"] = [jx.Network(build_cache["cell"])]

    if return_type is None:  # infer return type from idxs
        nuniques = idxs.nunique()[["cell_index", "branch_index", "comp_index"]]
        nuniques.index = ["cell", "branch", "compartment"]
        nuniques = pd.concat([pd.Series({"network": 1}), nuniques])
        return_type = nuniques.loc[nuniques == 1].index[-1]

    module = build_cache[return_type][0]
    build_cache.clear()
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
    module._update_nodes_with_xyz()  # make xyz coords attr of nodes

    # add global attrs
    module_graph.graph["module"] = module.__class__.__name__
    module_graph.graph["nseg"] = module.nseg
    module_graph.graph["total_nbranches"] = module.total_nbranches
    module_graph.graph["cumsum_nbranches"] = module.cumsum_nbranches
    module_graph.graph["initialized_morph"] = module.initialized_morph
    module_graph.graph["initialized_syns"] = module.initialized_syns
    module_graph.graph["synapses"] = module.synapses
    module_graph.graph["channels"] = module.channels
    module_graph.graph["allow_make_trainable"] = module.allow_make_trainable
    module_graph.graph["num_trainable_params"] = module.num_trainable_params
    module_graph.graph["xyzr"] = module.xyzr

    # add nodes
    module_graph.add_nodes_from(module.nodes.iterrows())

    # add groups to nodes
    group_node_dict = {k: list(v.index) for k, v in module.group_nodes.items()}
    nodes_in_groups = sum(list(group_node_dict.values()), [])
    node_group_dict = {
        k: [i for i, v in group_node_dict.items() if k in v] for k in nodes_in_groups
    }
    # asumes multiple groups per node are allowed
    for idx, key in node_group_dict.items():
        module_graph.add_node(idx, **{"groups": key})

    # add recordings to nodes
    if not module.recordings.empty:
        for index, group in module.recordings.groupby("rec_index"):
            rec_index = group["rec_index"].loc[0]
            rec_states = group["state"].values
            module_graph.add_node(rec_index, **{"recordings": rec_states})

    # add currents to nodes
    if module.currents is not None:
        for index, currents in zip(module.current_inds.index, module.currents):
            module_graph.add_node(index, **{"currents": currents})

    # add trainable params to nodes
    if module.trainable_params:
        d = {"index": [], "param": [], "value": []}
        for params, inds in zip(
            module.trainable_params, module.indices_set_by_trainables
        ):
            inds = inds.flatten()
            key, values = next(iter(params.items()))
            d["index"] += inds.tolist()
            d["param"] += np.broadcast_to([key], inds.shape).tolist()
            d["value"] += np.broadcast_to(values.flatten(), inds.shape).tolist()
        df = pd.DataFrame(d)
        to_dicts = lambda x: x.set_index("param").to_dict()["value"]
        trainable_iter = df.groupby("index").apply(to_dicts).to_dict()
        trainable_iter = {k: {"trainable": v} for k, v in trainable_iter.items()}
        module_graph.add_nodes_from(trainable_iter.items())

    # connect comps within branches
    for index, group in module.nodes.groupby("branch_index"):
        module_graph.add_edges_from(zip(group.index[:-1], group.index[1:]))

    # connect branches
    for index, edge in module.branch_edges.iterrows():
        parent_branch_idx = edge["parent_branch_index"]
        parent_comp_idx = max(
            module.nodes[module.nodes["branch_index"] == parent_branch_idx].index
        )
        child_branch_idx = edge["child_branch_index"]
        child_comp_idx = min(
            module.nodes[module.nodes["branch_index"] == child_branch_idx].index
        )
        module_graph.add_edge(parent_comp_idx, child_comp_idx, type="branch")

    # connect synapses
    for index, edge in module.edges.iterrows():
        attrs = edge.to_dict()
        pre = attrs["global_pre_comp_index"]
        post = attrs["global_post_comp_index"]
        module_graph.add_edge(pre, post, **attrs)
    return module_graph


is_leaf = lambda graph, n: graph.out_degree(n) == 0 and graph.in_degree(n) == 1
is_branching = lambda graph, n: graph.out_degree(n) > 1
path_e2n = lambda path: [path[0][0]] + [e[1] for e in path]
path_n2e = lambda path: [e for e in zip(path[:-1], path[1:])]
jnp_interp = vmap(jnp.interp, in_axes=(None, None, 1))
unpack = lambda d, keys: [d[k] for k in keys]
has_node_attr = lambda graph, attr: all(attr in graph.nodes[n] for n in graph.nodes)
has_edge_attr = lambda graph, attr: all(attr in graph.edges[e] for e in graph.edges)


def node_dist(graph: nx.DiGraph, i: int, j: int) -> float:
    """Compute the euclidean distance between two nodes in a graph.

    Args:
        graph: A networkx graph.
        i: Index of the first node.
        j: Index of the second node.

    Returns:
        The euclidean distance between the two nodes."""
    pre_loc = np.hstack(unpack(graph.nodes[i], "xyz"))
    post_loc = np.hstack(unpack(graph.nodes[j], "xyz"))
    return np.sqrt(np.sum((pre_loc - post_loc) ** 2))


def get_linear_paths(graph: nx.DiGraph) -> List[np.ndarray]:
    """Get all linear paths in a graph.

    The graph is traversed depth-first starting from the root node.

    Args:
        graph: A networkx graph.

    Returns:
        A list of linear paths in the graph. Each path is represented as an array of
        edges."""
    paths, current_path = [], []
    for i, j in nx.dfs_edges(graph, 0):
        current_path.append((i, j))
        if is_leaf(graph, j) or is_branching(graph, j):
            paths.append(np.array(current_path))
            current_path.clear()
    return paths


def add_edge_lengths(
    graph: nx.DiGraph, edges: Optional[List[Tuple]] = None
) -> nx.DiGraph:
    """Add the length edges to the graph.

    The length is computed as the euclidean distance between the nodes of the edge.
    If no edges are provided, the length is computed for all edges in the graph.

    Args:
        graph: A networkx graph.
        edges: A list of edges for which to compute the length. If None, the length is
            computed for all edges.

    Returns:
        The graph with the length attribute added to the edges."""
    has_loc = "x" in graph.nodes[0]
    edges = graph.edges if edges is None else edges
    for i, j in edges:
        graph.edges[i, j]["length"] = node_dist(graph, i, j) if has_loc else 1
    return graph


def resample_path(
    graph: nx.DiGraph,
    path: List[Tuple],
    locs: Union[List, np.ndarray],
    interp_attrs: List[str] = ["x", "y", "z", "r"],
    ensure_unique_node_inds: bool = False,
) -> nx.DiGraph:
    """Resample a path in a graph at different locations along its length.

    Selected node attributes are interpolated linearly between the nodes of the path.

    Args:
        graph: A networkx graph.
        path: A list of edges representing the path.
        locs: A list of locations along the path at which to resample.
        interp_attrs: A list of node attributes to interpolate.
        ensure_unique_node_inds: Add a small random number to the new nodes to ensure
            unique node indices. This is useful when resampling multiple paths. That
            are added to the same graph afterwards. Ensures the graphs are disjoint.

    Returns:
        A new graph with the resampled path."""
    path_nodes = np.array(path_e2n(path))
    lengths = [graph.edges[(i, j)]["length"] for i, j in path]
    pathlens = np.cumsum(np.array([0] + lengths))
    new_nodes = np.interp(locs, pathlens, path_nodes)
    # ensure unique nodes -> disjoint subgraphs
    new_nodes += 1e-5 * np.random.randn() if ensure_unique_node_inds else 0

    node_attrs = np.vstack([unpack(graph.nodes[n], interp_attrs) for n in path_nodes])
    new_node_attrs = jnp_interp(new_nodes, path_nodes, node_attrs).T.tolist()
    new_edge_attrs = [{"length": d} for d in np.diff(locs)]
    new_edges = path_n2e(new_nodes)
    pathgraph = nx.DiGraph(
        (i, j, attrs) for (i, j), attrs in zip(new_edges, new_edge_attrs)
    )
    pathgraph.add_nodes_from(
        (i, dict(zip(interp_attrs, attrs)))
        for i, attrs in zip(new_nodes, new_node_attrs)
    )
    return pathgraph


def split_edge(
    graph: nx.DiGraph,
    i: int,
    j: int,
    nsegs: int = 2,
    interp_node_attrs: List[str] = ["x", "y", "z", "r"],
) -> nx.DiGraph:
    """Split an edge in a graph into multiple segments.

    The edge is split into `nsegs` segments of equal length. The node attributes are
    interpolated linearly between the nodes of the edge.

    Args:
        graph: A networkx graph.
        i: Index of the first node of the edge.
        j: Index of the second node of the edge.
        nsegs: Number of segments to split the edge into.
        interp_node_attrs: A list of node attributes to interpolate.

    Returns:
        The graph with specified edge split into multiple segments."""
    edge = graph.edges[(i, j)]
    graph = graph if "length" in edge else add_edge_lengths(graph, [(i, j)])
    locs = np.linspace(0, edge["length"], nsegs + 1)
    pathgraph = resample_path(graph, [(i, j)], locs, interp_node_attrs)

    for attr in ["id", "branch_index"]:
        if attr in edge:
            nx.set_edge_attributes(pathgraph, edge[attr], attr)

    graph.add_edges_from(pathgraph.edges(data=True))
    graph.add_nodes_from(pathgraph.nodes(data=True))
    graph.remove_edge(i, j)
    return graph


def impose_branch_structure(
    graph: nx.DiGraph, max_branch_len: float = 100
) -> nx.DiGraph:
    """Impose branch structure on a graph representing a morphology.

    Adds branch indices to the edges of the graph. Graph needs to have nodes with
    "id" and either x,y,z as node or length as edge attributes.

    In order to simulate a morphology, that is specified by a graph, the graph has
    to be first segmented into branches, as is also done by NEURON. We do this by
    adding branch_index labels to the graph. We add these to indices to the branches
    rather than nodes since the (continuous) morphology (and hence attributes, i.e.
    radius) is defined by the graph's edges rather than its nodes.

    To do this we first add the "id"s (which are attributes of the nodes) to the
    edges of the graph. If i and j are connected by an edge, then the edge is
    labelled with the id of i. If i and j have different ids, then the edge is
    split into two edges of equal length and each edge is labelled with the id
    of i and j. Attributes of the added node are interpolated linearly between
    the nodes of the edge.

    For example:
    Graph:      |------|---------|--------------|
    radius:     1      2         3              4
    ID:         1      1         4              4

    is relabelled as:
    Graph:      |------|----|----|--------------|
    radius:     1      2   2.5   3              4
    ID:             1     1    4         4


    Then branch indices are assigned such that the length of each branch is below
    a maximum branch length and only belongs to has one "id". If it exceeds the
    maximum branch length, it is split into multiple branches. That are each
    of roughly equal length.

    For a segment of the graph with the following structure and a maximum branch
    length of 20:

    Graph: |----|------|----------|--------------|-----|---|--|----|----|---------|
    ID:       1     1        1            1         4    4   4   4    4      4
    Length:   4     6       10           14         5    3   2   4    4      9

    The graph is split into branches roughly as follows:
    Graph: |----|------|----------|--------------|-----|---|--|----|----|---------|
    ID:       1     1        1            1         4    4   4   4    4      4
    Length:   4     6       10           14         5    3   2   4    4      9
    Branch:   0     0        0            1         2    2   2   2    3      3
    B.-length:        20          |      14      |       12        |      13      |

    This means we now have a continuous morphology with edges assigned to specific
    branches and types.

    Args:
        graph: A networkx graph representing a morphology.
        max_branch_len: Maximum length of a branch.

    Returns:
        The graph with the edges labelled by "id" and "branch_index".
    """
    assert has_node_attr(graph, "id"), "Graph nodes must have an 'id' attribute."
    assert has_node_attr(graph, "x") or has_edge_attr(
        graph, "length"
    ), "Graph must have xyz locs as node or lengths as edge attribute."
    if not has_edge_attr(graph, "length"):  # compute lens from xyz if not present
        graph = add_edge_lengths(graph, graph.edges)

    # add id to edges, split edges if necessary
    for i, j in list(graph.edges):
        if graph.nodes[i]["id"] == graph.nodes[j]["id"]:
            graph.edges[i, j]["id"] = graph.nodes[i]["id"]
        else:
            graph = split_edge(graph, i, j, nsegs=2)
            edge_i, edge_j = path_n2e(nx.shortest_path(graph, i, j))
            graph.edges[edge_i]["id"] = graph.nodes[i]["id"]
            graph.edges[edge_j]["id"] = graph.nodes[j]["id"]

    # Split edge if single edge is longer than max_branch_len
    # run after adding ids to edges, so that new edges are also labelled
    edge_lengths = nx.get_edge_attributes(graph, "length")
    too_long_edges = {k: v for k, v in edge_lengths.items() if v > max_branch_len}
    for i, j in too_long_edges:
        nsegs = graph.edges[i, j]["length"] // max_branch_len + 1
        graph = split_edge(graph, i, j, nsegs=nsegs)

    # rm after, since same id label might be used multiple times in for loop
    [n.pop("id") for i, n in graph.nodes(data=True) if "id" in n]
    new_keys = {k: i for i, k in enumerate(sorted(graph.nodes))}
    graph = nx.relabel_nodes(graph, new_keys)  # ensure node labels are integers

    max_branch_idx = 0
    # segment linear sections of the graph/morphology
    linear_paths = get_linear_paths(graph)
    for path in linear_paths:
        ids = [graph.edges[(i, j)]["id"] for i, j in path]
        unique_ids = np.unique(ids)
        where_single_ids = ids == unique_ids[:, None]
        for single_id in where_single_ids:
            lengths = [graph.edges[(i, j)]["length"] for i, j in path[single_id]]

            # segment morphology into branches based on max_branch_len and ids
            pathlens = np.cumsum(lengths)
            total_pathlen = pathlens[-1]
            num_branches = int(total_pathlen / max_branch_len) + 1
            branch_ends = np.cumsum([total_pathlen / num_branches] * num_branches)
            branch_inds = (
                max_branch_idx
                + num_branches
                - np.sum(pathlens <= branch_ends[:, None], axis=0)
            )
            max_branch_idx += num_branches
            graph.add_edges_from(
                (
                    (*e, {"branch_index": branch_idx})
                    for e, branch_idx in zip(path[single_id], branch_inds)
                )
            )
    return graph


def compartmentalize_branches(
    graph: nx.DiGraph, nseg: int = 4, append_morphology: bool = True
) -> nx.DiGraph:
    """Compartmentalize the morphology, by compartmentalizing the branches.

    Currently, the nodes in the graph have no particular meaning. While they hold
    the x,y,z,r attributes, the graph nodes just represent points at which the
    morphology was sampled / measured. In order to simulate the morphology, the graph
    hence needs to be compartmentalized first. For this we place discrete
    compartments along the morphology. In the resulting compartmentalized graph
    nodes take the form of compartments and edges represent the connections between
    compartments. Therefore each branch is split into `nseg` compartments along its
    length, with the attributes being linearly interpolated along the existing
    'measured' nodes of the edge. Each compartment therefore has length=branch_length/nseg
    and is represented by a node in the graph at each center.

    For example for nseg=4 and a branch with the following structure:
    Graph: |----|------|----------|------------|-----|---|--|--|----|--------|
    Length:   4     6       10           12       5    3   2  2   4      8
    Branch:   0     0        0            1       2    2   2  2   3      3

    The graph is compartmentalized as follows
    (length now corresponds to compartment length, comp nodes marked with 'x'):
    Graph:     x----x----x----x---x--x--x--x--x--x--x--x--x--x--x--x
    Length:    5    5    5    5   3  3  3  3  3  3  3  3  3  3  3  3
    Branch:    0    0    0    0   1  1  1  1  2  2  2  2  3  3  3  3

    Args:
        graph: A networkx graph representing a morphology.
        nseg: Number of compartments to split each branch into.
        append_morphology: Wether to append the original morphology to the graph
            as an attribute.

    Returns:
        The graph with the branches compartmentalized into nseg compartments.
    """
    assert has_edge_attr(
        graph, "branch_index"
    ), "Graph edges must have branches indices."

    edges = pd.DataFrame([d for i, j, d in graph.edges(data=True)], index=graph.edges)
    edges = edges.reset_index(names=["i", "j"]).sort_values(["branch_index", "i", "j"])
    branch_groups = edges.groupby("branch_index")

    # individually compartmentalize branches and save resulting graph structure
    branchgraphs = []
    xyzr = []
    for idx, group in branch_groups:
        pathlens = group["length"].cumsum()
        total_pathlen = pathlens.iloc[-1]
        comp_len = total_pathlen / nseg
        locs = np.linspace(comp_len / 2, total_pathlen - comp_len / 2, nseg)
        path = group[["i", "j"]].values
        branchgraph = resample_path(graph, path, locs, ensure_unique_node_inds=False)
        nx.set_node_attributes(branchgraph, comp_len, "comp_len")
        nx.set_node_attributes(branchgraph, idx, "branch_index")
        nx.set_node_attributes(branchgraph, group["id"].iloc[0], "id")
        branchgraphs.append(branchgraph)

        branch_xyzr = np.array(
            [[graph.nodes[idx][k] for k in "xyzr"] for idx in path_e2n(path)]
        )
        xyzr.append(branch_xyzr)

    # merge compartmentalized branches into single graph
    new_graph = nx.union_all(branchgraphs)

    # reconnect seperate branches
    branch_roots = branch_groups.first()["i"]
    branch_leaves = branch_groups.last()["j"]
    branch_roots_leaves = pd.concat([branch_roots, branch_leaves], axis=1)
    root_equal_leaf = np.equal(*np.meshgrid(*(branch_roots_leaves.values.T)))
    edges_between_branches = list(zip(*np.where(root_equal_leaf)))
    branch2node_edge = lambda i, j: (min(branchgraphs[i]), min(branchgraphs[j]))
    new_edges_between_branches = [
        branch2node_edge(i, j) for i, j in edges_between_branches
    ]
    new_graph.add_edges_from(new_edges_between_branches)

    # reconnect root
    root_edges = list(zip(*np.where([branch_roots_leaves["i"] == 0])))[1:]  # drop (0,0)
    new_root_edges = [
        (min(branchgraphs[i]), min(branchgraphs[j])) for i, j in root_edges
    ]
    new_graph.add_edges_from(new_root_edges)

    # set node labels to integers
    new_keys = {k: i for i, k in enumerate(sorted(new_graph.nodes))}
    new_graph = nx.relabel_nodes(new_graph, new_keys)

    # TODO: handle soma (r == l)

    if append_morphology:
        new_graph.graph["xyzr"] = xyzr

    return new_graph


def from_graph(
    graph: nx.DiGraph,
    nseg: int = 4,
    max_branch_len: float = 300.0,
    assign_groups: bool = True,
) -> Union[jx.Network, jx.Cell, jx.Branch, jx.Compartment]:
    """Build a module from a networkx graph.

    All information stored in the nodes of the graph is imported into the Module.nodes
    attribute. Edges of type "branch" are considered branch edges, while all other
    edges are considered synapse edges. These are added to the Module.branch_edges and
    Module.edges attributes, respectively. Additionally, the graph can contain
    global attributes, which are added as attrs, i.e. to the module instance and
    optionally can store recordings, currents, groups, and trainables. These are
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
    the graph has to be connected and cannot contain loops, since we cannot
    distinguish between disconnected compartments or disconnected cells in a
    network purely from the graph structure.

    Returns nx.Network, nx.Cell, nx.Branch, or nx.Compartment depending on
    num_cell_idxs > 1, num_branch_idxs > 1 etc.

    Possible attributes that can be read off of the graph include:
    - graph
        - nseg: int
        - module: str
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
        - currents: list[float]
        - trainable: dict[str, float]
    - edges:
        - parent_branch_index: int
        - child_branch_index: int
        - global_pre_comp_index: int
        - global_post_comp_index: int
        - type: str ("synapse" or "branch" or None)

    Args:
        graph: A networkx graph representing a module.
        default_nseg: The default number of segments per compartment.
            Will only be selected if the graph does not have nseg as attr.
        max_branch_len: Maximal length of one branch. If a branch exceeds this length,
            it is split into equal parts such that each subbranch is below
            `max_branch_len`.
        assign_groups: Wether to assign groups to nodes based on the the id or groups
            attribute.

    Returns:
        A module instance that is populated with the node and egde attributes of
        the nx.DiGraph."""

    graph_was_exported_with_jaxley = "module" in graph.graph

    ########################################
    ### Make the graph jaxley compatible ###
    ########################################

    if not graph_was_exported_with_jaxley:
        # Graph only has to contain edges to be importable as a module
        # In this case, xyz are set to NaN, r to 1 and lengths to 10
        assert nx.is_weakly_connected(
            graph
        ), "Graph must be connected to be imported as a module."
        assert len(list(nx.simple_cycles(graph))) == 0, "Graph cannot have loops"
        if not graph.edges:
            raise ValueError("Graph must have edges to be imported as a module.")

        # fill graph with defaults if not present
        default_node_attrs = {"x": np.nan, "y": np.nan, "z": np.nan, "r": 1, "id": 0}
        for k, v in default_node_attrs.items():
            if not has_node_attr(graph, k):
                nx.set_node_attributes(graph, v, k)

        default_edge_attrs = {"length": 10.0}
        for k, v in default_edge_attrs.items():
            if not has_edge_attr(graph, k):
                nx.set_edge_attributes(graph, v, k)

        # add comp_index, branch_index, cell_index to graph
        # first check if they are already present, otherwise compute them
        if not has_node_attr(graph, "branch_index"):
            nseg = graph.graph["nseg"] if "nseg" in graph.graph else nseg

            # add branch_index and segment morphology into branches and compartments
            roots = np.where([graph.in_degree(n) == 0 for n in graph.nodes])[0]
            assert len(roots) == 1, "Currently only 1 morphology can be imported."
            graph = impose_branch_structure(graph, max_branch_len=max_branch_len)
            graph = compartmentalize_branches(graph)
            nx.set_node_attributes(graph, {i: 0 for i in graph.nodes}, "cell_index")
            nx.set_node_attributes(graph, {i: i for i in graph.nodes}, "comp_index")

        # setup branch structure and compute parents
        # edges connecting comps in different branches are set to type "branch"
        # this works for mutliple cells in theory, however only one cell is currently supported
        if nx.get_edge_attributes(graph, "type") == {}:
            cell_idxs = nx.get_node_attributes(graph, "cell_index")
            unique_cells = np.unique(list(cell_idxs.values()))
            graph.graph["comb_parents"] = []
            for idx in unique_cells:
                cell_nodes = [k for k, v in cell_idxs.items() if v == idx]
                cell_graph = graph.subgraph(cell_nodes)
                branch_idxs = nx.get_node_attributes(cell_graph, "branch_index")
                within_branch = np.array(
                    [(branch_idxs[i] == branch_idxs[j]) for i, j in graph.edges]
                )
                node_connections = np.stack([(i, j) for i, j in graph.edges])
                pre, post = node_connections[~within_branch].T
                graph.add_edges_from(zip(pre, post), type="branch")
                branch_connections = [
                    (branch_idxs[i], branch_idxs[j]) for i, j in zip(pre, post)
                ]
                parents = [-1]
                if len(branch_connections) > 0:
                    dfs_parents = nx.dfs_predecessors(nx.DiGraph(branch_connections), 0)
                    parents += list(dfs_parents.values())
                graph.graph["comb_parents"] += parents

        # add group information to nodes if available via id attribute
        ids = nx.get_node_attributes(graph, "id")
        if ids != {}:
            group_ids = {0: "undefined", 1: "soma", 2: "axon", 3: "basal", 4: "apical"}
            # Type of padded section is assumed to be of `custom` type:
            # http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html

            groups = [
                group_ids[id] if id in group_ids else f"custom{id-4}"
                for id in ids.values()
            ]
            graph.add_nodes_from(
                {i: {"groups": [id]} for i, id in enumerate(groups)}.items()
            )
        for i, node in graph.nodes(data=True):
            node.pop("id")  # remove id attribute from nodes

    #################################
    ### Import graph as jx.Module ###
    #################################

    global_attrs = graph.graph.copy()
    # try to infer module type from global attrs (exported with `to_graph`)
    return_type = (
        global_attrs.pop("module").lower() if "module" in global_attrs else None
    )

    # build skeleton of module instances
    nodes = pd.DataFrame((n for i, n in graph.nodes(data=True)))
    idxs = nodes[["cell_index", "branch_index", "comp_index"]]
    module = build_skeleton_module(idxs, return_type)

    # set global attributes of module
    for k, v in global_attrs.items():
        setattr(module, k, v)

    # import nodes and edges
    edge_type = nx.get_edge_attributes(graph, "type")
    edges = pd.DataFrame(edge_type.values(), index=edge_type.keys(), columns=["type"])
    is_synapse = edges["type"] != "branch"
    idxs = edges[~is_synapse].reset_index().values.T[:2]
    branch_edges = pd.DataFrame([nodes["branch_index"].loc[i].values for i in idxs]).T
    branch_edges.columns = ["parent_branch_index", "child_branch_index"]

    synapse_edges = pd.DataFrame(
        [graph.edges[idxs] for idxs in edges[is_synapse].index]
    )

    # drop special attrs from nodes and ignore error if col does not exist
    optional_attrs = ["recordings", "currents", "groups", "trainables"]
    nodes.drop(columns=optional_attrs, inplace=True, errors="ignore")

    module.nodes[nodes.columns] = nodes  # set column-wise. preserves cols not in nodes.
    module.branch_edges = branch_edges
    module.edges = synapse_edges

    # Add optional attributes if they can be found in nodes
    recordings = pd.DataFrame(nx.get_node_attributes(graph, "recordings"))
    currents = pd.DataFrame(nx.get_node_attributes(graph, "currents"))
    groups = pd.DataFrame(nx.get_node_attributes(graph, "groups").items())
    trainables = pd.DataFrame(nx.get_node_attributes(graph, "trainable"), dtype=float)

    if not groups.empty and assign_groups:
        groups = groups.explode(1).rename(columns={0: "index", 1: "group"})
        groups = groups[groups["group"] != "undefined"]  # skip undefined comps
        group_nodes = {k: nodes.loc[v["index"]] for k, v in groups.groupby("group")}
        module.group_nodes = group_nodes
    if not recordings.empty:
        recordings = recordings.T.unstack().reset_index().set_index("level_0")
        recordings = recordings.rename(columns={"level_1": "rec_index", 0: "state"})
        module.recordings = recordings
    if not currents.empty:
        current_inds = nodes.loc[currents.T.index]
        currents = np.stack(currents.values)
        module.currents = currents
        module.current_inds = current_inds
    if not trainables.empty:
        # trainables require special handling, since some of them are shared
        # and some are set individually
        trainables = pd.DataFrame(
            nx.get_node_attributes(graph, "trainable"), dtype=float
        )
        # prepare trainables dataframe
        trainables = trainables.T.unstack().reset_index().dropna()
        trainables = trainables.rename(
            columns={"level_0": "param", "level_1": "index", 0: "value"}
        )
        # 1. merge indices with same trainables into lists
        grouped_trainables = trainables.groupby(["param", "value"])
        merged_trainables = grouped_trainables.agg(
            {"index": lambda x: jnp.array(x.values)}
        ).reset_index()
        concat_here = merged_trainables["index"].apply(lambda x: len(x) == 1)

        # 2. split into shared and seperate trainables
        shared_trainables = merged_trainables.loc[~concat_here]
        sep_trainables = (
            merged_trainables.loc[concat_here].groupby("param").agg(list).reset_index()
        )
        # 3. convert lists to jnp arrays and stack indices
        sep_trainables.loc[:, "index"] = sep_trainables["index"].apply(jnp.stack)
        sep_trainables.loc[:, "value"] = sep_trainables["value"].apply(jnp.array)
        shared_trainables.loc[:, "value"] = shared_trainables["value"].apply(np.array)

        # 4. format to match module.trainable_params and module.indices_set_by_trainables
        trainable_params = pd.concat([shared_trainables, sep_trainables])
        indices_set_by_trainables = trainable_params["index"].values
        trainable_params = [
            {k: jnp.array(v).reshape(-1)}
            for k, v in trainable_params[["param", "value"]].values
        ]

        module.trainable_params = trainable_params
        module.indices_set_by_trainables = indices_set_by_trainables

    return module
