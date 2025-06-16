# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from jaxley.modules import Branch, Cell, Compartment, Module, Network

########################################################################################
################################### BUILD SWC GRAPH ####################################
########################################################################################


def to_swc_graph(fname: str, num_lines: Optional[int] = None) -> nx.DiGraph:
    # def swc_to_nx(fname: str, num_lines: Optional[int] = None) -> nx.DiGraph:
    """Read a SWC morphology file into a NetworkX DiGraph.

    The graph is read such that each entry in the swc file becomes a graph node
    with the column attributes (id, x, y, z, r). Then each node is connected to its
    designated parent via an edge.

    Args:
        fname: Path to the SWC file
        num_lines: Number of lines to read from the file. If None, all lines are read.

    Returns:
        A networkx DiGraph of the traced morphology in the swc file. It has attributes:
        nodes: {'id': 1, 'x': 0.0, 'y': 0.0, 'z': 0.0, 'r': 1.0}
        edges: {}

    Example usage
    ^^^^^^^^^^^^^

    ::

        from jaxley.io.graph swc_to_nx
        swc_graph = swc_to_nx("path_to_swc.swc")
    """
    i_id_xyzr_p = np.loadtxt(fname)[:num_lines]

    graph = nx.DiGraph()
    for i, id, x, y, z, r, p in i_id_xyzr_p.tolist():  # tolist: np.float64 -> float
        graph.add_node(int(i), **{"id": int(id), "x": x, "y": y, "z": z, "r": r})
        if p != -1:
            graph.add_edge(int(p), int(i))
    return graph


########################################################################################
############################## BUILD COMPARTMENT GRAPH #################################
########################################################################################


def pandas_to_nx(
    node_attrs: pd.DataFrame, edge_attrs: pd.DataFrame, global_attrs: pd.Series
) -> nx.DiGraph:
    """Convert node_attrs, edge_attrs and global_attrs from pandas datatypes to a NetworkX DiGraph.

    Args:
        node_attrs: DataFrame containing node attributes
        edge_attrs: DataFrame containing edge attributes
        global_attrs: Series containing global graph attributes

    Returns:
        A directed graph with nodes, edges and global attributes from the input data.
    """
    G = nx.from_pandas_edgelist(
        edge_attrs.reset_index(),
        source="level_0",
        target="level_1",
        edge_attr=True,
        create_using=nx.DiGraph(),
    )

    nx.set_node_attributes(G, node_attrs.to_dict(orient="index"))
    G.graph.update(global_attrs.to_dict())
    return G


def nx_to_pandas(G: nx.DiGraph) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Convert a NetworkX DiGraph to pandas datatypes.

    Args:
        G: Input directed graph

    Returns:
        Tuple containing:
        - DataFrame of node attributes
        - DataFrame of edge attributes
        - Series of global graph attributes
    """
    edge_df = nx.to_pandas_edgelist(G).set_index(["source", "target"])
    edge_df.index.names = [None, None]
    node_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")

    return node_df, edge_df, pd.Series(G.graph)


def _split_branches(
    branches: list[list[int]], split_edges: list[tuple[int, int]]
) -> list[list[int]]:
    """Split branches at the given edges.

    Args:
        branches: List of branches, each represented as list of nodes.
        split_edges: List of edges between nodes where tracing is discontinous.

    Returns:
        An updated list of branches.
    """
    for p, n in split_edges:
        for i, branch in enumerate(branches):
            if n in branch:
                split_idx = branch.index(n)
                branches[i : i + 1] = [branch[:split_idx], branch[split_idx:]]
                break
    return branches


def _find_swc_tracing_interruptions(G: nx.DiGraph) -> list[tuple[int, int]]:
    """Identify discontinuities in the swc tracing order.

    Some swc files contain artefacts, where tracing of the same neurite was done
    in disconnected pieces. NEURON swc reader introduce a break in the trace at these
    points, since they parse the file in order. This leads to split branches, which
    should be one. This function identifies these points in the graph.

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
        G: NetworkX graph tracing of swc file.

    Returns:
        An array of edges where tracing is discontinous.
    """
    degree_is_2 = lambda n: G.out_degree(n) + G.in_degree(n) == 2

    interrupt_edges = []
    for n in G.nodes:
        if len(parents := list(G.predecessors(n))) > 0:
            parent = parents[0]
            if parent != n - 1 and degree_is_2(n) and degree_is_2(parent):
                interrupt_edges.append((parent, n))
    return interrupt_edges


def list_branches(
    swc_graph: nx.DiGraph,
    source: Optional[int] = None,
    return_branchpoints: bool = False,
    ignore_swc_tracing_interruptions=True,
    relevant_ids: List[int] = [1, 2, 3, 4],
    max_len: Optional[float] = None,
) -> Union[List[List[int]], Tuple[List[List[int]], Set[int], List[Tuple[int, int]]]]:
    """Get all uninterrupted paths in the traced morphology (i.e. branches).

    The graph is traversed depth-first starting from the first found leaf node.
    Nodes are considered to be part of a branch if they have only one parent and one
    child, which are both of the same type (i.e. have the same `id`). Nodes which are
    branchpoints or leafs are considered start / end points of a branch. A branchpoint
    can start multiple branches.

    Args:
        G: NetworkX graph tracing of swc file.
        source: The node from which to start tracing the graph. If None, the first leaf
            node is used.
        return_branchpoints: Whether to return the branchpoints and edges between them
            seperately.
        ignore_swc_tracing_interruptions: Whether to ignore discontinuities in the swc
            tracing order. If False, this will result in split branches at these points.
        relevant_ids: All type ids that are not in this list will be ignored for
            tracing the morphology. This means that branches which have multiple type
            ids (which are not in `relevant_ids`) will be considered as one branch.
            Defaults to `[1, 2, 3, 4]`.

    Returns:
        A list of linear paths in the graph. Each path is represented as list of nodes.
    """
    directed_graph = swc_graph.copy()
    undirected_graph = swc_graph.copy().to_undirected()
    branches = []
    branchpoints = set()
    visited = set()

    was_visited = lambda n1, n2: (n1, n2) in visited or (n2, n1) in visited
    id_of = lambda n: undirected_graph.nodes[n]["id"]
    is_soma = lambda n: id_of(n) == 1
    soma_nodes = lambda: [i for i, n in undirected_graph.nodes.items() if n["id"] == 1]

    def same_id(graph: nx.Graph, n1: int, n2: int) -> bool:
        has_id = lambda n: id_of(n) in relevant_ids if "id" in graph.nodes[n] else False
        return id_of(n1) == id_of(n2) if has_id(n1) and has_id(n2) else True

    def is_branchpoint_or_tip(n: int) -> bool:
        if undirected_graph.degree(n) == 2:
            i, j = undirected_graph.neighbors(n)
            # trace dir matters here! For segment [0,1,2,3] with node IDs: [1,1,2,2]
            # -> [[1,1], [1,2,2]] => node 1 is taken as branchpoint
            # <- [[2,2], [2,1,1]] => node 2 is taken as branchpoint
            return not same_id(undirected_graph, n, j)

        is_leaf = undirected_graph.degree(n) <= 1
        is_branching = undirected_graph.degree(n) > 2
        return is_leaf or is_branching

    def walk_path(start: int, succ: int) -> List[int]:
        """Walk from start to succ, recording new nodes until a branchpoint or tip is hit."""
        path = [start, succ]
        visited.add((start, succ))

        while undirected_graph.degree(succ) == 2:
            next_node = next(
                n for n in undirected_graph.neighbors(succ) if n != path[-2]
            )

            if was_visited(succ, next_node) or is_branchpoint_or_tip(succ):
                break

            path.append(next_node)
            visited.add((succ, next_node))
            succ = next_node

        return path

    leaf = next(n for n in undirected_graph.nodes if undirected_graph.degree(n) == 1)
    source = leaf if source is None else source
    single_soma = len(soma_nodes()) == 1
    for node in nx.dfs_tree(undirected_graph, source):
        if single_soma and is_soma(node):  # a single soma is its own branch
            branches.append([node])

        elif is_branchpoint_or_tip(node):
            branchpoints.add(node)
            for succ in undirected_graph.neighbors(node):
                if not was_visited(node, succ):
                    branches.append(walk_path(node, succ))

    # split branches (if tracing was interrupted or max_len is reached)
    if not ignore_swc_tracing_interruptions:  # TODO: fix!
        split_edges = _find_swc_tracing_interruptions(directed_graph)
        branches = _split_branches(branches, split_edges)
        branchpoints.update(set(p for (p, n) in split_edges))

    # TODO: add max_len
    if max_len is not None:
        raise NotImplementedError("max_len not implemented")

    if return_branchpoints:
        branchpoint_edges = sum(
            [list(undirected_graph.edges(n)) for n in branchpoints], []
        )
        return branches, branchpoints, branchpoint_edges
    return branches


def _add_missing_swc_attrs(G) -> nx.DiGraph:
    """Add missing swc attributes to a SWC graph.

    Allows to specify morphology from just edges.

    Args:
        G: The SWC graph to add missing attributes to.

    Returns:
        The SWC graph with missing attributes set to their defaults.
    """
    defaults = {"id": 0, "r": 1}

    available_keys = G.nodes[next(iter(G.nodes()))].keys()
    # TODO: ADD BACK IN
    # xyz = compute_xyz(G) if "x" not in available_keys else {}
    # for n, (x, y, z) in xyz.items():
    #     # xyz is needed to compute compartment lengths
    #     G.nodes[n]["x"] = x
    #     G.nodes[n]["y"] = y
    #     G.nodes[n]["z"] = z

    for key in set(defaults.keys()).difference(available_keys):
        nx.set_node_attributes(G, defaults[key], key)
    return G


def build_compartment_graph(
    swc_graph: nx.DiGraph,
    ncomp: int = 1,
    root: Optional[int] = None,  # TODO: change to source ?
    min_radius: Optional[float] = None,
    max_len: Optional[float] = None,
    ignore_swc_tracing_interruptions: bool = True,
    relevant_type_ids: List[int] = [1, 2, 3, 4],
) -> nx.DiGraph:
    """Return a networkX graph that indicates the compartment structure.

    Build a new graph made up of compartments in every branch. These compartments are
    spaced at equidistant points along the branch. Node attributes, like radius are
    linearly interpolated along its length.

    Example: 4 compartments | edges = - | nodes = o | comp_nodes = x
    o-----------o----------o---o---o---o--------o
    o-------x---o----x-----o--xo---o---ox-------o

    This function returns a nx:DiGraph. The graph is directed only because every
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
        Graph of the compartmentalized morphology.

    Example usage
    ^^^^^^^^^^^^^

    ::

        from jaxley.io.graph swc_to_nx
        swc_graph = swc_to_nx("path_to_swc.swc")
        comp_graph = build_compartment_graph(swc_graph, ncomp=1)
    """
    G = _add_missing_swc_attrs(swc_graph)

    branches = list_branches(
        G,
        source=root,
        ignore_swc_tracing_interruptions=ignore_swc_tracing_interruptions,
        max_len=max_len,
        relevant_ids=relevant_type_ids,
    )
    nodes_df = nx_to_pandas(G)[0].astype(float)

    # create new set of indices which arent already used as node indices to label comps
    existing_inds = set(nodes_df.index)
    num_additional_inds = len(branches) * ncomp
    proposed_inds = set(range(num_additional_inds + len(existing_inds)))
    proposed_comp_inds = list(
        proposed_inds - existing_inds
    )  # avoid overlap w node inds

    # identify tip nodes (degree == 1 -> node appears only once in edges)
    nodes_in_edges, node_counts_in_edges = np.unique(G.edges, return_counts=True)
    tip_node_inds = nodes_in_edges[node_counts_in_edges == 1]

    # collect comps and comp_edges
    branch_nodes, branch_edges = [], []
    xyzr = []
    for i, branch in enumerate(branches):
        node_attrs = nodes_df.loc[branch]

        compute_edge_lens = lambda x: (x.diff(axis=0).fillna(0) ** 2).sum(axis=1) ** 0.5
        edge_lens = compute_edge_lens(node_attrs[["x", "y", "z"]])
        node_attrs["l"] = edge_lens.cumsum()  # path length

        # For single-point somatata, we set l = 2*r this ensures
        # A_cylinder = 2*pi*r*l = 4*pi*r^2 = A_sphere.
        if single_soma := (len(branch) == 1):
            node_attrs = node_attrs.loc[branch * 2]  # duplicate soma node
            radius = node_attrs["r"].iloc[0]
            node_attrs["l"] = np.array([0, 2 * radius])

        branch_id = node_attrs["id"].iloc[
            1
        ]  # node after branchpoint (0) det. branch id
        branch_len = max(node_attrs["l"])
        comp_len = branch_len / ncomp
        comp_locs = list(np.linspace(comp_len / 2, branch_len - comp_len / 2, ncomp))

        # Create node indices and attributes for branch-tips/branchpoints and comps
        # branch_inds, branchpoint, comp_id, comp_len, x, y, z, r
        branch_tips = branch[0], branch[-1]
        branch_tip_attrs = [
            [i, True, node_attrs["id"].iloc[0], 0],
            [i, True, node_attrs["id"].iloc[-1], 0],
        ]

        comp_inds = proposed_comp_inds[i * ncomp : (i + 1) * ncomp]
        comp_inds = np.array([branch_tips[0], *comp_inds, branch_tips[1]])

        comp_attrs = [[i, False, branch_id, comp_len]] * ncomp
        comp_attrs = [branch_tip_attrs[0]] + comp_attrs + [branch_tip_attrs[1]]
        comp_attrs = np.hstack([comp_inds[:, None], comp_attrs])

        # Interpolate xyzr along branch
        x = np.array([0] + comp_locs + [branch_len])
        xp = np.array(node_attrs["l"].values)
        fp = np.array(node_attrs[["x", "y", "z", "r"]].values)

        # TODO: average **r** instead of interpolating!
        interpolated_coords = np.column_stack(
            [np.interp(x, xp, fp[:, i]) for i in range(fp.shape[1])]
        )

        # Combine interpolated coordinates with existing attributes
        comp_attrs = np.hstack([comp_attrs, interpolated_coords])

        # remove tip nodes
        comp_attrs = comp_attrs[1:] if branch_tips[0] in tip_node_inds else comp_attrs
        comp_attrs = comp_attrs[:-1] if branch_tips[1] in tip_node_inds else comp_attrs

        # single soma branches lead to self looping edges, since branch[0] == branch[-1]
        # we therefore remove one tip node / branchpoint node, i.e. [0,s,0] -> [s,0]
        comp_attrs = comp_attrs[1:] if single_soma else comp_attrs

        # Store edges, nodes, and xyzr in branch-wise manner
        intra_branch_edges = np.stack([comp_attrs[:-1, 0], comp_attrs[1:, 0]]).T
        branch_edges.append(intra_branch_edges.astype(int).tolist())
        branch_nodes.append(comp_attrs)
        xyzr.append(node_attrs[["x", "y", "z", "r"]].values)

    branch_nodes = np.concatenate(branch_nodes)
    comp_attrs_keys = ["idx", "branch", "branchpoint", "id", "l", "x", "y", "z", "r"]
    comp_df = pd.DataFrame(branch_nodes, columns=comp_attrs_keys)

    int_cols = ["idx", "branch", "id"]
    comp_df[int_cols] = comp_df[int_cols].astype(int)

    bool_cols = ["branchpoint"]
    comp_df[bool_cols] = comp_df[bool_cols].astype(bool)

    # threshold radius
    if min_radius is not None:
        comp_df["r"] = np.maximum(comp_df["r"], min_radius)

    # drop duplicated branchpoint nodes
    comp_df = comp_df.drop_duplicates(subset=["idx", "branchpoint"])
    comp_df = comp_df.set_index("idx")

    # create comp edges
    comp_edges = sum(branch_edges, [])
    comp_edges_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(comp_edges))
    comp_edges_df["synapse"] = False  # edges between compartments that are synapses
    comp_edges_df["comp_edge"] = True  # edges between connected compartments

    global_attrs = pd.Series({"xyzr": xyzr})
    G = pandas_to_nx(comp_df, comp_edges_df, global_attrs)
    G = nx.relabel_nodes(G, {n: i for i, n in enumerate(G.nodes)})
    return G


########################################################################################
################################ BUILD SOLVE GRAPH #####################################
########################################################################################


def _add_jaxley_meta_data(G: nx.DiGraph) -> nx.DiGraph:
    """Add attributes to and rename existing attributes of the compartalized morphology.

    Makes the imported and compartmentalized morphology compatible with jaxley.
    """
    nodes_df, edge_df, global_attrs = nx_to_pandas(G)
    module_global_attrs = pd.Series({"channels": {}, "synapses": {}, "group_names": []})
    global_attrs = pd.concat([global_attrs, module_global_attrs])

    # Description of SWC file format:
    # http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
    group_ids = {0: "undefined", 1: "soma", 2: "axon", 3: "basal", 4: "apical"}

    # rename/reformat existing columns
    for group_id, group_name in group_ids.items():
        where_group = nodes_df["id"] == group_id
        if where_group.any():
            global_attrs["group_names"].append(group_name)
            nodes_df[group_name] = where_group
    nodes_df = nodes_df.drop(columns=["id"])
    module_col_names = {"r": "radius", "l": "length", "branch": "branch_index"}
    nodes_df = nodes_df.rename(module_col_names, axis=1)

    # new columns
    nodes_df["capacitance"] = 1.0
    nodes_df["v"] = -70.0
    nodes_df["axial_resistivity"] = 5000.0
    # TODO: rename cell_index > cell, comp_index > comp, branch_index > branch
    nodes_df["comp_index"] = np.arange(len(nodes_df))
    nodes_df["cell_index"] = 0

    return pandas_to_nx(nodes_df, edge_df, global_attrs)


def _replace_branchpoints_with_edges(G: nx.DiGraph, source=None) -> nx.DiGraph:
    """Replace branchpoint nodes with edges connecting parent and children.

    Also does a depth-first traversal of the graph and reorders the edges."""
    # reorder graph depth-first TODO: Which order to choose?
    leaf = next(n for n in G.nodes if G.degree(n) == 1)
    source = leaf if source is None else source
    for u, v in nx.dfs_edges(G.to_undirected(), source):
        if (u, v) not in G.edges and (v, u) in G.edges:  # flip edge direction
            G.add_edge(u, v, **G.get_edge_data(v, u))
            G.remove_edge(v, u)

    G.add_edges_from([(i, j, {"branch_edge": False}) for i, j in G.edges])
    branch_edge_attrs = {"comp_edge": True, "synapse": False, "branch_edge": True}

    branchpoints = [n for n in G.nodes if G.nodes[n]["branchpoint"]]
    for n in branchpoints:
        parents = list(G.predecessors(n))
        children = list(G.successors(n))

        # remove branchpoint and connect parent to children
        G.remove_node(n)
        if len(parents) == 0 and len(children) == 2:  # linear branchpoint
            G.add_edge(*children, **branch_edge_attrs)
        else:
            G.add_edges_from([(parents[0], c) for c in children], **branch_edge_attrs)

    for i, n in enumerate(G.nodes):
        G.nodes[n].pop("branchpoint")
        # TODO: Can we skip relabeling comps and reindexing the dataframe?
        G.nodes[n]["comp_index"] = i

    G = nx.relabel_nodes(G, {n: i for i, n in enumerate(G.nodes)})
    return G


# TODO: Remove this along with branch_edges attr in nodes in favour of comp_edges
def _compute_branch_parents(
    node_df: pd.DataFrame, edge_df: pd.DataFrame
) -> List[List[int]]:
    """Compute the parent structure of the branch graph (for each cell).

    Args:
        node_df: The node dataframe of the graph.
        edge_df: The edge dataframe of the graph.

    Returns:
        The parent structure of the branch graph for each cell.
    """
    branch_edge_inds = edge_df.index[edge_df["branch_edge"]]
    parent_inds = branch_edge_inds.get_level_values(0)
    child_inds = branch_edge_inds.get_level_values(1)

    branch_edges = pd.DataFrame(
        {
            "parent_branch": node_df["branch_index"].loc[parent_inds].values,
            "child_branch": node_df["branch_index"].loc[child_inds].values,
        }
    )

    acc_parents = []
    parent_branch_inds = branch_edges.set_index("child_branch").sort_index()[
        "parent_branch"
    ]

    for branch_inds in node_df.groupby("cell_index")["branch_index"].unique():
        root_branch_idx = branch_inds[0]
        parents = parent_branch_inds.loc[branch_inds[1:]] - root_branch_idx
        acc_parents.append([-1] + parents.tolist())
    return acc_parents


########################################################################################
################################ MODULE FROM GRAPH #####################################
########################################################################################


def _build_module(G: nx.DiGraph) -> Module:
    """Build a Module from a compartmentalized morphology.

    This function builds a Module from a nx.DiGraph that has been compartmentalized.

    Args:
        G: The graph to build the Module from.

    Returns:
        The Module built from the graph.
    """
    node_df, edge_df, global_attrs = nx_to_pandas(G)

    nodes_per_branch = node_df["branch_index"].value_counts()
    assert (
        nodes_per_branch.nunique() == 1
    ), "`from_graph()` does not support a varying number of compartments in each branch."

    acc_parents = _compute_branch_parents(node_df, edge_df)
    module = _build_module_scaffold(
        node_df, parent_branches=acc_parents, xyzr=global_attrs["xyzr"]
    )

    node_df.columns = [
        "global_" + col if "local" not in col and "index" in col else col
        for col in node_df.columns
    ]

    synapse_edges = edge_df[edge_df.synapse]

    # set column-wise. preserves cols not in df.
    module.nodes[node_df.columns] = node_df
    module.edges = synapse_edges if not synapse_edges.empty else module.edges

    # add all the extra attrs
    module.synapses = global_attrs["synapses"]
    module.channels = global_attrs["channels"]
    module.group_names = global_attrs["group_names"]
    module.membrane_current_names = [c.current_name for c in module.channels]
    module.synapse_names = [s._name for s in module.synapses]

    return module


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

    comp_graph = _add_jaxley_meta_data(comp_graph)
    solve_graph = _replace_branchpoints_with_edges(comp_graph, source=solve_root)
    module = _build_module(solve_graph)
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
