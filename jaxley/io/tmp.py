# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, Dict, List, Optional, Set, Tuple, Union
from warnings import warn

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from jaxley.modules import Branch, Cell, Compartment, Module, Network
from jaxley.utils.cell_utils import compute_cone_props

#########################################################################################
################################### Helper functions ####################################
#########################################################################################


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
        edge_attr=True if edge_attrs.columns.size > 0 else None,
        create_using=nx.DiGraph(),
    )
    G.add_nodes_from((n, d) for n, d in node_attrs.to_dict(orient="index").items())
    G.graph.update(global_attrs.to_dict())
    return G


def nx_to_pandas(
    G: nx.DiGraph, sort_index: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Convert a NetworkX DiGraph to pandas datatypes.

    Args:
        G: Input directed graph
        sort_index: Whether to sort the index of the DataFrames.

    Returns:
        Tuple containing:
        - DataFrame of node attributes
        - DataFrame of edge attributes
        - Series of global graph attributes
    """
    edge_df = nx.to_pandas_edgelist(G).set_index(["source", "target"])
    edge_df.index.names = [None, None]
    node_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
    node_df = node_df.sort_index() if sort_index else node_df
    edge_df = edge_df.sort_index() if sort_index else edge_df

    return node_df, edge_df, pd.Series(G.graph)


def compute_xyz(
    G: nx.DiGraph,
    length: float = 1.0,
    spread: float = np.pi / 8,
    spread_decay: float = 0.9,
    twist: float = 0.0,
    xy_only: bool = True,
) -> Dict[int, tuple[float, float, float]]:
    """Compute xyz coordinates for a tree-like appearance of a networkX graph in 2D or 3D.

    Handles branches implicitly since nodes in a branch have 1 child.

    Args:
        G: The Graph to compute node xyz coordinates for.
        length: The length of each edge.
        spread: The opening angle at which the edges spread out.
        spread_decay: Multiplicative decay factor for the opening angle / spread.
        twist: Add additional twisting. Means fewer overlapping nodes in 3D projections.
        xy_only: Whether to only compute the xy coordinates and fix the z-coordinate.

    Returns:
        A dictionary mapping node indices to xyz coordinates.
    """
    # TODO: Replace compute_xyz with this or vice versa, redundant!
    root = next(n for n, d in G.in_degree() if d == 0)
    pos = {root: (0.0, 0.0, 0.0)}

    def recurse(node, depth=1, theta=0.0, phi=np.pi / 2):
        children = [n for n in G.successors(node) if n not in pos]
        if not children:
            return
        n = len(children)
        curr_spread = spread * (spread_decay ** (depth - 1))
        x0, y0, z0 = pos[node]
        phi = np.pi / 2 if xy_only else phi
        base_theta = theta + depth * twist
        if n == 1:
            thetas, phis = [base_theta], [phi]
        else:
            if xy_only:
                thetas = np.linspace(
                    base_theta - curr_spread / 2, base_theta + curr_spread / 2, n
                )
                phis = [phi] * n
            else:
                thetas = np.linspace(
                    base_theta, base_theta + 2 * np.pi, n, endpoint=False
                )
                phis = [phi - curr_spread] * n
        for th, ph, child in zip(thetas, phis, children):
            x = x0 + length * np.sin(ph) * np.cos(th)
            y = y0 + length * np.sin(ph) * np.sin(th)
            z = z0 + length * np.cos(ph) * (not xy_only)
            pos[child] = (x, y, z)
            recurse(child, depth + 1, th, ph)

    recurse(root, theta=0.0, phi=np.pi / 2)
    return pos


########################################################################################
################################### BUILD SWC GRAPH ####################################
########################################################################################


def swc_to_nx(
    fname: str, num_lines: Optional[int] = None, relevant_ids: List[int] = [1, 2, 3, 4]
) -> nx.DiGraph:
    # def swc_to_nx(fname: str, num_lines: Optional[int] = None) -> nx.DiGraph:
    """Read a SWC morphology file into a NetworkX DiGraph.

    The graph is read such that each entry in the swc file becomes a graph node
    with the column attributes (id, x, y, z, r). Then each node is connected to its
    designated parent via an edge.

    Args:
        fname: Path to the SWC file
        num_lines: Number of lines to read from the file. If None, all lines are read.
        relevant_ids: List of ids to include in the graph. Defaults to [1, 2, 3, 4].
            All other ids are set to 0.

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
    fmt_id = lambda id: int(id) if id in relevant_ids else 0
    for i, id, x, y, z, r, p in i_id_xyzr_p.tolist():  # tolist: np.float64 -> float
        graph.add_node(int(i), **{"id": fmt_id(id), "x": x, "y": y, "z": z, "r": r})
        if p != -1:
            graph.add_edge(int(p), int(i))
    return graph


########################################################################################
############################## BUILD COMPARTMENT GRAPH #################################
########################################################################################


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
                parent_end = branch.index(p) + 1
                child_start = branch.index(n) - 1
                branches[i : i + 1] = [branch[:parent_end], branch[child_start:]]
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

    def not_branching(n):
        nn_ids = [G.nodes[n]["id"] for n in nx.all_neighbors(G, n)]
        return len(nn_ids) == 2 and all([id == nn_ids[0] for id in nn_ids[1:]])

    interrupt_edges = []
    for n in G.nodes:
        if len(parents := list(G.predecessors(n))) > 0:
            parent = parents[0]
            if parent != n - 1 and not_branching(n) and not_branching(parent):
                interrupt_edges.append((parent, n))
    return interrupt_edges


def _compute_long_branch_splits(
    G: nx.DiGraph, branches: list[list[int]], max_len: float
) -> list[list[int]]:
    """Find branches that are too long and split them at equidistant points.

    If branch >= 1*max_len, then we split it down the middle. If branch >= 2*max_len,
    then we split it into 3 parts. And so on. This ensures that sub-branches have the
    same length & length <= max_len.

    Args:
        G: NetworkX graph tracing of swc file.
        branches: List of branches, each represented as list of nodes.
        max_len: Maximum length any branch cannot exceed.

    Returns:
        Edges where to split branches, to ensure that no branch exceeds max_len and such
        that the resulting sub-branches are of similar length.
    """
    split_edges = []
    node_attrs = lambda n, keys: [G.nodes[n][key] for key in keys]
    branches_wo_single_soma = (branch for branch in branches if len(branch) > 2)
    for branch in branches_wo_single_soma:
        branch_xyz = np.array([node_attrs(i, ["x", "y", "z"]) for i in branch])

        # split at equidistant points along the branch if branch_len > max_len
        # this ensure that tracing order does not matter!
        path_lens = np.linalg.norm(branch_xyz[1:] - branch_xyz[:-1], axis=1).cumsum()
        num_splits = int(path_lens.max() // max_len)
        branch_ends = np.linspace(0, path_lens.max(), num_splits + 2)

        comp_in_branch = branch_ends[1:-1, None] >= path_lens
        break_at = [(branch[i - 1], branch[i]) for i in comp_in_branch.sum(axis=1)]
        split_edges += break_at
    return split_edges


def list_branches(
    swc_graph: nx.DiGraph,
    source: Optional[int] = None,
    return_branchpoints: bool = False,
    ignore_swc_tracing_interruptions=True,
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

    Returns:
        A list of linear paths in the graph. Each path is represented as list of nodes.
    """
    dir_graph = swc_graph.copy()
    undir_graph = swc_graph.copy().to_undirected()
    branches = []
    branchpoints = set()
    visited = set()

    was_visited = lambda n1, n2: (n1, n2) in visited or (n2, n1) in visited
    id_of = lambda n: undir_graph.nodes[n]["id"]
    is_soma = lambda n: id_of(n) == 1
    soma_nodes = lambda: [i for i, n in undir_graph.nodes.items() if n["id"] == 1]

    def same_id(graph: nx.Graph, n1: int, n2: int) -> bool:
        has_id = lambda n: id_of(n) != 0 if "id" in graph.nodes[n] else False
        return id_of(n1) == id_of(n2) if has_id(n1) and has_id(n2) else True

    def is_branchpoint_or_tip(n: int) -> bool:
        if undir_graph.degree(n) == 2:
            i, j = undir_graph.neighbors(n)
            # trace dir matters here! For segment [0,1,2,3] with node IDs: [1,1,2,2]
            # -> [[1,1], [1,2,2]] => node 1 is taken as branchpoint
            # <- [[2,2], [2,1,1]] => node 2 is taken as branchpoint
            return not same_id(undir_graph, n, j)

        is_leaf = undir_graph.degree(n) <= 1
        is_branching = undir_graph.degree(n) > 2
        return is_leaf or is_branching

    def walk_path(start: int, succ: int) -> List[int]:
        """Walk from start to succ, recording new nodes until a branchpoint or tip is hit."""
        path = [start, succ]
        visited.add((start, succ))

        while undir_graph.degree(succ) == 2:
            next_node = next(n for n in undir_graph.neighbors(succ) if n != path[-2])

            if was_visited(succ, next_node) or is_branchpoint_or_tip(succ):
                break

            path.append(next_node)
            visited.add((succ, next_node))
            succ = next_node

        return path

    # traverse graph and collect branches
    leaf = next(n for n in undir_graph.nodes if undir_graph.degree(n) == 1)
    source = leaf if source is None else source

    # start recording from first branchpoint or tip since starting from non-branchpoint
    # will split the branch in two. I.e. starting at (2) will split [0,1,2,3] into two
    # separate branches: [0,1,2] and [2,3].
    if not is_branchpoint_or_tip(source):
        source = next(
            n for n in nx.dfs_tree(undir_graph, source) if is_branchpoint_or_tip(n)
        )

    single_soma = len(soma_nodes()) == 1
    for node in nx.dfs_tree(undir_graph, source):
        if single_soma and is_soma(node):  # a single soma is its own branch
            branches.append([node])

        elif is_branchpoint_or_tip(node):
            branchpoints.add(node)
            for succ in undir_graph.neighbors(node):
                if not was_visited(node, succ):
                    branches.append(walk_path(node, succ))

    # split branches (if tracing was interrupted or max_len is reached)
    if not ignore_swc_tracing_interruptions:  # TODO: fix!
        interupted_edges = _find_swc_tracing_interruptions(dir_graph)
        branches = _split_branches(branches, interupted_edges)
        branchpoints.update(set(p for (p, n) in interupted_edges))

    # max_len splitting only after accounting for interrupted branches
    if max_len is not None:
        split_edges = _compute_long_branch_splits(dir_graph, branches, max_len)
        branches = _split_branches(branches, split_edges)
        branchpoints.update(set(p for (p, n) in split_edges))

    if return_branchpoints:
        branchpoint_edges = sum([list(undir_graph.edges(n)) for n in branchpoints], [])
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
    xyz = compute_xyz(G) if "x" not in available_keys else {}
    for n, (x, y, z) in xyz.items():
        # xyz is needed to compute compartment lengths
        G.nodes[n]["x"] = x
        G.nodes[n]["y"] = y
        G.nodes[n]["z"] = z

    for key in set(defaults.keys()).difference(available_keys):
        nx.set_node_attributes(G, defaults[key], key)
    return G


def compartmentalize_branch(
    branch_nodes: pd.DataFrame, ncomp: int, ignore_branchpoints: bool = False
) -> pd.DataFrame:
    """Interpolate or integrate node attributes along branch.

    Takes a dataframe with nodes (index) and node attributes (columns) and returns a
    dataframe of compartments and compartment attributes. Compartments are spaced at
    equidistant points along the branch. Node attributes, like radius are linearly
    interpolated along its length.

    Example: 4 compartments | edges = - | nodes = o | comp_nodes = x
    o-----------o----------o---o---o---o--------o
    o-------x---o----x-----o--xo---o---ox-------o

    Args:
        branch_nodes: DataFrame of node attributes for nodes in a branch.
            needs to include morph attributes `id`, `x`, `y`, `z`, `r`.
        ncomp: Number of compartments per branch.
        ignore_branchpoints: Whether to consider the branchpoint part of the neurite or
        not if it has a different id. This is for example relevant if the branch extends
        from a single point soma or somatic branchpoint. In these cases, the somatic SWC
        node is _not_ considered to be part of the dendrite.

    Returns:
        DataFrame of compartments and compartment attributes.
    """
    # TODO: This should be reusable for `set_ncomp()`

    # all nodes in a branch must have the same id. since branchpoints can have a
    # different id (attached at the ends), the node after the branchpoint det. branch id
    branch_id = branch_nodes["id"].iloc[1 if len(branch_nodes) > 1 else 0]
    not_branch_id = (branch_nodes["id"] != branch_id).values
    branch_nodes.loc[not_branch_id, "id"] = branch_id

    # if branchpoint has a different id, its radius is assumed to be equal to that
    # of the neighbouring node.
    if not_branch_id[0] and len(branch_nodes) > 2:
        branch_nodes.loc[branch_nodes.index[0], "r"] = branch_nodes["r"].values[1]
    if not_branch_id[-1] and len(branch_nodes) > 2:
        branch_nodes.loc[branch_nodes.index[-1], "r"] = branch_nodes["r"].values[-2]

    # set edge lengths to 0 for branchpoints of different id if ignore_branchpoints
    edge_lens = (branch_nodes[["x", "y", "z"]].diff(axis=0) ** 2).sum(axis=1) ** 0.5
    if ignore_branchpoints and len(branch_nodes) > 2:
        edge_lens.loc[not_branch_id] = 0  # ignore branchpoints

    branch_nodes["l"] = edge_lens.fillna(0).cumsum()  # path length

    # handle single point branches / somata
    node_inds_in_branch = branch_nodes.index.tolist()
    if len(node_inds_in_branch) == 1:
        # duplicate node to compartmentalize it along its "length", i.e. l = 2*r
        branch_nodes = branch_nodes.loc[node_inds_in_branch * 2]
        # Setting l = 2*r ensures A_cylinder = 2*pi*r*l = 4*pi*r^2 = A_sphere.
        branch_nodes["l"] = np.array([0, 2 * branch_nodes["r"].iloc[0]])

    ls = branch_nodes["l"].values
    rs = branch_nodes["r"].values
    xyz = branch_nodes[["x", "y", "z"]].values
    branch_len = max(ls)

    if branch_len < 1e-8:
        warn(
            "Found a branch with length 0. To avoid NaN while integrating the "
            "ODE, we capped this length to 0.1 um. The underlying cause for the "
            "branch with length 0 is likely a strange SWC file. The "
            "most common reason for this is that the SWC contains a soma "
            "traced by a single point, and a dendrite that connects to the soma "
            "has no further child nodes."
        )
    comp_len = branch_len / ncomp

    # Create node indices and attributes for branch-tips/branchpoints and comps
    # is_comp, comp_len, comp_id, x, y, z, r, area, volume, res_in, res_out
    cone_prop_cols = ["r", "area", "volume", "resistive_load_in", "resistive_load_out"]
    cols = ["l", "x", "y", "z"] + cone_prop_cols
    comp_attrs = pd.DataFrame(np.full((ncomp + 2, len(cols)), np.nan), columns=cols)

    comp_attrs["id"] = np.array([0, *[branch_id] * ncomp, 0], dtype=int)
    is_comp = np.array([False, *[True] * ncomp, False], dtype=bool)
    comp_attrs.loc[is_comp, "l"] = comp_len

    tip_cols = ["id", "x", "y", "z", "r"]
    comp_attrs.loc[~is_comp, tip_cols] = branch_nodes[tip_cols].iloc[[0, -1]].values

    # Interpolate xyz along branch
    comp_centers = np.linspace(comp_len / 2, branch_len - comp_len / 2, ncomp)
    comp_centers = np.array([0, *comp_centers, branch_len])
    x_at_centers = lambda x: np.interp(comp_centers, ls, x)
    comp_attrs[["x", "y", "z"]] = np.apply_along_axis(x_at_centers, axis=0, arr=xyz)

    comp_ends = np.linspace(0, branch_len, ncomp + 1)
    comp_tips = np.stack([comp_ends[:-1], comp_ends[1:]], axis=1)

    # compute radius, area, volume, res_in, res_out
    cone_props = np.array([compute_cone_props(ls, rs, x1, x2) for x1, x2 in comp_tips])
    comp_attrs.loc[is_comp, cone_prop_cols] = cone_props

    return comp_attrs


def propose_new_inds(existing_inds: List[int], num_additional_inds: int) -> List[int]:
    """Propose new set of indices that does not overlap with existing indices.

    Args:
        existing_inds: Existing indices.
        num_additional_inds: Number of additional indices to propose.

    Returns:
        List of proposed indices.
    """
    existing_inds = set(existing_inds)
    all_new_inds = set(range(num_additional_inds + len(existing_inds)))
    proposed_inds = list(all_new_inds - existing_inds)  # avoid node inds overlap
    return proposed_inds


def build_compartment_graph(
    swc_graph: nx.DiGraph,
    ncomp: int = 1,
    root: Optional[int] = None,  # TODO: change to source ?
    min_radius: Optional[float] = None,
    max_len: Optional[float] = None,
    ignore_swc_tracing_interruptions: bool = True,
) -> nx.DiGraph:
    """Return a networkX graph that indicates the compartment structure.

    Build a new graph made up of compartments in every branch. These compartments are
    spaced at equidistant points along the branch. Node attributes, like radius are
    linearly interpolated along its length.

    Example: 4 compartments | edges = - | nodes = o | comp_nodes = x
    o-----------o----------o---o---o---o--------o
    o-------x---o----x-----o--xo---o---ox-------o

    This function returns a nx.DiGraph. The graph is directed only because every
    compartment tracks the xyzr coordinates of the associated SWC file. These xyzr
    coordinates are ordered by the order of the traversal of the swc_graph. In later
    methods (e.g. build_solve_graph), we traverse the `comp_graph` and mostly ignore
    the directionality of the edges, but we only use the directionality to reverse the
    xyzr coordinates if necessary.

    Args:
        swc_graph: Graph generated by `swc_to_nx()`.
        ncomp: How many compartments per branch to insert.
        root: The root branch from which to start tracing the nodes. This defines the
            branch indices.
        min_radius: Minimal radius for each compartment.
        max_len: Maximal length for each branch. Longer branches are split into
            separate branches.
        ignore_swc_tracing_interruptions: If `False`, it this function automatically
            starts a new branch when a section is traced with interruptions.

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

    # # determine consistent traversal direction along graph
    # root = next(n for n in G.nodes if G.degree(n) == 1) if root is None else root
    # for e in nx.dfs_edges(G.to_undirected(), root):
    #     if e not in G.edges:
    #         G.add_edge(*e, **G.edges[e[::-1]])
    #         G.remove_edge(*e[::-1])

    branches, branchpoints, _ = list_branches(
        G,
        source=root,
        ignore_swc_tracing_interruptions=ignore_swc_tracing_interruptions,
        max_len=max_len,
        return_branchpoints=True,
    )

    nodes_df = nx_to_pandas(G)[0].astype(float)

    # threshold radius
    if min_radius is None:
        assert (
            nodes_df["r"] > 0.0
        ).all(), "Radius 0.0 in SWC file. Set `read_swc(..., min_radius=...)`."
    else:
        nodes_df["r"] = np.maximum(nodes_df["r"], min_radius)

    # identify somatic branchpoints. A somatic branchpoint is a branchpoint at which at
    # least two connecting branches are somatic. In that case (and in the case of a
    # single-point soma), non-somatic branches are assumed to start from their first
    # traced point, not from the soma.
    soma_nodes = [n for n in G.nodes if G.nodes[n]["id"] == 1]
    single_soma = len(soma_nodes) == 1
    is_soma_branchpoint = (
        lambda n: len([n for n in G.neighbors(n) if G.nodes[n]["id"] == 1]) >= 2
    )
    soma_branchpoints = [n for n in soma_nodes if is_soma_branchpoint(n) or single_soma]

    # create new set of indices which arent already used as node indices to label comps
    existing_inds = nodes_df.index
    num_additional_inds = len(branches) * ncomp
    proposed_node_inds = propose_new_inds(existing_inds, num_additional_inds)

    # collect comps and comp_edges
    comps, comp_edges, xyzr = [], [], []
    for branch_idx, branch in enumerate(branches):
        # ensure node_index increases monotonically along branch. Required for branch.loc()
        # branch = branch[::-1] if branch[0] < branch[-1] else branch # TODO: Check if this is still needed

        branch_nodes = nodes_df.loc[branch]

        # Compute the compartmentalization of the branch.
        has_somatic_branch_pt = np.any(np.isin(branch, soma_branchpoints))
        comp_attrs = compartmentalize_branch(branch_nodes, ncomp, has_somatic_branch_pt)

        # Attach branchpoint and tip nodes to the branch.
        # Since branchpoints / tips have the same node_index as in the original graph
        # there is no need to keep track of branch connectivity.
        comp_inds = proposed_node_inds[branch_idx * ncomp : (branch_idx + 1) * ncomp]
        comp_attrs["node"] = np.array([branch[0], *comp_inds, branch[-1]], dtype=int)
        comp_attrs["branch_index"] = [np.nan, *[branch_idx] * ncomp, np.nan]
        comp_attrs["branch_index"] = comp_attrs["branch_index"].astype(
            pd.Int64Dtype()
        )  # incl. NaN for integers

        # single soma branches lead to self looping edges, since branch[0] == branch[-1]
        # we therefore remove one tip node / branchpoint node, i.e. [0,s,0] -> [s,0]
        comp_attrs = comp_attrs.iloc[1:] if branch[0] == branch[-1] else comp_attrs

        # Store edges, nodes, and xyzr in branch-wise manner
        intra_branch_edges = np.stack(
            [comp_attrs["node"].iloc[:-1], comp_attrs["node"].iloc[1:]]
        ).T
        comp_edges.append(intra_branch_edges.astype(int).tolist())
        comps.append(comp_attrs)

        # store xyzr for each node in branch
        xyzr.append(branch_nodes[["x", "y", "z", "r"]].values)

    comp_df = pd.concat(comps)

    # drop duplicated branchpoint nodes and fill with original attrs of branchpoint node
    comp_df = comp_df.drop_duplicates(subset=["node"])
    comp_df = comp_df.set_index("node")
    xyzr_cols = ["x", "y", "z", "r"]
    is_comp = comp_df["branch_index"].notna()
    at_branchpoints = comp_df.loc[~is_comp].index
    comp_df.loc[at_branchpoints, xyzr_cols] = nodes_df.loc[at_branchpoints, xyzr_cols]

    # create comp edges
    comp_edges = sum(comp_edges, [])
    comp_edges_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(comp_edges))
    comp_edges_df["synapse"] = False  # edges between compartments that are synapses
    comp_edges_df["comp_edge"] = True  # edges between connected compartments

    global_attrs = pd.Series({"xyzr": xyzr})
    G = pandas_to_nx(comp_df, comp_edges_df, global_attrs)

    # relabel nodes to [0, 1, ..., ncomp-1] -> [ncomp, ncomp+1, ..., nbranchpoints-1]
    # this way the branchpoints can be appended to the end of nodes
    branch_inds = nx.get_node_attributes(G, "branch_index")
    is_comp = [n for n, c in branch_inds.items() if not pd.isna(c)]
    not_comp = [n for n, c in branch_inds.items() if pd.isna(c)]
    comp_labels = {n: i for i, n in enumerate(is_comp)}
    branchpoint_labels = {n: i + len(comp_labels) for i, n in enumerate(not_comp)}
    G = nx.relabel_nodes(G, {**comp_labels, **branchpoint_labels})
    return G


########################################################################################
################################ BUILD SOLVE GRAPH #####################################
########################################################################################


def _add_jaxley_meta_data(G: nx.DiGraph) -> nx.DiGraph:
    """Add attributes to and rename existing attributes of the compartalized morphology.

    Makes the imported and compartmentalized morphology compatible with jaxley.
    """
    nodes_df, edge_df, global_attrs = nx_to_pandas(G)
    # TODO: make channels and synapses dicts?
    module_global_attrs = pd.Series({"channels": [], "synapses": [], "group_names": []})
    module_global_attrs["branchpoints_and_tips"] = pd.DataFrame([])
    global_attrs = pd.concat([global_attrs, module_global_attrs])

    # Description of SWC file format:
    # http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
    ids = nodes_df["id"].unique()
    group_names = {0: "undefined", 1: "soma", 2: "axon", 3: "basal", 4: "apical"}
    group_names.update({i: f"custom{i}" for i in ids if i not in group_names})

    # rename/reformat existing columns (incl. one-hot groups)
    one_hot_ids = pd.get_dummies(nodes_df.pop("id"))
    groups = one_hot_ids.rename(columns=group_names)
    # ignore undefined ids. If errors="ignore" -> only remove if present
    groups = groups.drop("undefined", axis=1, errors="ignore")
    nodes_df = pd.concat([nodes_df, groups], axis=1)
    global_attrs["group_names"] += groups.columns.tolist()

    nodes_df = nodes_df.rename({"r": "radius", "l": "length"}, axis=1)

    # new columns
    # TODO: add only if attribute does not exist already
    is_comp = nodes_df["branch_index"].notna()

    defaults = {"capacitance": 1.0, "v": -70.0, "axial_resistivity": 5000.0}
    nodes_df.loc[is_comp, defaults.keys()] = defaults.values()

    # TODO: rename cell_index > cell, comp_index > comp, branch_index > branch
    nodes_df.loc[is_comp, "comp_index"] = pd.Series(
        range(sum(is_comp)), dtype=pd.Int64Dtype()
    )
    nodes_df["cell_index"] = 0

    return pandas_to_nx(nodes_df, edge_df, global_attrs)


def _extract_branchpoints(G: nx.DiGraph) -> nx.DiGraph:
    """Move branchpoints to graph.graph and contract branchpoint nodes with their neighbour.

    Removes all branchpoint and tip nodes by contracting it into the neighbour with the
    lowest node_index. All branchpoints and tips are moved to the global graph attribute
    `branchpoints_and_tips`.

    Choosing the lowest node is somewhat arbitrary and can result in different graphs.
    See example below:
    [[1] = branchpoint, (1) = compartment]

                Example 1             |            Example 2
    ----------------------------------|----------------------------------
     (1) --> [2] --> (3)  (1) --> (3) | (3) <-- [2] <-- (1)  (2) <-- (1)
              |            |          |          |                    |
              v            v          |          v                    v
             (4)          (4)         |         (4)                  (4)

    Args:
        G: The graph with branchpoints and tips.

    Returns:
        The graph without branchpoints and tips.
    """
    branchpoints_tips = {
        n: d for n, d in G.nodes(data=True) if pd.isna(d["branch_index"])
    }

    updated_G = G.copy()
    for n in list(branchpoints_tips.keys()):
        branchpoint_edges = sorted((*G.in_edges(n), *G.out_edges(n)))
        branchpoints_tips[n]["edges"] = branchpoint_edges
        rm_edge = branchpoint_edges[0]
        rm_edge = rm_edge[::-1] if n == rm_edge[0] else rm_edge
        updated_G = nx.contracted_nodes(
            updated_G, *rm_edge, self_loops=False, copy=False
        )
        del updated_G.nodes[rm_edge[0]]["contraction"]

    updated_G.graph["branchpoints_and_tips"] = pd.DataFrame(branchpoints_tips).T
    return updated_G


def _insert_branchpoints(G: nx.DiGraph) -> nx.DiGraph:
    """Move branchpoints back from graph.graph into graph.nodes.

    Inserts branchpoint and tip nodes back into the graph and restores reconnects them
    to their respective neighbours. Removes `branchpoints_and_tips` attribute from graph.

    Examples:
    [[1] = branchpoint, (1) = compartment]

                Example 1             |            Example 2
    ----------------------------------|----------------------------------
     (1) --> (3)  (1) --> [2] --> (3) | (1) <-- (2)  (1) <-- [2] <-- (3)
      |                    |          |          |            |
      v                    v          |          v            v
     (4)                  (4)         |         (4)          (4)

    Args:
        G: The graph without branchpoints and tips.

    Returns:
        The graph with branchpoints and tips.
    """
    branchpoint_edges = G.graph["branchpoints_and_tips"].pop("edges")
    branchpoints_tips = G.graph["branchpoints_and_tips"].T.to_dict()
    del G.graph["branchpoints_and_tips"]

    updated_G = G.copy()
    edge_attrs = {"comp_edge": True, "synapse": False}

    for node, edges in branchpoint_edges.items():
        # Find the sink node (the one that absorbed the branchpoint)
        # assumes 1st edge was contracted
        sink = edges[0][0] if edges[0][1] == node else edges[0][1]

        # Create list of edges to remove by replacing node with sink
        edges_to_remove = [
            (sink if u == node else u, sink if v == node else v) for u, v in edges
        ]

        # Remove the contracted edges
        updated_G.remove_edges_from(edges_to_remove)

        # Restore branchpoint and its original edges
        updated_G.add_node(node, **branchpoints_tips[node])
        updated_G.add_edges_from(edges, **edge_attrs)

    return updated_G


def to_graph(module, insert_branchpoints: bool = False):
    # check needed to work modules that were not imported via from_graph
    # TODO: branchpoints_and_tips should be a Module attribute, then this check is not needed
    if module.branchpoints_and_tips is None:
        module.branchpoints_and_tips = module._branchpoints

    edges = module._comp_edges.copy()
    condition1 = edges["type"].isin([2, 3])
    condition2 = edges["type"] == 0
    condition3 = edges["source"] < edges["sink"]
    edges = edges[condition1 | (condition3 & condition2)][["source", "sink"]]
    edges.set_index(["source", "sink"], inplace=True)
    edges.index.names = (None, None)
    edges["synapse"] = False

    nodes = module.nodes.drop(
        columns=[col for col in module.nodes.columns if "local_" in col]
    )
    nodes.columns = nodes.columns.str.replace("global_", "")
    nodes = nodes.drop(["controlled_by_param"], axis=1)
    nodes["branch_index"] = nodes["branch_index"].astype(pd.Int64Dtype())
    nodes["cell_index"] = nodes["cell_index"].astype(pd.Int64Dtype())
    nodes["comp_index"] = nodes["comp_index"].astype(pd.Int64Dtype())
    nodes = nodes.combine_first(module._branchpoints)
    nodes = nodes.drop(columns=["type"])

    G = pandas_to_nx(nodes, edges, pd.Series())
    G = _extract_branchpoints(G)

    global_attrs = [
        "xyzr",
        "channels",
        "synapses",
        "group_names",
        "branchpoints_and_tips",
    ]
    for attr in global_attrs:
        G.graph[attr] = getattr(
            module, attr
        ).copy()  # TODO: copy is needed here, I'm unsure why tho

    return _insert_branchpoints(G) if insert_branchpoints else G


########################################################################################
################################ MODULE FROM GRAPH #####################################
########################################################################################


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
    brach_index_of = lambda x: node_df["branch_index"].loc[x].values
    if edge_df.empty:  # for single compartment graphs
        return [-1]
    node_i, node_j = np.stack(edge_df.index).T
    is_branch_edge = brach_index_of(node_i) != brach_index_of(node_j)
    branch_edge_inds = edge_df.index[is_branch_edge]
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
        acc_parents.append([-1] + parents.astype(int).tolist())
    return acc_parents


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

    # jaxley expects contiguous indices, but since we drop branchpoints in
    # _extract_branchpoints, we need to re-assign the indices here
    node_df.index = module.nodes.index

    # set column-wise. preserves cols not in df.
    module.nodes[node_df.columns] = node_df

    synapse_edges = edge_df[edge_df.synapse if len(edge_df) > 0 else []]
    module.edges = synapse_edges if not synapse_edges.empty else module.edges

    # add all the extra attrs
    module.synapses = global_attrs["synapses"]
    module.channels = global_attrs["channels"]
    module.group_names = global_attrs["group_names"]
    module.membrane_current_names = [c.current_name for c in module.channels]
    module.synapse_names = [s._name for s in module.synapses]
    module.branchpoints_and_tips = global_attrs["branchpoints_and_tips"]

    return module


# TODO: add kwarg functionality
def from_graph(
    comp_graph: nx.DiGraph,
    assign_groups: bool = True,
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

        from jaxley.io.graph import swc_to_nx, build_compartment_graph, from_graph
        swc_graph = swc_to_nx("path_to_swc.swc")
        comp_graph = build_compartment_graph(swc_graph, ncomp=1)
        cell = from_graph(comp_graph)
    """

    if not "channels" in comp_graph.graph:
        comp_graph = _add_jaxley_meta_data(comp_graph)
    if comp_graph.graph["branchpoints_and_tips"].empty:
        comp_graph = _extract_branchpoints(comp_graph)
    module = _build_module(comp_graph)
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
    cmap = lambda x: branchpoint_color if x else comp_color
    colors = [
        cmap(pd.isna(comp_graph.nodes[n].get("branch_index", True)))
        for n in comp_graph.nodes
    ]

    pos = {k: (v["x"], v["y"]) for k, v in comp_graph.nodes.items()}

    ax = plt.subplots(1, 1, figsize=(4, 4))[1] if ax is None else ax

    nx.draw(
        comp_graph,
        pos=pos,
        with_labels=True,
        font_size=font_size,
        node_size=node_size,
        ax=ax,
        node_color=colors,
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

    # Combine the graph1 and grpah2 into one graph.
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
