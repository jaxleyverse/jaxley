# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from jaxley.modules import Branch, Cell, Compartment, Module, Network
from jaxley.utils.cell_utils import rev_solid_props

########################################################################################
################################### Cell utils #########################################
########################################################################################

def trapz_integral(
    xp: np.ndarray,
    fp: np.ndarray,
    x1: Optional[float] = None,
    x2: Optional[float] = None,
) -> float:
    """Trapezoidally integrate a function between two points.
    Args:
        xp: The x-values of the function.
        fp: The y-values of the function.
        x1: The lower bound of the integration. If `None`, the first point of `xp` is used.
        x2: The upper bound of the integration. If `None`, the last point of `xp` is used.
    Returns:
        The integral of the function between x1 and x2.
    """
    x1 = xp[0] if x1 is None else x1
    x2 = xp[-1] if x2 is None else x2

    # Find indices for the segment [x1, x2]
    mask = (xp >= x1) & (xp <= x2)
    x_seg = xp[mask]
    fp_seg = fp[mask]

    # Add boundary points if needed
    if x1 not in x_seg:
        r1 = np.interp(x1, xp, fp)
        x_seg = np.insert(x_seg, 0, x1)
        fp_seg = np.insert(fp_seg, 0, r1)

    if x2 not in x_seg:
        r2 = np.interp(x2, xp, fp)
        x_seg = np.append(x_seg, x2)
        fp_seg = np.append(fp_seg, r2)

    # Trapezoidal integration
    integral = np.trapezoid(fp_seg, x_seg)

    return integral


def rev_solid_props(
    ls: np.ndarray,
    rs: np.ndarray,
    l_start: Optional[float] = None,
    l_end: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Calculate properties of a solid of revolution given length and radius coordinates.
    Args:
        ls: array of length coordinates along the path
        rs: array of radius coordinates
        l_start: optional start position for integration (default: min(ls))
        l_end: optional end position for integration (default: max(ls))
    Returns:
        average_radius, surface_area, volume
    """
    if len(ls) != len(rs):
        raise ValueError("ls and rs must have the same length")

    # Set integration bounds
    l_start = ls[0] if l_start is None else l_start
    l_end = ls[-1] if l_end is None else l_end

    # Calculate derivatives dr/dl for surface area calculation
    # Use central differences where possible, forward/backward at endpoints
    dr_dl = np.zeros_like(rs)

    # Forward difference at start
    if len(rs) > 1:
        dl = ls[1] - ls[0]
        dr_dl[0] = (rs[1] - rs[0]) / dl if dl > 0 else 0

    # Central differences in middle
    for i in range(1, len(rs) - 1):
        dr_dl[i] = (rs[i + 1] - rs[i - 1]) / (ls[i + 1] - ls[i - 1])

    # Backward difference at end
    if len(rs) > 1:
        dl = ls[-1] - ls[-2]
        dr_dl[-1] = (rs[-1] - rs[-2]) / dl if dl > 0 else 0

    # a) Surface Area: SA = 2π ∫ r * sqrt(1 + (dr/dl)²) dl
    surface_integrand = 2 * np.pi * rs * np.sqrt(1 + dr_dl**2)
    surface_area = trapz_integral(ls, surface_integrand, l_start, l_end)

    # b) Volume: V = π ∫ r² dl
    volume_integrand = np.pi * rs**2
    volume = trapz_integral(ls, volume_integrand, l_start, l_end)

    # c) Average Radius: r_avg = ∫ r dl / ∫ dl = ∫ r dl / L
    # where L is the integration length
    radius_integrand = rs
    integration_length = l_end - l_start
    average_radius = (
        trapz_integral(ls, radius_integrand, l_start, l_end) / integration_length
    )

    return average_radius, surface_area, volume

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
        edge_attr=True,
        create_using=nx.DiGraph(),
    )
    G.add_nodes_from((n, dict(d)) for n, d in node_attrs.iterrows())
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
    degree_is_2 = lambda n: G.out_degree(n) + G.in_degree(n) == 2

    interrupt_edges = []
    for n in G.nodes:
        if len(parents := list(G.predecessors(n))) > 0:
            parent = parents[0]
            if parent != n - 1 and degree_is_2(n) and degree_is_2(parent):
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


def branch_comps_from_nodes(
    branch_nodes: pd.DataFrame, ncomp: int, ignore_branchpoint: bool = False
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
        ignore_branchpoint: Whether to consider the branchpoint part of the neurite or
        not. This is for example relevant if the branch extends from a single point soma
        or somatic branchpoint. In these cases, the somatic SWC node is _not_ considered
        to be part of the dendrite.

    Returns:
        DataFrame of compartments and compartment attributes.
    """
    # TODO: This should be reusable for `set_ncomp()`
    node_inds_in_branch = branch_nodes.index.tolist()

    compute_edge_lens = lambda x: (x.diff(axis=0).fillna(0) ** 2).sum(axis=1) ** 0.5
    edge_lens = compute_edge_lens(branch_nodes[["x", "y", "z"]])
    branch_nodes["l"] = edge_lens.cumsum()  # path length

    # handle single point branches / somata
    if len(node_inds_in_branch) == 1:
        # duplicate node to compartmentalize it along its "length", i.e. l = 2*r
        branch_nodes = branch_nodes.loc[node_inds_in_branch * 2]
        # Setting l = 2*r ensures A_cylinder = 2*pi*r*l = 4*pi*r^2 = A_sphere.
        branch_nodes["l"] = np.array([0, 2 * branch_nodes["r"].iloc[0]])

    # branches originating from branchpoint with different type id start at first branch
    # node, i.e. have attrs set to those of first branch node.
    is_branch_id = (branch_nodes["id"] == branch_nodes["id"].iloc[1]).values
    if np.any(~is_branch_id) and is_branch_id[1] and ignore_branchpoint:
        next2branchpoint_attrs = branch_nodes.iloc[[1, -2]]
        # if branchpoint has a different id, set attrs equal to neighbour of branchpoint
        next2branchpoint_attrs = next2branchpoint_attrs.iloc[~is_branch_id[[0, -1]]]
        branch_nodes.loc[~is_branch_id] = next2branchpoint_attrs.values

    branch_id = branch_nodes["id"].iloc[1]  # node after branchpoint det. branch id
    branch_len = max(branch_nodes["l"])
    comp_len = branch_len / ncomp
    comp_centers = list(np.linspace(comp_len / 2, branch_len - comp_len / 2, ncomp))

    # Create node indices and attributes for branch-tips/branchpoints and comps
    # is_comp, comp_id, comp_len, x, y, z, r
    comp_attrs = [
        [False, branch_nodes["id"].iloc[0], 0],  # branch tip
        *[[True, branch_id, comp_len]] * ncomp,  # comps
        [False, branch_nodes["id"].iloc[-1], 0],  # branch tip
    ]

    # Interpolate xyz along branch
    x = np.array([0, *comp_centers, branch_len])
    xp = branch_nodes["l"].values
    fp_xyz = branch_nodes[["x", "y", "z"]].values

    interpolated_coords = np.column_stack([np.interp(x, xp, fp) for fp in fp_xyz.T])

    # trapezoidal integration of r between compartment tips
    comp_ends = np.linspace(0, branch_len, ncomp + 1)
    comp_tips = np.stack([comp_ends[:-1], comp_ends[1:]], axis=1)
    fp_r = branch_nodes["r"].values

    frustum_props = [rev_solid_props(xp, fp_r, x1, x2) for x1, x2 in comp_tips]
    # radius, area, volume
    frustum_props = np.array([[fp_r[0], 0, 0], *frustum_props, [fp_r[-1], 0, 0]])

    # Combine interpolated coordinates with existing attributes
    return np.hstack([comp_attrs, interpolated_coords, frustum_props])


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

    branches = list_branches(
        G,
        source=root,
        ignore_swc_tracing_interruptions=ignore_swc_tracing_interruptions,
        max_len=max_len,
    )
    nodes_df = nx_to_pandas(G)[0].astype(float)

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
    num_additional_inds = len(branches) * ncomp
    existing_inds = nodes_df.index

    proposed_node_inds = propose_new_inds(existing_inds, num_additional_inds)

    # collect comps and comp_edges
    comps, comp_edges, xyzr = [], [], []
    for branch_idx, branch in enumerate(branches):
        # ensure node_index increases monotonically along branch. Required for branch.loc()
        branch = branch[::-1] if branch[0] < branch[-1] else branch

        comp_inds = proposed_node_inds[branch_idx * ncomp : (branch_idx + 1) * ncomp]
        comp_inds = np.array([branch[0], *comp_inds, branch[-1]])
        branch_inds = np.array([float("nan"), *[branch_idx] * ncomp, float("nan")])

        node_attrs = nodes_df.loc[branch]

        has_somatic_branchpoint = np.any(np.isin(branch, soma_branchpoints))
        comp_attrs = branch_comps_from_nodes(node_attrs, ncomp, has_somatic_branchpoint)
        comp_attrs = np.hstack([comp_inds[:, None], branch_inds[:, None], comp_attrs])

        # single soma branches lead to self looping edges, since branch[0] == branch[-1]
        # we therefore remove one tip node / branchpoint node, i.e. [0,s,0] -> [s,0]
        comp_attrs = comp_attrs[1:] if branch[0] == branch[-1] else comp_attrs

        # Store edges, nodes, and xyzr in branch-wise manner
        intra_branch_edges = np.stack([comp_attrs[:-1, 0], comp_attrs[1:, 0]]).T
        comp_edges.append(intra_branch_edges.astype(int).tolist())
        comps.append(comp_attrs)

        # store xyzr for each node in branch
        xyzr.append(node_attrs[["x", "y", "z", "r"]].values)

    comps = np.concatenate(comps)
    comp_inds_cols = ["node", "branch", "is_comp"]
    comp_cols = ["id", "l", "x", "y", "z", "r", "area", "volume"]
    comp_df = pd.DataFrame(comps, columns=comp_inds_cols + comp_cols)

    int_cols = ["node", "id"]  # branch cols are floats due to branchpoints
    comp_df[int_cols] = comp_df[int_cols].astype(int)

    bool_cols = ["is_comp"]
    comp_df[bool_cols] = comp_df[bool_cols].astype(bool)

    # threshold radius
    if min_radius is None:
        assert (
            comp_df["r"] > 0.0
        ).all(), "Radius 0.0 in SWC file. Set `read_swc(..., min_radius=...)`."
    else:
        comp_df["r"] = np.maximum(comp_df["r"], min_radius)

    # drop duplicated branchpoint nodes and replace with original branchpoint node attrs
    comp_df = comp_df.drop_duplicates(subset=["node", "is_comp"])
    comp_df = comp_df.set_index("node")
    xyzr_cols = ["x", "y", "z", "r"]
    at_branchpoints = comp_df.loc[~comp_df["is_comp"]].index
    comp_df.loc[at_branchpoints, xyzr_cols] = nodes_df.loc[at_branchpoints, xyzr_cols]

    # create comp edges
    comp_edges = sum(comp_edges, [])
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
    # TODO: make channels and synapses dicts?
    module_global_attrs = pd.Series({"channels": [], "synapses": [], "group_names": []})
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

    jaxley_keys = {"r": "radius", "l": "length", "branch": "branch_index"}
    nodes_df = nodes_df.rename(jaxley_keys, axis=1)

    # new columns
    is_comp = nodes_df["is_comp"]
    nodes_df.loc[is_comp, "capacitance"] = 1.0
    nodes_df.loc[is_comp, "v"] = -70.0
    nodes_df.loc[is_comp, "axial_resistivity"] = 5000.0
    # TODO: rename cell_index > cell, comp_index > comp, branch_index > branch
    nodes_df.loc[is_comp, "comp_index"] = np.arange(sum(is_comp))
    nodes_df["cell_index"] = 0

    return pandas_to_nx(nodes_df, edge_df, global_attrs)


def _set_graph_direction(G: nx.DiGraph, source=None) -> nx.DiGraph:
    """Determine from which root to traverse the graph dfs and orient edges accordingly.

    For this, the graph is traversed depth first along the undirected edges. If edges
    in the directed graph are oriented in the wrong direction, they are flipped.

    Args:
        G: Directed graph in which to flip edges along the dfs traversal.
        source: The source node to start the traversal from. If `None`, the first leaf
            node is used.

    Returns:
        The graph with the edges oriented in dfs fashion.
    """
    # reorder graph depth-first TODO: Which order to choose?
    leaf = next(n for n in G.nodes if G.degree(n) == 1)
    source = leaf if source is None else source

    if "is_comp" in G.nodes[source]:
        # branchpoints cannot act as solve roots since they have to be replaced by a
        # set of edges. For this branchpoints need a parent node for the order to be
        # uniquely determined. For details see `_replace_branchpoints_with_edges``.
        assert (
            G.nodes[source]["is_comp"] or G.degree(source) == 1
        ), "Source cannot be a branchpoint."

    for u, v in nx.dfs_edges(G.to_undirected(), source):
        if (u, v) not in G.edges and (v, u) in G.edges:  # flip edge direction
            G.add_edge(u, v, **G.get_edge_data(v, u))
            G.remove_edge(v, u)

    return G


def _replace_branchpoints_with_edges(G: nx.DiGraph) -> nx.DiGraph:
    """Replace branchpoint nodes with edges connecting parent and children and remove tips.

    Removes all non-compartment nodes. Branchpoints and tips are stored in the global
    graph attribute `branchpoints_and_tips`.

    This depends on the directionality of the edges! To ensure that the edges are
    correctly oriented, you can use `_set_graph_direction()` to reorder the graph
    before calling this function. [(x) = branchpoint, (1) = compartment]

                Example 1             |            Example 2
    ----------------------------------|----------------------------------
     (1) --> (x) --> (2)  (1) --> (2) | (1) <-- (x) <-- (2)  (1) <-- (2)
              |            |          |          |                    |
              v            v          |          v                    v
             (3)          (3)         |         (3)                  (3)

    Args:
        G: The graph to replace branchpoints with edges.

    Returns:
        The graph with branchpoints replaced with edges and tips removed.
    """
    # TODO: fix source kwarg
    G = G.copy()
    G.add_edges_from([(i, j, {"branch_edge": False}) for i, j in G.edges])
    branch_edge_attrs = {"comp_edge": True, "synapse": False, "branch_edge": True}
    xyz_of = lambda n: [G.nodes[n][key] for key in ["x", "y", "z"]]

    branchpoints_tips = {}
    for n in G.nodes:
        if not G.nodes[n]["is_comp"]:
            parents = list(G.predecessors(n))
            children = list(G.successors(n))
            branchpoints_tips[n] = {
                **G.nodes[n],
                "xyz_children": [xyz_of(c) for c in children],
            }
            if len(parents) == 0:  # root node
                branchpoints_tips[n]["xyz_parent"] = float("nan")
            else:  # branchpoint node or tip node (no children)
                parent = parents[0]
                branchpoints_tips[n]["xyz_parent"] = xyz_of(parent)
                for child in children:
                    G.add_edges_from([(parent, child)], **branch_edge_attrs)
                G.remove_edge(parent, n)
        else:
            G.nodes[n]["comp_index"] = int(G.nodes[n]["comp_index"])
            G.nodes[n]["branch_index"] = int(G.nodes[n]["branch_index"])
            del G.nodes[n]["is_comp"]  # del `is_comp`

    # TODO: keep disconnected branchpoint / tip nodes in graph?
    #  instead of adding them as a global attribute?
    for n in branchpoints_tips:
        # for child in  list(G.successors(n)):
        #     G.remove_edge(n, child)
        G.remove_node(n)

    cols = ["x", "y", "z", "radius", "cell_index"]
    cols += G.graph["group_names"] + ["xyz_parent", "xyz_children"]
    G.graph["branchpoints_and_tips"] = pd.DataFrame(branchpoints_tips).T[cols]

    return G


def _replace_edges_with_branchpoints(G: nx.DiGraph) -> nx.DiGraph:
    """Replace edges with branchpoints connecting parent and children and add tips.

    Insert branchpoint nodes into graph. Branchpoints and tips are stored in the global
    graph attribute `branchpoints_and_tips`. Connects the nodes based on the xyz
    coordinates of the parent and child nodes. This way node indices and edge direction
    can be changed in between running `_replace_branchpoints_with_edges` and
    `_replace_edges_with_branchpoints`.

    [(x) = branchpoint, (1) = compartment]

                Example 1             |            Example 2
    ----------------------------------|----------------------------------
     (1) --> (2)  (1) --> (x) --> (2) | (1) <-- (2)  (1) <-- (x) <-- (2)
      |                    |          |          |            |
      v                    v          |          v            v
     (3)                  (3)         |         (3)          (3)

    Args:
        G: The graph to replace edges with branchpoints.

    Returns:
        The graph with branchpoints and tips.
    """
    branchpoints_tips = G.graph["branchpoints_and_tips"]
    new_inds = propose_new_inds(list(G.nodes), len(branchpoints_tips))
    node_xyz = nx_to_pandas(G)[0][["x", "y", "z"]]

    branch_edge_attrs = {"comp_edge": True, "synapse": False, "branch_edge": True}
    closest_node = lambda n_xyz: (node_xyz - n_xyz).abs().sum(axis=1).idxmin()

    for idx, (r, row) in zip(new_inds, branchpoints_tips.iterrows()):
        idx = r if r not in G.nodes else idx
        G.add_node(idx, **row.drop(["xyz_children", "xyz_parent"]).to_dict())

        if np.isnan(row["xyz_parent"]).all():  # root node
            child_nodes = [closest_node(c) for c in row["xyz_children"]]
            G.add_edges_from([(idx, c) for c in child_nodes], **branch_edge_attrs)
        elif len(row["xyz_children"]) == 0:  # tip node
            parent_node = closest_node(row["xyz_parent"])
            G.add_edges_from([(parent_node, idx)], **branch_edge_attrs)
        else:  # branchpoint nodes
            parent_node = closest_node(row["xyz_parent"])
            child_nodes = [closest_node(c) for c in row["xyz_children"]]

            G.remove_edges_from([(parent_node, c) for c in child_nodes])
            G.add_edges_from([(parent_node, idx)], **branch_edge_attrs)
            G.add_edges_from([(idx, c) for c in child_nodes], **branch_edge_attrs)

    del G.graph["branchpoints_and_tips"]
    return G


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
    # _replace_branchpoints_with_edges, we need to re-assign the indices here
    node_df.index = module.nodes.index

    # set column-wise. preserves cols not in df.
    module.nodes[node_df.columns] = node_df

    synapse_edges = edge_df[edge_df.synapse]
    module.edges = synapse_edges if not synapse_edges.empty else module.edges

    # add all the extra attrs
    module.synapses = global_attrs["synapses"]
    module.channels = global_attrs["channels"]
    module.group_names = global_attrs["group_names"]
    module.membrane_current_names = [c.current_name for c in module.channels]
    module.synapse_names = [s._name for s in module.synapses]

    return module


# TODO: add kwarg functionality
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

        from jaxley.io.graph import swc_to_nx, build_compartment_graph, from_graph
        swc_graph = swc_to_nx("path_to_swc.swc")
        comp_graph = build_compartment_graph(swc_graph, ncomp=1)
        cell = from_graph(comp_graph)
    """

    comp_graph = _add_jaxley_meta_data(comp_graph)
    # edge direction matters from here on out
    comp_graph = _set_graph_direction(comp_graph, source=solve_root)
    solve_graph = _replace_branchpoints_with_edges(comp_graph)
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
    cmap = lambda x: branchpoint_color if x else comp_color
    colors = [cmap(comp_graph.nodes[n].get("is_comp", True)) for n in comp_graph.nodes]

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
