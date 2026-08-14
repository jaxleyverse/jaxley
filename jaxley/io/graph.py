# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from itertools import count
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import networkx as nx
import numpy as np
import pandas as pd

from jaxley.modules import Branch, Cell, Compartment, Module, Network
from jaxley.modules.base import infer_module_type
from jaxley.utils.misc_utils import cumsum_leading_zero
from jaxley.utils.morph_attributes import compartmentalize, compute_cone_props

#########################################################################################
################################### Helper functions ####################################
#########################################################################################


def pandas_to_nx(
    node_attrs: pd.DataFrame, edge_attrs: pd.DataFrame, global_attrs: pd.Series
) -> nx.Graph:
    """Convert node_attrs, edge_attrs and global_attrs from pandas datatypes to a NetworkX Graph.

    Args:
        node_attrs: DataFrame containing node attributes
        edge_attrs: DataFrame containing edge attributes
        global_attrs: Series containing global graph attributes

    Returns:
        An undirected graph with nodes, edges and global attributes from the input data.
    """
    graph = nx.from_pandas_edgelist(
        edge_attrs.reset_index(),
        source="level_0",
        target="level_1",
        edge_attr=True if edge_attrs.columns.size > 0 else None,
        create_using=nx.Graph(),
    )
    graph.add_nodes_from((n, d) for n, d in node_attrs.to_dict(orient="index").items())
    graph.graph.update(global_attrs.to_dict())
    return graph


def nx_to_pandas(
    graph: nx.Graph, sort_index: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Convert a NetworkX Graph to pandas datatypes.

    Args:
        graph: Input graph
        sort_index: Whether to sort the index of the DataFrames.

    Returns:
        Tuple containing:
        - DataFrame of node attributes
        - DataFrame of edge attributes
        - Series of global graph attributes
    """
    edge_df = nx.to_pandas_edgelist(graph).set_index(["source", "target"])
    edge_df.index.names = [None, None]
    node_df = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient="index")
    node_df = node_df.sort_index() if sort_index else node_df
    edge_df = edge_df.sort_index() if sort_index else edge_df

    return node_df, edge_df, pd.Series(graph.graph)


def swc_to_pandas(
    fname: str, num_lines: Optional[int] = None, converters: Optional[callable] = None
) -> pd.DataFrame:
    """Read a SWC morphology file into a pandas DataFrame.

    Args:
        fname: Path to the SWC file
        num_lines: Number of lines to read from the file. If None, all lines are read.
        converters: Functions that can be applied to each column [id, x, y, z, r, p] of
            the SWC file i.e. to convert the swc file to the desired units.
            See here: http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html

    Returns:
        A pandas DataFrame of the SWC file.
    """
    swc = pd.read_csv(
        fname,
        sep=r"\s+",
        comment="#",
        names=["id", "x", "y", "z", "radius", "p"],
        nrows=num_lines,
        skipinitialspace=True,
        index_col=0,
    )

    converters = converters or {}
    for col, func in converters.items():
        swc[col] = swc[col].apply(func)
    return swc


def nx_to_swc(graph: nx.Graph) -> pd.DataFrame:
    """Convert a NetworkX Graph to a pandas DataFrame.

    Args:
        graph: NetworkX Graph

    Returns:
        A pandas DataFrame of the SWC file.
    """
    swc = nx_to_pandas(graph)[0]

    # `p` holds the parent's node label, not its positional index, such that
    # `swc_to_nx()` (which reads `p` as a node label) is the inverse of this function.
    swc["p"] = [-1] * len(swc.index)
    for parent, child in nx.bfs_edges(graph, np.min(swc.index)):
        swc.loc[child, "p"] = parent

    return swc


def swc_to_nx(
    swc: pd.DataFrame,
    relevant_ids: Optional[List[int]] = None,
) -> nx.Graph:
    """Read a SWC morphology (loaded as pandas DataFrame) into a NetworkX Graph.

    The graph is read such that each entry in the swc file becomes a graph node
    with the column attributes (id, x, y, z, r). Then each node is connected to its
    designated parent via an edge.

    Args:
        swc: pandas DataFrame of the SWC file.
        relevant_ids: List of ids to include in the graph. Defaults to [1, 2, 3, 4].
            All other ids are set to 0.

    Returns:
        A networkx Graph of the traced morphology in the swc file. It has attributes:
        nodes: {'id': 1, 'x': 0.0, 'y': 0.0, 'z': 0.0, 'r': 1.0}
        edges: {}

    Example usage
    ^^^^^^^^^^^^^

    ::

        from jaxley.io.graph import swc_to_nx, swc_to_pandas
        swc_graph = swc_to_nx(swc_to_pandas("path_to_swc.swc"))
    """
    relevant_ids = relevant_ids or [1, 2, 3, 4]
    swc = swc.copy()  # do not mutate the caller's DataFrame
    swc["id"] = swc["id"].where(swc["id"].isin(relevant_ids), 0)

    graph = nx.Graph()
    xyzr = swc[["id", "x", "y", "z", "radius"]].astype(float)
    graph.add_nodes_from(
        (i, {"id": id_, "x": x, "y": y, "z": z, "radius": radius})
        for i, id_, x, y, z, radius in zip(swc.index, *(xyzr[c] for c in xyzr))
    )

    parents, children = swc["p"].to_numpy(), swc.index.to_numpy()
    traced = parents != -1
    graph.add_edges_from(zip(parents[traced].astype(int), children[traced]))
    return graph


########################################################################################
################################ BUILD COMPARTMENT GRAPH ###############################
########################################################################################


def split_branches(
    branches: list[list[int]], split_edges: list[tuple[int, int]]
) -> list[list[int]]:
    """Split branches at the given edges.

    The two resulting sub-branches share the node at which the split occurred.

    Args:
        branches: List of branches, each represented as list of nodes.
        split_edges: List of edges between nodes where tracing is discontinous.

    Returns:
        An updated list of branches.
    """
    for n1, n2 in split_edges:
        for i, branch in enumerate(branches):
            if n1 in branch and n2 in branch:
                if branch.index(n1) > branch.index(n2):
                    n1, n2 = n2, n1
                start = branch.index(n1) + 1
                end = branch.index(n2) - 1
                # `n1` already starts the branch, so the split point is a branch
                # boundary and would lead to a single-node branch.
                if end == 0:
                    break
                branches[i : i + 1] = [branch[:start], branch[end:]]
                break
    return branches


def split_long_branches(
    graph: nx.Graph, branches: list[list[int]], max_len: float = 242.0
) -> list[list[int]]:
    """Splits too long branches at equidistant points.

    If branch >= 1*max_len, then we split it down the middle. If branch >= 2*max_len,
    then we split it into 3 parts. And so on. This ensures that sub-branches have similar
    length & length <= max_len.

    Args:
        graph: NetworkX graph tracing of swc file.
        branches: List of branches, each represented as list of nodes.
        max_len: Maximum length any branch cannot exceed.

    Returns:
        Branches such that no branch exceeds max_len and such that the resulting sub-branches
        are of similar length.
    """
    xyz = nx_to_pandas(graph)[0][["x", "y", "z"]]

    splits = []
    for branch in branches:
        branch_xyz = xyz.loc[branch]
        lens = np.linalg.norm(np.diff(branch_xyz.values, axis=0), axis=1)
        lens = cumsum_leading_zero(lens)
        if lens.max() > max_len:
            num_splits = int(lens.max() // max_len) + 1
            for seg in np.linspace(0, lens.max(), num_splits + 1)[1:-1]:
                is_less = lens <= seg
                splits.append(
                    (branch_xyz.index[is_less][-1], branch_xyz.index[~is_less][0])
                )
    return split_branches(branches, splits)


def _order_branches(
    graph: nx.Graph, branches: list[list[int]], root: int, swc_ordering: bool = True
) -> list[list[int]]:
    """Assign the order in which branches are numbered.

    The branch containing `root` is always placed first, since `from_graph()` roots each
    cell at its lowest `branch_index`.

    Args:
        graph: NetworkX graph tracing of swc file.
        branches: List of branches, each represented as list of nodes.
        root: The node the morphology is rooted at.
        swc_ordering: If `True`, order by the node indices of each branch, i.e. by
            SWC file order. This is the default because it reproduces the numbering that
            we test against. If `False`, order by geometry, which is invariant to node
            numbering and edge direction.

            NOTE: NEURON numbers sections in order of `Import3d_SWC_read.mksections()`
            which is not sorted by the first traced point of each section. However, the
            branch decomposition itself matches NEURON, even if the numbering differs.

    Returns:
        The ordered branches.
    """
    if swc_ordering:
        keys = [tuple(sorted(b)) for b in branches]
    else:
        xyz_of = lambda n: tuple(round(float(graph.nodes[n][c]), 6) for c in "xyz")
        keys = [xyz_of(b[0]) + xyz_of(b[-1]) + (len(b),) for b in branches]

    order = sorted(range(len(branches)), key=lambda i: keys[i])
    # Put the branch that contains `root` first.
    root_first = sorted(order, key=lambda i: root not in branches[i])
    return [branches[i] for i in root_first]


def list_branches(
    graph: nx.Graph,
    source: Optional[int] = None,
    max_len: Optional[float] = None,
    ignore_swc_tracing_interruptions: bool = True,
    return_branchpoints: bool = False,
    root: Optional[int] = None,
    nodes_swc_ordered: bool = True,
    id_branchpoint_rule: str = "swc_order",
) -> list[list[int]]:
    """Get all uninterrupted paths in the traced morphology (i.e. branches).

    The graph is traversed depth-first starting from the first found leaf node.
    Nodes are considered to be part of a branch if they have only one parent and one
    child, which are both of the same type (i.e. have the same `id`). Nodes which are
    branchpoints or leafs are considered start / end points of a branch. A branchpoint
    can start multiple branches.

    Some swc files contain artefacts, where tracing of the same neurite was done
    in disconnected pieces. NEURON's swc reader introduces a break in the trace at these
    points, since it parses the file in order. This leads to split branches, which
    should be one.

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
        graph: NetworkX graph tracing of swc file.
        source: The node from which to start tracing the graph. If None, the first leaf
            node is used.
        max_len: The maximum length of a branch. If None, there is no limit.
        ignore_swc_tracing_interruptions: Whether to ignore discontinuities in the swc
            tracing order. If False, this will result in split branches at these points.
        return_branchpoints: Whether to return the branchpoints and edges between them
            seperately.
        root: The node that the morphology is rooted at. Branches are oriented away from
            it, and the branch containing it is numbered first. Defaults to the lowest
            node index, which is the SWC root.
        nodes_swc_ordered: Whether to order nodes in SWC order. If `True` (default),
            nodes are ordered by SWC node indices. If `False`, they are ordered
            geometrically.
        id_branchpoint_rule: Determines which node at a change of `id` becomes the
            branchpoint. This choice is order dependent, see `is_id_branchpoint` below.
            - `"swc_order"` (default): the node that comes first in the SWC file, i.e.
                the lower node index. This matches NEURON's `Import3d`, which parses the
                file in order and starts the new section at the last point of the old
                `id`, and is independent of both the traversal direction and the order
                in which edges were added to `graph`. Assumes SWC numbering, i.e. that a
                parent has a lower node index than its children.
            - `"traversal"`: the last node of the old `id` along the traversal
                direction. Deterministic given `source`, but the resulting branches
                depend on which end the branch is traced from.
            - `"branchpoint"`: both nodes become branchpoints, so the change of `id` is
                carried by a short branch spanning it. Fully order agnostic, at the cost
                of one extra branch per change of `id`.

    Returns:
        A list of linear paths in the graph. Each path is represented as list of nodes.

        The order of the branches (and hence `branch_index` downstream) is set by
        `nodes_swc_ordered`, which is the only step that depends on how the SWC file is
        numbered. It does not match NEURON's section index, see `_order_branches()`.
    """
    rules = ("swc_order", "traversal", "branchpoint")
    assert (
        id_branchpoint_rule in rules
    ), f"Unknown id_branchpoint_rule '{id_branchpoint_rule}', expected one of {rules}."
    id_of = lambda n: graph.nodes[n]["id"] if "id" in graph.nodes[n] else 0

    def is_id_branchpoint(n1: int, n2: int) -> bool:
        """Check if degree-2 node n1 is a branchpoint based on ID.

        Which of the two nodes at a change of `id` is taken as the branchpoint is order
        dependent. For a segment [0,1,2,3] with node IDs [1,1,2,2]:
        -> [[1,1], [1,2,2]] => node 1 is taken as branchpoint
        <- [[2,2], [2,1,1]] => node 2 is taken as branchpoint
        `id_branchpoint_rule` selects which of the three conventions is used.
        """
        if graph.degree(n1) != 2:
            return False
        if id_branchpoint_rule == "traversal":
            # Boundary falls on the last node of the old `id` along the traversal.
            return id_of(n1) != id_of(n2)
        if id_branchpoint_rule == "branchpoint":
            # Both nodes at the change of `id` become branchpoints
            return any(id_of(m) != id_of(n1) for m in graph.neighbors(n1))
        # Boundary falls on the node that comes first in the SWC file.
        return any(m > n1 and id_of(m) != id_of(n1) for m in graph.neighbors(n1))

    # A morphology has to be a tree.
    assert nx.is_forest(
        graph
    ), "The morphology contains a cycle, but it has to be a tree."

    soma_nodes = [n for n in graph.nodes() if id_of(n) == 1]
    leaf = next(n for n in graph.nodes() if graph.degree(n) == 1)
    source = leaf if source is None else source
    # Recover the SWC root since we do not propagate parent information
    root = min(graph.nodes()) if root is None else root

    swc_interupts = []
    branches = (
        [soma_nodes] if len(soma_nodes) == 1 else []
    )  # a single soma is its own branch
    for n1, n2 in nx.dfs_edges(graph, source=source):
        if graph.degree(n1) != 2 or n1 == source or is_id_branchpoint(n1, n2):
            branches.append([n1, n2])
        else:
            branches[-1].append(n2)
            # non-continous node indices which are not branchpoints, i.e. edges where node
            # indices are > 1 apart, signal that a branch was interrupted during tracing
            if np.abs(n2 - n1) != 1:
                swc_interupts.append((n1, n2))

    # split branches (if tracing was interrupted or max_len is reached)
    if not ignore_swc_tracing_interruptions:
        branches = split_branches(branches, swc_interupts)

    # max_len splitting only after accounting for interrupted branches
    if max_len is not None:
        branches = split_long_branches(graph, branches, max_len)

    # Orient every branch away from `root`. Assumed by `compartmentalize_branch()`
    # and `Cell(parents)`. Purely topological
    hops = nx.single_source_shortest_path_length(graph, root)
    branches = [b if hops[b[0]] <= hops[b[-1]] else b[::-1] for b in branches]

    branches = _order_branches(graph, branches, root, nodes_swc_ordered)

    branch_tips = sum([[b[0], b[-1]] for b in branches], [])
    branchpoints_tips = sorted(set(branch_tips))

    return (branches, branchpoints_tips) if return_branchpoints else branches


def compute_xyz(
    graph: nx.Graph,
    length: float = 1.0,
    spread: float = np.pi / 8,
    spread_decay: float = 0.9,
    twist: float = 0.0,
    xy_only: bool = True,
) -> Dict[int, tuple[float, float, float]]:
    """Compute xyz coordinates for a tree-like appearance of a networkX graph in 2D or 3D.

    Handles branches implicitly since nodes in a branch have 1 child.

    Args:
        graph: The Graph to compute node xyz coordinates for.
        length: The length of each edge.
        spread: The opening angle at which the edges spread out.
        spread_decay: Multiplicative decay factor for the opening angle / spread.
        twist: Add additional twisting. Means fewer overlapping nodes in 3D projections.
        xy_only: Whether to only compute the xy coordinates and fix the z-coordinate.

    Returns:
        A dictionary mapping node indices to xyz coordinates.
    """
    root = next(n for n, d in graph.degree() if d == 1)
    pos = {root: (0.0, 0.0, 0.0)}

    def recurse(node, depth=1, theta=0.0, phi=np.pi / 2):
        neighbors = list(graph.neighbors(node))
        children = [n for n in neighbors if n not in pos]
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


def _add_missing_swc_attrs(graph) -> nx.Graph:
    """Add missing swc attributes to a SWC graph.

    Allows to specify morphology from just edges.

    Args:
        graph: The SWC graph to add missing attributes to.

    Returns:
        The SWC graph with missing attributes set to their defaults.
    """
    graph = graph.copy()  # do not mutate the caller's graph
    defaults = {"id": 0, "radius": 1}

    available_keys = graph.nodes[next(iter(graph.nodes()))].keys()
    xyz = compute_xyz(graph) if "x" not in available_keys else {}
    for n, (x, y, z) in xyz.items():
        # xyz is needed to compute compartment lengths
        graph.nodes[n]["x"] = x
        graph.nodes[n]["y"] = y
        graph.nodes[n]["z"] = z

    for key in set(defaults.keys()).difference(available_keys):
        nx.set_node_attributes(graph, defaults[key], key)
    return graph


def build_compartment_graph(
    swc_graph: nx.Graph,
    ncomp: Union[int, Callable[[pd.DataFrame], int]] = 1,
    source: Optional[int] = None,
    min_radius: Optional[float] = None,
    max_len: Optional[float] = None,
    ignore_swc_tracing_interruptions: bool = True,
    root: Optional[int] = None,
    nodes_swc_ordered: bool = True,
    id_branchpoint_rule: str = "swc_order",
) -> nx.Graph:
    """Return a networkX graph that indicates the compartment structure.

    Build a new graph made up of compartments in every branch. These compartments are
    spaced at equidistant points along the branch. Node attributes, like radius are
    linearly interpolated along its length.

    Example: 4 compartments | edges = - | nodes = o | comp_nodes = x
    o-----------o----------o---o---o---o--------o
    o-------x---o----x-----o--xo---o---ox-------o

    This function returns an undirected nx.Graph. The xyzr coordinates that each branch
    tracks are stored in the graph attribute `xyzr`, ordered by the traversal of the
    `swc_graph`, so no edge directionality is needed to recover their order.

    Args:
        swc_graph: Graph generated by `swc_to_nx()`.
        ncomp: How many compartments per branch to insert. Either an `int` (the same for
            every branch) or a callable that is given the branch's SWC nodes (a DataFrame
            with columns `id, x, y, z, r`, ordered along the branch) and returns the
            number of compartments for that branch. The latter allows for a non-uniform
            discretization, e.g. the d-lambda rule, at import time.
        source: The node from which to start tracing the graph, i.e. the traversal start
            passed on to `list_branches()`. This is *not* the root of the resulting cell,
            see `root`. As long as it is a leaf, it does not affect the result.
        root: The node the morphology is rooted at. Branches are oriented away from it
            and the branch containing it becomes `branch_index` 0, which `from_graph()`
            uses as the root of the cell. Defaults to the SWC root, see
            `list_branches()`.
        nodes_swc_ordered: Whether the nodes in each branch are in the same order as
            they appear in the SWC file.
        min_radius: Minimal radius for each compartment.
        max_len: Maximal length for each branch. Longer branches are split into
            separate branches.
        ignore_swc_tracing_interruptions: If `False`, it this function automatically
            starts a new branch when a section is traced with interruptions.
        id_branchpoint_rule: Which node becomes the branchpoint at a change of `id`.
            See `list_branches()`.

    Returns:
        Graph of the compartmentalized morphology.

    Example usage
    ^^^^^^^^^^^^^

    ::

        from jaxley.io.graph import build_compartment_graph, swc_to_nx, swc_to_pandas
        swc_graph = swc_to_nx(swc_to_pandas("path_to_swc.swc"))
        comp_graph = build_compartment_graph(swc_graph, ncomp=1)
    """
    graph = _add_missing_swc_attrs(swc_graph)
    branches = list_branches(
        graph,
        source=source,
        ignore_swc_tracing_interruptions=ignore_swc_tracing_interruptions,
        max_len=max_len,
        root=root,
        nodes_swc_ordered=nodes_swc_ordered,
        id_branchpoint_rule=id_branchpoint_rule,
    )
    nodes_df = nx_to_pandas(graph)[0]

    # threshold radius
    if min_radius is None:
        msg = "Radius 0.0 in SWC file. Set `read_swc(..., min_radius=...)`."
        assert (nodes_df["radius"] > 0.0).all(), msg
    else:
        nodes_df["radius"] = np.maximum(nodes_df["radius"], min_radius)

    # identify somatic branchpoints. A somatic branchpoint is a branchpoint at which at
    # least two connecting branches are somatic. In that case (and in the case of a
    # single-point soma), non-somatic branches are assumed to start from their first
    # traced point, not from the soma.
    soma_nodes = [n for n in graph.nodes if graph.nodes[n]["id"] == 1]
    single_soma = len(soma_nodes) == 1
    soma_branchpoints = [n for n in soma_nodes if graph.degree(n) > 2 or single_soma]
    somatic_nns = lambda n: [n for n in graph.neighbors(n) if graph.nodes[n]["id"] == 1]
    somatic_branchpoints = [
        n for n in soma_branchpoints if len(somatic_nns(n)) >= 2 or single_soma
    ]

    # Temporarily relabel comps with indices not already used as SWC node indices.
    new_node_inds = count(int(max(nodes_df.index)) + 1)

    # Read the node table once and index it positionally below.
    row_of_node = {node: row for row, node in enumerate(nodes_df.index)}
    all_xyzr = nodes_df[["x", "y", "z", "radius"]].to_numpy(dtype=float)
    all_ids = nodes_df["id"].to_numpy()
    somatic_branchpoints = set(somatic_branchpoints)

    # collect comps and comp_edges
    comps, comp_edges, xyzr = [], [], []
    for branch_idx, branch in enumerate(branches):
        rows = np.fromiter(
            (row_of_node[n] for n in branch), dtype=int, count=len(branch)
        )
        first_node, last_node = branch[0], branch[-1]
        branch_ids = all_ids[rows]
        branch_id = branch_ids[1 if len(rows) > 1 else 0]

        # Fancy indexing copies, so the branch can be edited without touching `all_xyzr`.
        branch_xyzr = all_xyzr[rows]

        # A branch end whose `id` differs from the branch's takes the radius of its
        # neighbour. Applied to compartments and to `xyzr`.
        not_branch_id = branch_ids != branch_id
        if not_branch_id[0] and len(rows) > 2:
            branch_xyzr[0, 3] = branch_xyzr[1, 3]
        if not_branch_id[-1] and len(rows) > 2:
            branch_xyzr[-1, 3] = branch_xyzr[-2, 3]

        # A non-somatic branch that starts or ends at a somatic branchpoint drops that
        # point, so its `xyzr` does not span the whole branch. NEURON does the same: such a
        # branch begins at its own first traced point, not at the soma. This results in
        # discontinuities.
        not_soma = branch_id != 1
        start = 1 if (not_soma and first_node in somatic_branchpoints) else 0
        stop = (
            len(rows) - 1 if (not_soma and last_node in somatic_branchpoints) else None
        )
        branch_xyzr = branch_xyzr[start:stop]

        # Compute the compartmentalization of the branch.
        branch_ncomp = (
            ncomp(nodes_df.iloc[rows[start:stop]]) if callable(ncomp) else ncomp
        )
        assert branch_ncomp >= 1, f"ncomp must be >= 1, got {branch_ncomp}."
        comp_attrs = compartmentalize(branch_xyzr, branch_ncomp)
        # Branchpoints and tips have id 0 and no length.
        comp_attrs["id"] = np.where(
            np.isnan(comp_attrs["length"]), 0.0, float(branch_id)
        )

        # Attach branchpoint and tip nodes to the branch.
        # Since branchpoints / tips have the same node_index as in the original graph
        # there is no need to keep track of branch connectivity.
        comp_inds = [next(new_node_inds) for _ in range(branch_ncomp)]
        comp_attrs["node"] = np.array([branch[0], *comp_inds, branch[-1]], dtype=int)
        comp_attrs["global_branch_index"] = np.array(
            [np.nan, *[branch_idx] * branch_ncomp, np.nan]
        )

        # single soma branches lead to self looping edges, since branch[0] == branch[-1]
        # we therefore remove one tip node / branchpoint node, i.e. [0,s,0] -> [s,0]
        if first_node == last_node:
            comp_attrs = {col: vals[1:] for col, vals in comp_attrs.items()}

        # Store edges, nodes, and xyzr in branch-wise manner
        node_inds = comp_attrs["node"]
        comp_edges += [np.stack([node_inds[:-1], node_inds[1:]]).T.tolist()]
        comps.append(comp_attrs)

        # store xyzr for each node in branch
        xyzr.append(branch_xyzr)

    comp_df = pd.DataFrame(
        {col: np.concatenate([c[col] for c in comps]) for col in comps[0]}
    )

    # drop duplicated branchpoint nodes and fill with original attrs of branchpoint node
    comp_df = comp_df.drop_duplicates(subset=["node"])
    comp_df = comp_df.set_index("node")
    xyzr_cols = ["x", "y", "z", "radius"]
    is_comp = comp_df["global_branch_index"].notna()
    at_branchpoints = comp_df.loc[~is_comp].index
    comp_df.loc[at_branchpoints, xyzr_cols] = nodes_df.loc[at_branchpoints, xyzr_cols]

    # create comp edges
    comp_edges = sum(comp_edges, [])
    comp_edges_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(comp_edges))
    comp_edges_df["synapse"] = False  # edges between compartments that are synapses
    comp_edges_df["comp_edge"] = True  # edges between connected compartments

    global_attrs = pd.Series({"xyzr": xyzr})
    graph = pandas_to_nx(comp_df, comp_edges_df, global_attrs)

    # Relabel comps to [0, ..., ncomps-1] and branchpoints/tips to [ncomps, ...], such
    # that the branchpoints can be appended to the end of `Module.nodes`.
    is_comp = comp_df["global_branch_index"].notna()
    comp_labels = {n: i for i, n in enumerate(comp_df.index[is_comp])}
    branchpoint_labels = {
        n: i + len(comp_labels) for i, n in enumerate(comp_df.index[~is_comp])
    }
    graph = nx.relabel_nodes(graph, {**comp_labels, **branchpoint_labels})
    return graph


########################################################################################
################################## BUILD MODULE ########################################
########################################################################################


def _add_jaxley_meta_data(graph: nx.Graph) -> nx.Graph:
    """Add attributes to and rename existing attributes of the compartalized morphology.

    Makes the imported and compartmentalized morphology compatible with jaxley.
    """
    nodes_df, edge_df, global_attrs = nx_to_pandas(graph)
    module_global_attrs = pd.Series(
        {"channels": [], "synapses": [], "group_names": [], "pumps": []}
    )
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

    # new columns
    is_comp = nodes_df["global_branch_index"].notna()

    # Only fill in the missing defaults, so that a graph from  something like
    # `jaxley.io.neuron` or a hand-built one keeps its own values.
    defaults = {"capacitance": 1.0, "v": -70.0, "axial_resistivity": 5000.0}
    defaults = {k: v for k, v in defaults.items() if k not in nodes_df.columns}
    if defaults:
        nodes_df.loc[is_comp, defaults.keys()] = defaults.values()

    nodes_df.loc[is_comp, "global_comp_index"] = pd.Series(
        range(sum(is_comp)), dtype=pd.Int64Dtype()
    )
    nodes_df["global_branch_index"] = nodes_df["global_branch_index"].astype(
        pd.Int64Dtype()
    )
    nodes_df["global_cell_index"] = 0

    return pandas_to_nx(nodes_df, edge_df, global_attrs)


def _extract_branchpoints(graph: nx.Graph) -> nx.Graph:
    """Contract every branchpoint and tip node into one of its neighbouring compartments.

    Removes all branchpoint and tip nodes by contracting them into the neighbour with the
    lowest branch_index, which is what `_build_module()` needs.

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
        graph: The graph with branchpoints and tips.

    Returns:
        The graph without branchpoints and tips.
    """
    branchpoints_tips = {
        n: d for n, d in graph.nodes(data=True) if pd.isna(d["global_branch_index"])
    }

    updated_graph = graph.copy()
    for n in list(branchpoints_tips):
        neighbours = list(graph.neighbors(n))
        if not neighbours:
            # Nothing left to contract into: the compartments this branchpoint or tip sat
            # between have been removed from the graph.
            updated_graph.remove_node(n)
            del branchpoints_tips[n]
            continue
        lowest = neighbours[
            np.argsort([graph.nodes[nn]["global_branch_index"] for nn in neighbours])[0]
        ]
        updated_graph = nx.contracted_nodes(
            updated_graph, lowest, n, self_loops=False, copy=False
        )
        del updated_graph.nodes[lowest]["contraction"]

    return updated_graph


def comp_to_branch_graph(graph: nx.Graph, relabel_nodes: bool = True) -> nx.Graph:
    """Converts a compartment graph to a branch graph.

    Branch graphs are created by contracting all nodes with the same branch_index within
    each cell, such that only one node per branch_index is left. The node with the lowest
    index is chosen as the root of the branch.

    Args:
        graph: The compartment graph to convert.
        relabel_nodes: Whether to relabel the nodes with the branch_index or to keep
            the orignal node index.

    Returns:
        The branch graph.
    """
    node_df = nx_to_pandas(graph)[0]
    branch_graph = graph.copy()

    cell_dfs = (
        node_df.groupby("global_cell_index")
        if "global_cell_index" in node_df.columns
        else [(None, node_df)]
    )

    for _, cell_df in cell_dfs:
        for _, branch_df in cell_df.groupby("global_branch_index"):
            root, *branch_nodes = sorted(branch_df.index)
            for n in branch_nodes:
                branch_graph = nx.contracted_nodes(
                    branch_graph, root, n, self_loops=False, copy=False
                )

    if relabel_nodes:
        branch_labels = nx.get_node_attributes(branch_graph, "global_branch_index")
        branch_labels = {n: i for n, i in branch_labels.items() if pd.notna(i)}
        branch_graph = nx.relabel_nodes(branch_graph, branch_labels)
    return branch_graph


def _compute_branch_parents(graph: nx.Graph) -> Dict[Any, list[int]]:
    """Computes the parent branches for each branch in a compartment graph.

    Each cell is rooted at its lowest `branch_index`.

    Args:
        graph: The compartment graph without branchpoints and tips.

    Returns:
        The parent branches for each branch in the branch graph, keyed by `cell_index`
        (or by `None` if the graph has no `cell_index`).
    """

    # This requires the branchpoints and tips to have been contracted away already,
    # which `from_graph()` does.
    branch_graph = comp_to_branch_graph(graph)
    branch_df = nx_to_pandas(branch_graph)[0]

    cell_dfs = (
        branch_df.groupby("global_cell_index")
        if "global_cell_index" in branch_df.columns
        else [(None, branch_df)]
    )

    acc_parents = {}
    for cell_id, cell_df in cell_dfs:
        branch_graph_of_cell = branch_graph.subgraph(
            cell_df["global_branch_index"].values
        )

        branch_inds = sorted(cell_df["global_branch_index"].unique())
        pos_of_branch = {idx: pos for pos, idx in enumerate(branch_inds)}

        parent_list = [-1] * len(branch_inds)
        for parent, child in nx.bfs_edges(branch_graph_of_cell, np.min(branch_inds)):
            parent_list[pos_of_branch[child]] = pos_of_branch[parent]
        acc_parents[cell_id] = parent_list
    return acc_parents


def _build_module(graph: nx.Graph, assign_groups: bool = True) -> Module:
    """Build a Module from a compartmentalized morphology.

    This function builds a Module from a nx.Graph that has been compartmentalized.

    Args:
        graph: The graph to build the Module from.
        assign_groups: Whether to assign groups to the compartments based on their id.

    Returns:
        The Module built from the graph.
    """
    # TODO: the module attributes assigned at the end of this function duplicate what
    # `Module.insert()` and `_append_multiple_synapses()` do, so they drift silently when
    # those change. Route channels, synapses and pumps through those instead, then
    # re-initialize params and states.
    node_df, edge_df, global_attrs = nx_to_pandas(graph)

    # ensure edges in edges are always from smaller index to larger index
    if len(edge_df) > 0:
        inds = np.stack(edge_df.index)
        new_inds = np.where(
            (inds[:, 0] < inds[:, 1])[:, np.newaxis], inds, inds[:, ::-1]
        )
        edge_df.index = pd.MultiIndex.from_tuples(new_inds.tolist())

    synapse_edge_df = edge_df[edge_df.synapse == True if len(edge_df) > 0 else []]
    synapse_edge_df = synapse_edge_df.reset_index(names=["pre_index", "post_index"])
    synapse_edge_df = synapse_edge_df.drop(columns=["synapse"], errors="ignore")
    # These come out of the graph as `object`, because there they also hold the `pd.NA` of
    # the compartment edges. Only synapse rows are left here, so they can be cast back.
    int_cols = [
        "global_edge_index",
        "index_within_type",
        "type_ind",
        "controlled_by_param",
    ]
    int_cols = [col for col in int_cols if col in synapse_edge_df.columns]
    synapse_edge_df[int_cols] = synapse_edge_df[int_cols].astype(int)

    acc_parents = _compute_branch_parents(graph)
    return_type = global_attrs["module"] if "module" in global_attrs else "cell"

    module = _build_module_scaffold(
        node_df,
        parent_branches=acc_parents,
        xyzr=global_attrs["xyzr"],
        return_type=return_type,
    )

    # jaxley expects contiguous indices, but since we drop branchpoints in
    # _extract_branchpoints, we need to re-assign the indices here
    node_df.index = module.nodes.index

    # set column-wise. preserves cols not in df.
    if not assign_groups:
        node_df = node_df.drop(columns=global_attrs["group_names"], errors="ignore")
        global_attrs["group_names"] = []

    module.nodes[node_df.columns] = node_df
    module.edges = synapse_edge_df if not synapse_edge_df.empty else module.edges

    # add all the extra attrs
    module.synapses = global_attrs["synapses"]
    module.channels = global_attrs["channels"]
    module.pumps = global_attrs["pumps"]
    module.pumped_ions = [p.ion_name for p in module.pumps]
    module.group_names = global_attrs["group_names"]
    module.synapse_current_names = [f"i_{s._name}" for s in module.synapses]
    module.synapse_param_names = [
        k for s in module.synapses for k in s.synapse_params.keys()
    ]
    module.synapse_state_names = [
        k for s in module.synapses for k in s.synapse_states.keys()
    ]
    module.membrane_current_names = [c.current_name for c in module.channels]
    module.synapse_names = [s._name for s in module.synapses]

    return module


def from_graph(
    comp_graph: nx.Graph,
    assign_groups: bool = True,
):
    """Return a Jaxley module from a compartmentalized networkX graph.

    Args:
        comp_graph: The compartment graph built with `build_compartment_graph()` or
            with `to_graph()`.
        assign_groups: Whether to assign groups to the compartments based on their id.

    Return:
        A `jx.Module` representing the graph.

    Example usage
    ^^^^^^^^^^^^^

    ::

        from jaxley.io.graph import build_compartment_graph, from_graph
        comp_graph = build_compartment_graph(swc_graph, ncomp=1)
        cell = from_graph(comp_graph)
    """

    if not "channels" in comp_graph.graph:
        comp_graph = _add_jaxley_meta_data(comp_graph)
    if any(pd.isna(d["global_branch_index"]) for _, d in comp_graph.nodes(data=True)):
        comp_graph = _extract_branchpoints(comp_graph)
    module = _build_module(comp_graph, assign_groups=assign_groups)
    return module


def _build_module_scaffold(
    idxs: pd.DataFrame,
    return_type: str = "cell",
    parent_branches: Optional[Dict[Any, List[int]]] = None,
    xyzr: Optional[List[np.ndarray]] = None,
) -> Union[Network, Cell, Branch, Compartment]:
    """Builds a skeleton module from a DataFrame of indices.

    This is useful for instantiating a module that can be filled with data later.

    Args:
        idxs: DataFrame containing the global cell, branch and compartment index, i.e.
            Module.nodes or View.view.
        return_type: Type of module to return. If None, the type is inferred from the
            number of unique values in the indices. I.e. only 1 unique cell_index
                and 1 unique branch_index -> return_type = "jx.Branch".
        parent_branches: Branch parents for each cell, keyed by `cell_index`.
        xyzr: List of xyzr arrays for each branch.

    Returns:
        A skeleton module with the correct number of compartments, branches, cells, or
        networks."""
    # NOTE: The first call takes much longer, because building the first `Compartment`
    # warms up JAX.
    xyzr = [] if xyzr is None else xyzr
    build_cache = {k: [] for k in ["compartment", "branch", "cell", "network"]}

    comp = Compartment()
    build_cache["compartment"] = [comp]

    # Number of comps of each branch, ordered by branch_index.
    ncomp_per_branch = idxs.groupby("global_branch_index").size().sort_index()

    if return_type in ["branch", "cell", "network"]:
        # One `Branch` per *distinct* ncomp, reused across branches
        branch_of_ncomp = {
            n: Branch([comp for _ in range(n)]) for n in set(ncomp_per_branch)
        }
        build_cache["branch"] = [branch_of_ncomp[ncomp_per_branch.iloc[0]]]

    if return_type in ["cell", "network"]:
        branch_counter = 0
        for cell_id, cell_groups in idxs.groupby("global_cell_index"):
            branch_inds = sorted(cell_groups["global_branch_index"].unique())
            num_branches = len(branch_inds)
            default_parents = np.arange(num_branches) - 1  # ignores morphology
            cell = Cell(
                [branch_of_ncomp[ncomp_per_branch[b]] for b in branch_inds],
                parents=(
                    default_parents
                    if parent_branches is None
                    else parent_branches[cell_id]
                ),
                xyzr=xyzr[branch_counter : branch_counter + num_branches],
            )
            build_cache["cell"].append(cell)
            branch_counter += num_branches

    if return_type == "network":
        build_cache["network"] = [Network(build_cache["cell"])]

    module = build_cache[return_type][0]
    build_cache.clear()
    return module


def _branchpoints_and_tips_of(
    module: Module,
) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
    """Rebuild the branchpoint and tip nodes of a module's compartment graph.

    Branchpoints and tips only carry `x`, `y`, `z` and a radius, and each of them sits on an
    endpoint of a branch in `module.xyzr`.

    Only the names come from `_branchpoints`, not the coordinates:
    `compute_compartment_centers()` overwrites its `x`, `y`, `z` with the mean of the
    neighbouring compartment centers, which moves a branchpoint off the branch end it
    belongs to.

    Args:
        module: The module to read the branch structure and `xyzr` of.

    Returns:
        A DataFrame of the branchpoint and tip attributes, indexed by node name, and the
        list of `(tip, compartment)` edges. Branchpoints keep the names they have in
        `_comp_edges`, which already holds their edges; the tips are named after them.
    """
    parents = np.asarray(module.comb_parents)
    par_inds = np.asarray(module._par_inds, dtype=int)

    comps_of_branch = module.nodes.index.to_series().groupby(
        module.nodes["global_branch_index"].to_numpy()
    )
    first_comp, last_comp = comps_of_branch.min(), comps_of_branch.max()

    rows, tip_edges = {}, []
    for name, branch in zip(module._branchpoints.index, par_inds):
        rows[name] = dict(zip(["x", "y", "z", "radius"], module.xyzr[branch][-1, :4]))

    # Tips are named after the branchpoints, which end at `_n_nodes`.
    has_child = np.zeros(len(parents), dtype=bool)
    has_child[par_inds] = True
    next_name = int(module._n_nodes)
    for branch in range(len(parents)):
        xyzr = module.xyzr[branch]
        one_ended = np.allclose(xyzr[0, :3], xyzr[-1, :3])
        ends = []
        # If the two ends coincide and the branch has a parent, the node at the far end is
        # the parent's branchpoint.
        if not has_child[branch] and not (one_ended and parents[branch] != -1):
            ends.append((xyzr[-1], last_comp[branch]))
        if parents[branch] == -1 and not one_ended:
            ends.append((xyzr[0], first_comp[branch]))
        for end, comp in ends:
            rows[next_name] = dict(zip(["x", "y", "z", "radius"], end[:4]))
            tip_edges.append((next_name, int(comp)))
            next_name += 1

    return pd.DataFrame.from_dict(rows, orient="index"), tip_edges


def to_graph(module: Module, channels: bool = True, synapses: bool = True) -> nx.Graph:
    """Convert a Module to a compartment graph.

    The graph has one node per compartment, plus the branchpoint and tip nodes that
    `build_compartment_graph()` also emits. Compartment attributes are read off
    `module.nodes`, so any changes made to them are kept, while the branchpoints and
    tips are rebuilt from `module.xyzr`, see `_branchpoints_and_tips_of()`.

    Args:
        module: The Module to convert.
        channels: Whether to carry the channels and their parameters and states. Set to
            `False` for a graph of the bare morphology.
        synapses: Whether to carry the synapses and their parameters and states.

    Returns:
        A NetworkX graph. Can be read back with `from_graph()`, which reproduces `module`
        exactly as long as both flags are left at `True`.
    """
    branchpoints, tip_edges = _branchpoints_and_tips_of(module)

    edges = module._comp_edges
    condition1 = edges["type"].isin([2, 3])
    condition2 = edges["type"] == 0
    condition3 = edges["source"] < edges["sink"]
    edges = edges[condition1 | (condition3 & condition2)][["source", "sink"]]
    edges = pd.concat(
        [edges, pd.DataFrame(tip_edges, columns=["source", "sink"])], ignore_index=True
    )
    edges.set_index(["source", "sink"], inplace=True)
    edges.index.names = (None, None)
    edges["synapse"] = False

    if synapses:
        synapse_edges = module.edges.set_index(["pre_index", "post_index"], drop=True)
        synapse_edges.index.names = (None, None)
        synapse_edges["synapse"] = True
        edges = edges.combine_first(synapse_edges)

    int_cols = [
        "global_edge_index",
        "index_within_type",
        "type_ind",
        "controlled_by_param",
    ]
    int_cols = [col for col in int_cols if col in edges.columns]
    edges[int_cols] = edges[int_cols].astype(pd.Int64Dtype())
    edges["synapse"] = edges["synapse"].astype(bool)

    nodes = module.nodes.drop(
        columns=[col for col in module.nodes.columns if "local_" in col]
    )
    nodes = nodes.drop(["controlled_by_param"], axis=1)
    if not channels:
        # A channel owns one boolean column plus one per parameter and state.
        dropped = [c._name for c in module.channels]
        dropped += [p for c in module.channels for p in c.channel_params]
        dropped += [s for c in module.channels for s in c.channel_states]
        nodes = nodes.drop(columns=dropped, errors="ignore")
    nodes["global_branch_index"] = nodes["global_branch_index"].astype(pd.Int64Dtype())
    nodes["global_cell_index"] = nodes["global_cell_index"].astype(pd.Int64Dtype())
    nodes["global_comp_index"] = nodes["global_comp_index"].astype(pd.Int64Dtype())
    nodes = nodes.combine_first(branchpoints)

    module_type = infer_module_type(module)
    graph = pandas_to_nx(nodes, edges, pd.Series({"module": module_type}))

    # Copy ensures that rebuilt module does not share lists with `module`.
    for attr in ["xyzr", "channels", "synapses", "group_names", "pumps"]:
        graph.graph[attr] = getattr(module, attr).copy()
    if not channels:
        graph.graph["channels"] = []
    if not synapses:
        graph.graph["synapses"] = []

    return graph
