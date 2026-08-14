# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import networkx as nx
import numpy as np
import pandas as pd

from jaxley.io.graph import from_graph, nx_to_pandas, pandas_to_nx, to_graph


def _assert_editable(view, caller: str) -> None:
    """Check that a view's module may be reshaped, and give it coordinates if it has none.

    Both `morph_delete()` and `morph_connect()` rebuild the module from its compartment
    graph, which carries no simulation state and no `xyzr` for a module built from scratch.
    """
    module = view.base
    assert module.__class__.__name__ == "Cell", (
        f"You are trying to use `{caller}()` on a `jx.{module.__class__.__name__}`. "
        "Only `jx.Cell` is allowed."
    )
    for state, label, fix in [
        ("rec_info", "recordings", "delete_recordings"),
        ("externals", "external states (stimuli or clamps)", "delete_stimuli()` or `cell.delete_clamps"),
        ("trainable_params", "trainable parameters", "delete_trainables"),
    ]:
        found = len(getattr(module, state))
        assert found == 0, (
            f"Found {found} {label}. This is not supported. "
            f"Please run `cell.{fix}()`."
        )

    # If the user did not run `compute_xyz` or `compute_compartment_centers`, we run it
    # automatically.
    if np.isnan(module.xyzr[0][0, 0]):
        module.compute_xyz()
    if "x" not in module.nodes.columns:
        module.compute_compartment_centers()


def morph_delete(module_view) -> "Cell":
    """Deletes part of a morphology.

    This function can only delete entire branches. It does not support deleting
    compartments of a branch.

    This function deletes all existing recordings, stimuli, trainable parameters, and
    channels.

    Args:
        module_view: View of a `jx.Cell`. Defines the branches to be deleted.

    Returns:
        A cell in which specified branches are deleted.

    Example usage
    ^^^^^^^^^^^^^

    ::

        cell = jx.read_swc("path_to_swc_file.swc", ncomp=1)
        cell = morph_delete(cell.axon)

    ::

        cell = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])
        cell = morph_delete(cell.branch([3, 4]))
    """
    _assert_editable(module_view, "morph_delete")

    comps_to_delete = module_view.nodes.index
    comp_graph = to_graph(module_view.base)

    nodes_to_keep = []
    for node, data in comp_graph.nodes(data=True):
        # Keep every branchpoint and tip; they are the nodes without a `branch_index`. The
        # ones left with a single neighbour are contracted away by `_extract_branchpoints()`
        # during `from_graph()`.
        if (
            pd.isna(data["global_branch_index"])
            or data["global_comp_index"] not in comps_to_delete
        ):
            nodes_to_keep.append(node)

    comp_graph = nx.subgraph(comp_graph, nodes_to_keep)
    return from_graph(_renumber_branches(comp_graph))


def _renumber_branches(graph: nx.Graph) -> nx.Graph:
    """Make branch and compartment indices contiguous again, and drop the vanished `xyzr`.

    Removing or merging compartments leaves gaps, and `from_graph()` uses `branch_index` to
    index the graph's `xyzr` list positionally, so the two have to be brought back in step.
    """
    nodes, edges, attrs = nx_to_pandas(graph)
    branches = sorted(nodes["global_branch_index"].dropna().unique())
    nodes["global_branch_index"] = nodes["global_branch_index"].map(
        {branch: i for i, branch in enumerate(branches)}
    )
    is_comp = nodes["global_branch_index"].notna()
    nodes.loc[is_comp, "global_comp_index"] = np.arange(is_comp.sum())
    for column in ["global_branch_index", "global_comp_index", "global_cell_index"]:
        nodes[column] = nodes[column].astype(pd.Int64Dtype())

    graph = pandas_to_nx(nodes, edges, attrs)
    graph.graph["xyzr"] = [attrs["xyzr"][int(branch)] for branch in branches]
    return graph


def morph_connect(module_view1, module_view2) -> "Cell":
    """Combines two morphologies into a single cell.

    This function deletes all existing recordings, stimuli, and trainable parameters.

    Args:
        module_view1: View of a ``jx.Cell()``. Must have been created with a
            command ending on ``loc(0.0)`` or ``loc(1.0)``. For example, the following
            are valid:
            ``cell.branch(0).loc(0.0)``, ``cell.branch(5).loc(0.0)``,
            ``cell.branch(5).loc(1.0)``.
            But those are not valid:
            ``cell.branch(0).comp(0)`` (uses ``.comp``), ``cell.branch(5).loc(0.9)``
            (does not use ``loc(0.0)`` or ``loc(1.0)``).
        module_view2: The view of a ``jx.Cell()``. Must follow the same rules as
            ``module_view1``.

    Returns:
        A ``jx.Cell`` which is made up of both input cells.

    Example usage
    ^^^^^^^^^^^^^

    ::

        cell = jx.read_swc("path_to_swc_file.swc", ncomp=1)
        stub = jx.Cell()
        cell = morph_connect(cell.branch(0).loc(0.0), stub.branch(0).loc(0.0))
    """
    for view in [module_view1, module_view2]:
        _assert_editable(view, "morph_connect")

    graph1 = to_graph(module_view1.base)
    graph2 = to_graph(module_view2.base)
    node1 = _connection_node(module_view1, graph1)
    node2 = _connection_node(module_view2, graph2)
    return from_graph(_join_graphs(graph1, graph2, node1, node2))


def _connection_node(view, graph: nx.Graph) -> int:
    """The node of `graph` at the branch end that `view` selects.

    Every branch end is a node of the compartment graph: a branchpoint if something already
    attaches there, a tip otherwise. So connecting two cells is merging one node of each.

    A view's `_comp_edges` holds the branchpoint edges of the end it selects, and is empty
    for a free end, which is how `loc(0.0)` and `loc(1.0)` are told apart.
    """
    comp = int(view.nodes.index[0])
    branchpoints = view._comp_edges["sink"]
    if len(branchpoints) > 0:
        return int(branchpoints.max())

    tips = [
        n
        for n in graph.neighbors(comp)
        if pd.isna(graph.nodes[n]["global_branch_index"]) and graph.degree(n) == 1
    ]
    assert tips, f"Compartment {comp} has no free end to connect to."
    return min(tips)


def _join_graphs(
    graph1: nx.Graph, graph2: nx.Graph, node1: int, node2: int
) -> nx.Graph:
    """Return one compartment graph joining `graph1` and `graph2` at the given nodes.

    `graph2` is translated so that `node2` lands on `node1`, and the two are then merged
    into a single node. Whether that node ends up a branchpoint or stays a tip follows from
    how many compartments it connects, so no case distinction is needed.
    """
    nodes1, edges1, attrs1 = nx_to_pandas(graph1)
    nodes2, edges2, attrs2 = nx_to_pandas(graph2)

    # Move graph2 so the two connection points coincide, coordinates and traced points.
    offset = nodes1.loc[node1, ["x", "y", "z"]].to_numpy(float) - nodes2.loc[
        node2, ["x", "y", "z"]
    ].to_numpy(float)
    nodes2[["x", "y", "z"]] += offset
    xyzr2 = [branch.copy() for branch in attrs2["xyzr"]]
    for branch in xyzr2:
        branch[:, :3] += offset

    # Renumber graph2 so nothing collides: its nodes, and their indices.
    node_offset = int(nodes1.index.max()) + 1
    nodes2.index += node_offset
    edges2.index = pd.MultiIndex.from_tuples(
        [(u + node_offset, v + node_offset) for u, v in edges2.index]
    )
    nodes2["global_branch_index"] += int(nodes1["global_branch_index"].max()) + 1
    nodes2["global_comp_index"] += int(nodes1["global_comp_index"].max()) + 1
    node2 += node_offset

    nodes = pd.concat([nodes1, nodes2])
    edges = pd.concat([edges1, edges2])
    nodes["global_cell_index"] = 0  # the result is a single cell

    for column in ["global_branch_index", "global_comp_index", "global_cell_index"]:
        nodes[column] = nodes[column].astype(pd.Int64Dtype())

    # A group or channel that only one of the two cells has is absent from the other's
    # compartments.
    for column in set(attrs1["group_names"]) | set(attrs2["group_names"]):
        nodes[column] = nodes[column].fillna(False).astype(bool)
    for channel in attrs1["channels"] + attrs2["channels"]:
        nodes[channel._name] = nodes[channel._name].fillna(False).astype(bool)

    # Merge the two connection points into one node, keeping graph1's.
    nodes = nodes.drop(index=node2)
    edges.index = pd.MultiIndex.from_tuples(
        [
            (node1 if u == node2 else u, node1 if v == node2 else v)
            for u, v in edges.index
        ]
    )
    edges = edges[[u != v for u, v in edges.index]]

    graph = pandas_to_nx(nodes, edges, attrs1)
    graph.graph["xyzr"] = list(attrs1["xyzr"]) + xyzr2
    for key in ["channels", "synapses", "pumps"]:
        merged = {type(m): m for m in attrs1[key] + attrs2[key]}
        graph.graph[key] = list(merged.values())
    graph.graph["group_names"] = list(
        dict.fromkeys(attrs1["group_names"] + attrs2["group_names"])
    )
    return _renumber_branches(graph)
