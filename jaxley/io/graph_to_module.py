# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from jaxley.modules import Branch, Cell, Compartment, Network
from jaxley.io.graph import _build_solve_graph, _add_meta_data
from jaxley.utils.solver_utils import reorder_dhs


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

    Example:
    --------

    ::

        from jaxley.io.graph import to_swc_graph, build_compartment_graph, from_graph
        swc_graph = to_swc_graph("path_to_swc.swc")
        comp_graph = build_compartment_graph(swc_graph, ncomp=1)
        cell = from_graph(comp_graph)
    """
    solve_graph, _, _ = _build_solve_graph(
        comp_graph, root=solve_root, traverse_for_solve_order=traverse_for_solve_order
    )
    solve_graph = _add_meta_data(solve_graph)
    return _build_module(solve_graph, comp_graph, assign_groups=assign_groups)


def _build_module(
    solve_graph: nx.DiGraph,
    comp_graph: nx.DiGraph,
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
                "pre_global_comp_index": i,
                "post_global_comp_index": j,
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

