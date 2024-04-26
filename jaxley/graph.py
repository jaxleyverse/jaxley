from math import pi
from typing import Dict, List, Optional, Union

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd
import networkx as nx

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
        build_cache["cell"].clear()  # clear build cache
        for branch_id, branch_groups in cell_groups.groupby("branch_index"):
            build_cache["branch"].clear()  # clear build cache
            comps = [jx.Compartment() for _ in range(len(branch_groups))]
            build_cache["branch"].append(jx.Branch(comps))

        parents = np.arange(len(build_cache["branch"])) - 1  # will be overwritten later
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


def to_graph(module) -> nx.DiGraph:
    """Export the module as a networkx graph.

    Constructs a nx.DiGraph from the module. Each compartment in the module
    is represented by a node in the graph. The edges between the nodes represent
    the connections between the compartments. These edges can either be connections
    between compartments within the same branch, between different branches or
    even between different cells. In this case the latter the synapse parameters
    are stored as edge attributes. Additionally, global attributes of the module,
    for example `nseg`, are stored as graph attributes.

    Exported graphs can be imported again to `jaxley` using the `from_graph` method.

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
    # ensure multiple groups allowed per node
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


def from_graph(
    module_graph: nx.DiGraph,
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

    Args:
        module_graph: A networkx graph representing a module.

    Returns:
        A module instance that is populated with the node and egde attributes of
        the nx.DiGraph."""
    global_attrs = module_graph.graph.copy()

    nodes = pd.DataFrame((n for i, n in module_graph.nodes(data=True)))
    optional_attrs = ["recordings", "currents", "groups", "trainables"]
    nodes.drop(
        columns=optional_attrs, inplace=True, errors="ignore"
    )  # ignore if columns do not exist

    # build skeleton of module instances
    return_type = (
        global_attrs.pop("module").lower() if "module" in global_attrs else None
    )
    module = build_skeleton_module(nodes, return_type)

    # set global attributes of module
    for k, v in global_attrs.items():
        setattr(module, k, v)

    # import nodes and edges
    edge_type = nx.get_edge_attributes(module_graph, "type")
    edges = pd.DataFrame(edge_type.values(), index=edge_type.keys(), columns=["type"])
    is_synapse = edges["type"] != "branch"
    idxs = edges[~is_synapse].reset_index().values.T[:2]
    branch_edges = pd.DataFrame([nodes["branch_index"].loc[i].values for i in idxs]).T
    branch_edges.columns = ["parent_branch_index", "child_branch_index"]

    synapse_edges = pd.DataFrame(
        [module_graph.edges[idxs] for idxs in edges[is_synapse].index]
    )

    module.nodes = nodes
    module.branch_edges = branch_edges
    module.edges = synapse_edges

    # get special attrs from nodes
    recordings = pd.DataFrame(nx.get_node_attributes(module_graph, "recordings"))
    currents = pd.DataFrame(nx.get_node_attributes(module_graph, "currents"))
    groups = pd.DataFrame(nx.get_node_attributes(module_graph, "groups").items())
    trainables = pd.DataFrame(
        nx.get_node_attributes(module_graph, "trainable"), dtype=float
    )

    if not recordings.empty:
        recordings = recordings.T.unstack().reset_index().set_index("level_0")
        recordings = recordings.rename(columns={"level_1": "rec_index", 0: "state"})
        module.recordings = recordings
    if not currents.empty:
        current_inds = nodes.loc[currents.T.index]
        currents = np.stack(currents.values)
        module.currents = currents
        module.current_inds = current_inds
    if not groups.empty:
        groups = groups.explode(1).rename(columns={0: "index", 1: "group"})
        group_nodes = {k: nodes.loc[v["index"]] for k, v in groups.groupby("group")}
        module.group_nodes = group_nodes
    if not trainables.empty:
        trainables = pd.DataFrame(
            nx.get_node_attributes(module_graph, "trainable"), dtype=float
        )
        trainables = trainables.T.unstack().reset_index().dropna()
        trainables = trainables.rename(
            columns={"level_0": "param", "level_1": "index", 0: "value"}
        )
        grouped_trainables = trainables.groupby(["param", "value"])
        merged_trainables = grouped_trainables.agg(
            {"index": lambda x: jnp.array(x.values)}
        ).reset_index()
        concat_here = merged_trainables["index"].apply(lambda x: len(x) == 1)
        common_trainables = merged_trainables.loc[~concat_here]
        diff_trainables = (
            merged_trainables.loc[concat_here].groupby("param").agg(list).reset_index()
        )
        diff_trainables.loc[:, "index"] = diff_trainables["index"].apply(jnp.stack)
        diff_trainables.loc[:, "value"] = diff_trainables["value"].apply(jnp.array)
        common_trainables.loc[:, "value"] = common_trainables["value"].apply(np.array)
        trainable_params = pd.concat([common_trainables, diff_trainables])
        indices_set_by_trainables = trainable_params["index"].values
        trainable_params = [
            {k: jnp.array(v).reshape(-1)}
            for k, v in trainable_params[["param", "value"]].values
        ]

        module.trainable_params = trainable_params
        module.indices_set_by_trainables = indices_set_by_trainables

    return module
