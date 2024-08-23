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


def infer_module_type_from_inds(idxs: pd.DataFrame) -> str:
    nuniques = idxs.nunique()[["cell_index", "branch_index", "comp_index"]]
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
    build_cache["compartment"] = comp

    if return_type in return_types[1:]:
        nsegs = idxs["branch_index"].value_counts().iloc[0]
        branch = jx.Branch([comp for _ in range(nsegs)])
        build_cache["branch"] = branch

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
    module_graph.graph["type"] = module.__class__.__name__
    for attr in [
        "nseg",
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
    module_graph.add_nodes_from(
        module.nodes.to_dict("index").items()
    )  # preserves dtypes

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
        module_graph.add_edges_from(
            zip(group.index[:-1], group.index[1:]), type="intra_branch"
        )

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
        module_graph.add_edge(parent_comp_idx, child_comp_idx, type="inter_branch")

    # connect synapses
    for index, edge in module.edges.iterrows():
        attrs = edge.to_dict()
        pre = attrs["global_pre_comp_index"]
        post = attrs["global_post_comp_index"]
        module_graph.add_edge(pre, post, type="synapse")
        # allow for multiple synapses between the same compartments
        if "parameters" in module_graph.edges[pre, post]:
            module_graph.edges[pre, post]["parameters"].append(attrs)
        else:
            module_graph.edges[pre, post]["parameters"] = [attrs]

    return module_graph


# helper functions
is_leaf = lambda graph, n: graph.out_degree(n) == 0 and graph.in_degree(n) == 1
is_branching = lambda graph, n: graph.out_degree(n) > 1
has_same_id = lambda graph, i, j: graph.nodes[i]["id"] == graph.nodes[j]["id"]
path_e2n = lambda path: (
    [path[0][0]] + [e[1] for e in path]
    if path.shape[0] > 1
    else [path[0, 0], path[0, 1]]
)
path_n2e = lambda path: [e for e in zip(path[:-1], path[1:])]
unpack = lambda d, keys: [d[k] for k in keys]
# has_node_attr = lambda graph, attr: all(attr in graph.nodes[n] for n in graph.nodes)
# has_edge_attr = lambda graph, attr: all(attr in graph.edges[e] for e in graph.edges)
# v_interp = vmap(jnp.interp, in_axes=(None, None, 1))


def get_paths(graph, source_node=0):
    """Get all linear paths in a graph.

    The graph is traversed depth-first starting from the root node.

    Args:
        graph: A networkx graph.
        source_node: node at which to start graph traversal. If "leaf", the traversal
            starts at the first identified leaf node.

    Returns:
        A list of linear paths in the graph. Each path is represented as an array of
        edges."""
    paths, current_path = [], []
    leaf_idx = next(
        x for x in graph.nodes() if graph.out_degree(x) == 0 and graph.in_degree(x) == 1
    )
    # starting from leaf node avoids need to reconnect multiple root nodes later. 
    # graph needs to be undirected for this to work
    source = leaf_idx if source_node == "leaf" else source_node
    for i, j in nx.dfs_edges(graph.to_undirected(), source):
        current_path += [(i, j)]
        if is_leaf(graph, j) or is_branching(graph, j):
            paths.append(current_path)
            current_path = []
        elif not has_same_id(graph, i, j): # start new path if ids differ
            paths.append(current_path[:-1])
            current_path = [current_path[-1]]
    return [np.array(p) for p in paths if len(p) > 0]


def add_edge_lens(graph):
    for i, j in graph.edges:
        xyz_i = np.array(unpack(graph.nodes[i], ["x", "y", "z"]))
        xyz_j = np.array(unpack(graph.nodes[j], ["x", "y", "z"]))
        d_ij = np.sqrt(((xyz_i - xyz_j) ** 2).sum())
        graph.edges[i, j]["l"] = d_ij
    return graph


def split_paths(paths, edge_lens, max_len=100):
    # TODO: split paths into equally long sections
    edge_lens.update({(j, i): l for (i, j), l in edge_lens.items()})
    new_paths = []
    for path in paths:
        cum_path_len = np.cumsum([edge_lens[*e] for e in path])
        k = cum_path_len // max_len
        split_path = [path[np.where(np.array(k) == kk)[0]] for kk in np.unique(k)]
        new_paths += split_path
    return new_paths


def make_jaxley_compatible(graph, nseg=8, max_branch_len=2000, source_node=0):
    graph = add_edge_lens(graph)
    paths = get_paths(graph, source_node=source_node)

    # add source compartment with length = 0
    # if source_node != "leaf":
    #     paths = [np.array([[source_node,source_node]])] + paths
    #     graph.add_edge(source_node,source_node)
    #     graph.edges[source_node,source_node]["l"] = 0.01 #graph.nodes[source_node]["r"]

    edge_lens = nx.get_edge_attributes(graph, "l")
    branch_edges = split_paths(paths, edge_lens, max_branch_len)
    branch_nodes = [path_e2n(b) for b in branch_edges]

    jaxley_branches = pd.DataFrame(
        sum(
            [
                [{**graph.nodes[n], "branch_index": i} for n in nodes]
                for i, nodes in enumerate(branch_nodes)
            ],
            [],
        )
    )
    for idx, group in jaxley_branches.groupby("branch_index"):
        if (
            jaxley_branches.loc[group.index[0], "id"]
            != jaxley_branches.loc[group.index[1], "id"]
        ):
            jaxley_branches.loc[group.index[0], ["id", "r"]] = jaxley_branches.loc[
                group.index[-1], ["id", "r"]
            ]

    edge_lens = nx.get_edge_attributes(graph, "l")
    edge_lens.update({(j, i): l for (i, j), l in edge_lens.items()})
    jaxley_branches["l"] = np.hstack(
        [np.cumsum([0] + [edge_lens[*e] for e in branch]) for branch in branch_edges]
    )

    # compartmentalize branches
    jaxley_comps = []
    keys = ["x", "y", "z", "r", "id"]
    for idx, nodes in jaxley_branches.groupby("branch_index"):
        branch_len = nodes["l"].max()
        comp_len = branch_len / nseg
        locs = np.linspace(comp_len / 2, branch_len - comp_len / 2, nseg)
        comp_attrs = v_interp(locs, nodes["l"].values, nodes[keys].values)

        comp_attrs = pd.DataFrame(comp_attrs.T, columns=keys)
        comp_attrs["id"] = comp_attrs["id"].astype(int)
        comp_attrs["length"] = comp_len
        comp_attrs["branch_index"] = idx

        jaxley_comps.append(comp_attrs)
    jaxley_comps = pd.concat(jaxley_comps, ignore_index=True)

    jaxley_comps.rename(columns={"r": "radius"}, inplace=True)
    jaxley_comps["comp_index"] = np.arange(jaxley_comps.shape[0])
    jaxley_comps["cell_index"] = 0

    group_ids = {0: "undefined", 1: "soma", 2: "axon", 3: "basal", 4: "apical"}
    id2group = lambda x: [group_ids[x] if x < 5 else f"custom{x}"]
    jaxley_comps["groups"] = jaxley_comps["id"].apply(id2group)

    branch_roots_and_leafs = np.stack(
        [np.array(branch)[[0, -1]] for branch in branch_nodes]
    )
    parent_equal_child = np.equal(*np.meshgrid(*(branch_roots_and_leafs.T)))
    # edges_between_branches = np.stack(list(zip(*np.where(parent_equal_child))))[1:] # if source_node != "leaf"
    edges_between_branches = np.stack(list(zip(*np.where(parent_equal_child))))

    comps_in_branches = jaxley_comps.groupby("branch_index")["comp_index"]
    intra_branch_edges = sum([path_n2e(c) for i, c in comps_in_branches], [])

    branch_roots = comps_in_branches.first().values
    branch_leafs = comps_in_branches.last().values

    inter_branch_children = branch_roots[edges_between_branches[:, 1]]
    inter_branch_parents = branch_leafs[edges_between_branches[:, 0]]
    inter_branch_edges = np.stack([inter_branch_parents, inter_branch_children]).T

    source_branches = np.where(branch_roots_and_leafs[:, 0] == source_node)[0][1:]
    source_branch_edges = np.stack(
        [source_node * np.ones_like(source_branches), source_branches]
    ).T
    inter_source_children = branch_roots[source_branch_edges[:, 1]]
    inter_source_parents = branch_leafs[source_branch_edges[:, 0]]
    inter_source_edges = np.stack([inter_source_parents, inter_source_children]).T

    comp_graph = nx.DiGraph()
    comp_graph.add_edges_from(inter_branch_edges, type="inter_branch")
    comp_graph.add_edges_from(intra_branch_edges, type="intra_branch")
    comp_graph.add_edges_from(inter_source_edges, type="inter_branch")

    keys = [
        "radius",
        "length",
        "x",
        "y",
        "z",
        "comp_index",
        "branch_index",
        "cell_index",
        "groups",
    ]
    comp_graph.add_nodes_from(
        (
            (n, {k: v for k, v in zip(keys, vals)})
            for i, (n, *vals) in jaxley_comps[["comp_index"] + keys].iterrows()
        )
    )
    comp_graph.graph["nseg"] = nseg
    comp_graph.graph["xyzr"] = [
        n[["x", "y", "z", "r"]].values
        for i, n in jaxley_branches.groupby("branch_index")
    ]
    # comp_graph.graph["type"] = infer_module_type_from_inds(jaxley_comps[["cell_index", "branch_index", "comp_index"]])
    comp_graph.graph["type"] = "cell"
    return comp_graph


def from_graph(
    graph: nx.DiGraph,
    nseg: int = 4,
    max_branch_len: float = 300.0,
    assign_groups: bool = True,
) -> Union[jx.Network, jx.Cell, jx.Branch, jx.Compartment]:
    """Build a module from a networkx graph.

    All information stored in the nodes of the graph is imported into the Module.nodes
    attribute. Edges of type "inter_branch" connect two different branches, while
    edges of type "intra_branch" connect compartments within the same branch. All other
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
    the graph has to be connected and cannot contain loops. Currently networks can
    only be imported if they have been previously exported with jaxley!

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

    Returns:
        A module instance that is populated with the node and egde attributes of
        the nx.DiGraph."""

    ########################################
    ### Make the graph jaxley compatible ###
    ########################################

    if graph.graph["type"] == "swc":
        graph = make_jaxley_compatible(graph, nseg=nseg, max_branch_len=max_branch_len)

    #################################
    ### Import graph as jx.Module ###
    #################################

    # import nodes and edges
    nodes = pd.DataFrame((n for i, n in graph.nodes(data=True)))
    nodes = nodes.sort_values(
        "comp_index", ignore_index=True
    )  # ensure index == comp_index
    edge_type = nx.get_edge_attributes(graph, "type")
    edges = pd.DataFrame(edge_type.values(), index=edge_type.keys(), columns=["type"])
    edges = edges.reset_index(names=["pre", "post"])
    is_synapse = edges["type"] == "synapse"
    is_inter_branch = edges["type"] == "inter_branch"
    inter_branch_edges = edges.loc[is_inter_branch][["pre", "post"]].values
    synapse_edges = edges.loc[is_synapse][["pre", "post"]].values
    branch_edges = pd.DataFrame(
        nodes["branch_index"].values[inter_branch_edges],
        columns=["parent_branch_index", "child_branch_index"],
    )
    # branch_graph = nx.Graph((r.values for i,r in branch_edges.iterrows()))
    # branch_edges = pd.DataFrame([(k,v) for k,v in nx.dfs_successors(branch_graph, source=0).items()], columns=["parent_branch_index", "child_branch_index"]).explode("child_branch_index")

    edge_params = nx.get_edge_attributes(graph, "parameters")
    edge_params = {k: v for k, v in edge_params.items() if k in synapse_edges}
    synapse_edges = pd.DataFrame(sum(edge_params.values(), [])).T

    acc_parents = []
    parent_branch_inds = branch_edges.set_index("child_branch_index").sort_index()[
        "parent_branch_index"
    ]
    for branch_inds in nodes.groupby("cell_index")["branch_index"].unique():
        root_branch_idx = branch_inds[0]
        parents = parent_branch_inds.loc[branch_inds[1:]] - root_branch_idx
        acc_parents.append([-1] + parents.tolist())

    # drop special attrs from nodes and ignore error if col does not exist
    optional_attrs = ["recordings", "currents", "groups", "trainable"]
    nodes.drop(columns=optional_attrs, inplace=True, errors="ignore")

    # build module
    idxs = nodes[["cell_index", "branch_index", "comp_index"]]
    module = build_module_scaffold(idxs, graph.graph["type"], acc_parents)

    # set global attributes of module
    for k, v in graph.graph.items():
        setattr(module, k, v)

    module.nodes[nodes.columns] = nodes  # set column-wise. preserves cols not in nodes.
    module.edges = synapse_edges

    # Add optional attributes if they can be found in nodes
    recordings = pd.DataFrame(nx.get_node_attributes(graph, "recordings"))
    currents = pd.DataFrame(nx.get_node_attributes(graph, "currents"))
    groups = pd.DataFrame(nx.get_node_attributes(graph, "groups").items())
    trainables = pd.DataFrame(nx.get_node_attributes(graph, "trainable"), dtype=float)

    if not recordings.empty:
        recordings = recordings.T.unstack().reset_index().set_index("level_0")
        recordings = recordings.rename(columns={"level_1": "rec_index", 0: "state"})
        module.recordings = recordings

    if not currents.empty:
        current_inds = nodes.loc[currents.T.index]
        currents = jnp.vstack(currents.values).T
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
        indices_set_by_trainables = trainable_params["index"].values.tolist()
        trainable_params = [
            {k: jnp.array(v).reshape(-1)}
            for k, v in trainable_params[["param", "value"]].values
        ]
        module.trainable_params = trainable_params
        module.indices_set_by_trainables = indices_set_by_trainables

    if not groups.empty and assign_groups:
        groups = groups.explode(1).rename(columns={0: "index", 1: "group"})
        groups = groups[groups["group"] != "undefined"]  # skip undefined comps
        group_nodes = {k: nodes.loc[v["index"]] for k, v in groups.groupby("group")}
        module.group_nodes = group_nodes
        # update group nodes in module to reflect what is shown in view
        for group, nodes in module.group_nodes.items():
            module.group_nodes[group] = module.__getattr__(group).view

    return module
