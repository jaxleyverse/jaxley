from math import pi
from typing import Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
from jax import vmap
import networkx as nx
import numpy as np
import pandas as pd

import jaxley as jx

def infer_module_type_from_inds(idxs: pd.DataFrame) -> str:
    nuniques = idxs.nunique()[["cell_index", "branch_index", "comp_index"]]
    nuniques.index = ["cell", "branch", "compartment"]
    nuniques = pd.concat([pd.Series({"network": 1}), nuniques])
    return_type = nuniques.loc[nuniques == 1].index[-1]
    return return_type

def build_module_scaffold(
    idxs: pd.DataFrame, 
    return_type: Optional[str] = None, 
    parent_branches: Optional[List[np.ndarray]] = None
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
            parents = default_parents if parent_branches is None else parent_branches[cell_id]
            cell = jx.Cell([branch]*num_branches, parents)
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
        module_graph.add_edges_from(zip(group.index[:-1], group.index[1:]), type="intra_branch")

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
path_e2n = lambda path: [path[0][0]] + [e[1] for e in path]
path_n2e = lambda path: [e for e in zip(path[:-1], path[1:])]
unpack = lambda d, keys: [d[k] for k in keys]
has_node_attr = lambda graph, attr: all(attr in graph.nodes[n] for n in graph.nodes)
has_edge_attr = lambda graph, attr: all(attr in graph.edges[e] for e in graph.edges)
v_interp = vmap(jnp.interp, in_axes=(None, None, 1))

def get_linear_paths(graph: nx.DiGraph) -> List[np.ndarray]:
    """Get all linear paths in a graph.

    The graph is traversed depth-first starting from the root node.

    Args:
        graph: A networkx graph.

    Returns:
        A list of linear paths in the graph. Each path is represented as an array of
        edges."""
    paths, current_path = [], []
    leaf_idx = next(x for x in graph.nodes() if graph.out_degree(x)==0 and graph.in_degree(x)==1)
    # initiate traversal from leaf node, this avoids having to reconnect the
    # source node to the rest of the graph later on
    for i, j in nx.dfs_edges(graph.to_undirected(), leaf_idx):
        current_path.append((i, j))
        if is_leaf(graph, j) or is_branching(graph, j):
            paths.append(np.array(current_path))
            current_path.clear()
    return paths

def insert_rows(df: pd.DataFrame, at: int, locs: List[float] = [0.5, 0.5]) -> pd.DataFrame:
    rows = df.iloc[at-1:at+1]
    lens = rows["l"].values
    locs = np.array(locs)
    assert np.logical_and(0 <= locs, locs <= 1).all(), "at must be in [0, 1]"
    points = np.interp(locs, [0, 1], lens)
    interp_data = np.array(v_interp(points, lens, rows.values))
    interp_data = pd.DataFrame(interp_data.T, columns=rows.columns)
    df = pd.concat([df.iloc[:at], interp_data, df.iloc[at:]], ignore_index=True)
    return df

def make_jaxley_compatible(graph: nx.DiGraph, nseg: int = 8, max_branch_len: float = 2000) -> nx.DiGraph:
    linear_paths = get_linear_paths(graph)
    graph_data = pd.DataFrame([d for i,d in graph.nodes(data=True)])
    linear_path_nodes = [path_e2n(p) for p in linear_paths]

    # segment into branches
    jaxley_branches = []
    branch_counter = 0
    for i, path in enumerate(linear_path_nodes):
        path_nodes = graph_data.loc[path]
        path_nodes["node_index"] = path
        seglengths = path_nodes[["x", "y", "z"]].diff().pow(2).sum(1).pow(0.5)
        pathlengths = seglengths.cumsum().reset_index(drop=True)
        path_nodes["l"] = pathlengths.values

        pathlens_by_id = path_nodes.groupby("id")["l"]
        total_pathlen = pathlens_by_id.max() - pathlens_by_id.min()
        num_branches = (total_pathlen / max_branch_len + 1).astype(int)
        
        branch_ends = (total_pathlen / num_branches)
        branch_ends = (branch_ends.apply(lambda x: [x])*num_branches).apply(np.cumsum) + pathlens_by_id.min()
        branch_ends = np.hstack(branch_ends.values)

        enum_branches = lambda x: (x <= branch_ends).sum()
        path_branch_inds = path_nodes["l"].apply(enum_branches) -1
        path_nodes.loc[:, "branch_index"] = branch_counter + path_branch_inds
        branch_counter += len(branch_ends)
        
        # close gaps in path_nodes between different branches
        interp_with = [0.5, 0.5]
        if path_nodes.groupby("branch_index").ngroups > 1:
            change_of_branch = path_nodes["branch_index"].diff() != 0
            for idx in np.where(change_of_branch)[0][1:]:
                path_nodes = insert_rows(path_nodes, idx, locs=interp_with)
                keys = ["branch_index", "id"]
                new_inds = np.arange(idx, idx+len(interp_with))
                prev_inds = new_inds[[0, -1]] + [-1, 1]
                path_nodes.loc[new_inds, keys] = path_nodes.loc[prev_inds, keys].values
    
        path_nodes["path_index"] = i
        jaxley_branches.append(path_nodes)
    jaxley_branches = pd.concat(jaxley_branches).reset_index(drop=True)

    max_branch_idx = jaxley_branches["branch_index"].unique().max()
    num_branches = jaxley_branches["branch_index"].unique().shape
    assert max_branch_idx + 1 == num_branches, "max_branch_len is presumably too small!"

    # compartmentalize branches
    jaxley_comps = []
    keys = ["x", "y", "z", "r", "id", "branch_index", "path_index"]
    for idx, branch_nodes in jaxley_branches.groupby("branch_index"):
        branch_len = branch_nodes["l"].max()
        comp_len = branch_len / nseg
        locs = np.linspace(comp_len/2, branch_len-comp_len/2, nseg)
        comp_attrs = v_interp(locs, branch_nodes["l"].values, branch_nodes[keys].values)
        comp_attrs = pd.DataFrame(comp_attrs.T, columns=keys)
        comp_attrs[keys[-3:]] = comp_attrs[keys[-3:]].astype(int)
        comp_attrs["length"] = comp_len
        jaxley_comps.append(comp_attrs)
    jaxley_comps = pd.concat(jaxley_comps, ignore_index=True)

    jaxley_comps.rename(columns={"r": "radius"}, inplace=True)
    jaxley_comps["cell_index"] = 0
    jaxley_comps["comp_index"] = np.arange(jaxley_comps.shape[0])

    # Types assigned according to the SWC format:
    # http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
    group_ids = {0: "undefined", 1: "soma", 2: "axon", 3: "basal", 4: "apical"}
    id2group = lambda x: [group_ids[x] if x < 5 else f"custom{x}"]
    jaxley_comps["groups"] = jaxley_comps["id"].apply(id2group)

    # set up edges
    path_roots_and_leafs = np.stack([np.array(edge)[[0, -1]] for edge in linear_path_nodes])
    parent_equal_child = np.equal(*np.meshgrid(*(path_roots_and_leafs.T)))
    edges_between_paths = np.stack(list(zip(*np.where(parent_equal_child))))

    comps_in_paths = jaxley_comps.groupby("path_index")["comp_index"]
    intra_path_edges = [path_n2e(g) for i, g in comps_in_paths]
    intra_path_edges = np.array(sum(intra_path_edges, []))
    num_path_inds = comps_in_paths.count().values -1
    intra_branch_edge_inds = np.hstack([np.arange(1,n)*nseg+idx for idx, n in zip(np.cumsum([0]+num_path_inds.tolist()), ((np.array(num_path_inds)+1) // nseg)) if n > 1]) -1
    inter_branch_edges = intra_path_edges[intra_branch_edge_inds]
    intra_branch_edges = np.delete(intra_path_edges, intra_branch_edge_inds, axis=0)

    inter_path_leafs = comps_in_paths.first().values[edges_between_paths[:,1]]
    inter_path_roots = comps_in_paths.last().values[edges_between_paths[:,0]]
    inter_path_edges = np.stack([inter_path_roots, inter_path_leafs]).T

    # construct compartmentalized graph
    comp_graph = nx.DiGraph()
    comp_graph.add_edges_from(intra_branch_edges, type="intra_branch")
    comp_graph.add_edges_from(inter_branch_edges, type="inter_branch")
    comp_graph.add_edges_from(inter_path_edges, type="inter_branch")

    keys = ["radius", "length", "x", "y", "z", "comp_index", "branch_index", "cell_index", "groups"]
    comp_graph.add_nodes_from(((n, {k:v for k,v in zip(keys, vals)}) for i, (n,*vals) in jaxley_comps[["comp_index"]+keys].iterrows()))   
    
    comp_graph.graph["nseg"] = nseg
    #TODO: FIX XYZ breakage! (seems to happen when num_comps_in_branch > 1)
    comp_graph.graph["xyzr"] = [n[["x","y","z","r"]].values for i,n in jaxley_branches.groupby("branch_index")]
    comp_graph.graph["type"] = infer_module_type_from_inds(jaxley_comps[["cell_index", "branch_index", "comp_index"]])

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
    nodes = nodes.sort_values("comp_index", ignore_index=True) # ensure index == comp_index
    edge_type = nx.get_edge_attributes(graph, "type")
    edges = pd.DataFrame(edge_type.values(), index=edge_type.keys(), columns=["type"])
    edges = edges.reset_index(names=["pre", "post"])
    is_synapse = edges["type"] == "synapse"
    is_inter_branch = edges["type"] == "inter_branch"
    inter_branch_edges = edges.loc[is_inter_branch][["pre", "post"]].values
    synapse_edges = edges.loc[is_synapse][["pre", "post"]].values
    branch_edges = pd.DataFrame(nodes["branch_index"].values[inter_branch_edges], columns=["parent_branch_index", "child_branch_index"])

    edge_params = nx.get_edge_attributes(graph, "parameters")
    edge_params = {k: v for k, v in edge_params.items() if k in synapse_edges}
    synapse_edges = pd.DataFrame(sum(edge_params.values(), [])).T

    acc_parents = []
    parent_branch_inds = branch_edges.set_index("child_branch_index").sort_index()["parent_branch_index"]
    for branch_inds in nodes.groupby("cell_index")["branch_index"].unique():
        soma_branch_idx = branch_inds[0]
        parents = parent_branch_inds.loc[branch_inds[1:]] - soma_branch_idx
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
