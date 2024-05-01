from math import pi
from typing import Dict, List, Optional, Union

import jax.numpy as jnp
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

        parents = np.arange(len(build_cache["branch"])) - 1 # ignores morphology
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
        module_graph.add_node(idx, **{"groups": key}) # TODO: allow multiple group memberships?

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

def dist(pre, post):
    pre_loc = np.hstack((pre["x"], pre["y"], pre["z"]))
    post_loc = np.hstack((post["x"], post["y"], post["z"]))
    return np.sqrt(np.sum((pre_loc - post_loc)**2))

def get_linear_paths(graph):
    path_edges = []
    current_path = []
    for i,j in nx.dfs_edges(graph, 0):
        current_path.append((i,j))
        is_leave_node = graph.out_degree(j) == 0 and graph.in_degree(j) == 1
        is_branching_node = graph.out_degree(j) > 1
        if is_leave_node or is_branching_node:
            path_edges.append(current_path)
            current_path = []
    return path_edges

path_e2n = lambda path: [path[0][0]]+[e[1] for e in path]
path_n2e = lambda path: [e for e in zip(path[:-1], path[1:])]

def impose_branch_structure(graph, nsegs = 4, max_branch_len = 100):
    has_loc = "x" in graph.nodes[0]
    has_rad = "r" in graph.nodes[0]
    has_id = "id" in graph.nodes[0]

    pathgraphs = []
    max_branch_idx = 0

    for i,j in graph.edges:
        graph.edges[i,j]["length"] = dist(graph.nodes[i], graph.nodes[j]) if has_loc else 1

    # segment linear sections of the graph/morphology
    linear_paths  = get_linear_paths(graph)
    for path in linear_paths:
        # get node and edge attrs
        path_nodes = path_e2n(path)
        path_node_attrs = [graph.nodes[i] for i in path_nodes]
        lengths = [graph.edges[(i,j)]["length"] for i,j in path]

        xyz = [[n[k] if has_loc else np.nan for k in "xyz"] for n in path_node_attrs]
        r = [[n["r"] if has_loc else 1.0] for n in path_node_attrs]
        xyzr = np.hstack([np.array(xyz), np.array(r)])
        ids = np.array([n["id"] if has_id else 0 for n in path_node_attrs])

        # segment morphology into branches based on nseg and max_branch_len
        pathlens = np.cumsum(np.array([0]+lengths))
        total_pathlen = pathlens[-1]
        num_branches = int(total_pathlen / max_branch_len) + 1
        
        # index branches
        branch_inds = (max_branch_idx + np.arange(num_branches).repeat(nsegs))
        max_branch_idx = branch_inds[-1] + 1
        
        # compute where to place compartment centers along morphology
        comp_len = total_pathlen / (num_branches*nsegs)
        locs = np.linspace(comp_len / 2, total_pathlen-comp_len / 2, num_branches*nsegs)
        new_nodes = np.interp(locs, pathlens, path_nodes) 
        new_nodes += 0.1*np.random.randn() # ensure unique nodes -> disjoint subgraphs 

        # interpolate xyzr and ids along the morphology
        new_xyzr = vmap(jnp.interp, in_axes=(None, None, 1))(locs, pathlens, xyzr)
        new_ids = np.interp(locs, pathlens, ids).astype(int)

        # construct graph for segemented morphology 
        keys = ["x", "y", "z", "r", "id", "branch_index"]
        new_node_attrs = zip(*new_xyzr.tolist(), new_ids, branch_inds)
        new_node_attrs = [{k:v for k,v in zip(keys, items)} for items in new_node_attrs]

        pathgraph = nx.DiGraph(path_n2e(new_nodes))
        pathgraph.add_nodes_from(zip(new_nodes, new_node_attrs))
        pathgraphs.append(pathgraph)

    # combine linear paths
    new_graph = nx.union_all(pathgraphs)

    # connect linear subgraphs according to the original graph
    linear_path_roots_leaves = np.array([(path[0][0], path[-1][-1]) for path in linear_paths])
    edges_between_linear_paths = list(zip(*np.where(np.equal(*np.meshgrid(*(linear_path_roots_leaves.T))))))
    new_edges_between_linear_paths = [(max(pathgraphs[i]), min(pathgraphs[j])) for i,j in edges_between_linear_paths]
    new_graph.add_edges_from(new_edges_between_linear_paths)

    # reconnect root
    root_edges = list(zip(*np.where([linear_path_roots_leaves[:,0] == 0])))[1:] # drop (0,0)
    new_root_edges = [(min(pathgraphs[i]), min(pathgraphs[j])) for i,j in root_edges]
    new_graph.add_edges_from(new_root_edges)

    # rename nodes by enumeration (since nodes idxs were interpolated)
    new_keys = {k: i for i, k in enumerate(new_graph.nodes)}
    new_graph = nx.relabel_nodes(new_graph, new_keys)
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

    The only requirement to construct a module from a graph is that the graph contains
    edges. In this case, all requrired parameters are set to their default values
    and the branch structure is computed naively.

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
    
    # Graph only has to contain edges to be importable as a module
    # In this case, xyz are set to NaN and r to 1
    assert nx.is_weakly_connected(graph), "Graph must be connected to be imported as a module."
    assert len(list(nx.simple_cycles(new_graph))) == 0, "Graph cannot have loops"
    if not graph.edges:
        raise ValueError("Graph must have edges to be imported as a module.")

    ########################################
    ### Make the graph jaxley compatible ###
    ########################################

    # add comp_index, branch_index, cell_index to graph
    # first check if they are already present, otherwise compute them
    if not "branch_index" in graph.nodes[0]:
        nseg = graph.graph["nseg"] if "nseg" in graph.graph else default_nseg
        
        # add branch_index and segment morphology into branches and compartments
        graph = impose_branch_structure(graph, nsegs = nseg, max_branch_len = max_branch_len)
        nx.set_node_attributes(graph, {i:0 for i in graph.nodes}, "cell_index")
        nx.set_node_attributes(graph, {i:i for i in graph.nodes}, "comp_index")

    # setup branch structure and compute parents
    # edges connecting comps in different branches are set to type "branch"
    # also works for mutliple cells
    if nx.get_edge_attributes(graph, "type") == {}:
        cell_idxs = nx.get_node_attributes(graph, "cell_index")
        unique_cells = np.unique(list(cell_idxs.values()))
        graph.graph["comb_parents"] = []
        for idx in unique_cells:
            cell_nodes = [k for k, v in cell_idxs.items() if v == idx]
            cell_graph = graph.subgraph(cell_nodes)
            branch_idxs = nx.get_node_attributes(cell_graph, "branch_index")
            within_branch = np.array([(branch_idxs[i] == branch_idxs[j]) for i,j in graph.edges])
            node_connections = np.stack([(i, j) for i,j in graph.edges])
            pre, post = node_connections[~within_branch].T
            graph.add_edges_from(zip(pre, post), type="branch")
            branch_connections = [(branch_idxs[i], branch_idxs[j]) for i,j in zip(pre, post)]
            parents = nx.dfs_predecessors(nx.DiGraph(branch_connections),0)
            parents = [-1] + list(parents.values())
            graph.graph["comb_parents"] += parents

    # add group information to nodes if available
    ids = nx.get_node_attributes(graph, "id")
    if ids == {}:
        group_ids = {0: "undefined", 1: "soma", 2: "axon", 3: "basal", 4: "apical"}
        # Type of padded section is assumed to be of `custom` type:
        # http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
        groups = [group_ids[id] if id in group_ids else "custom" for id in ids]
        graph.add_nodes_from({i:{"groups":[id]} for i, id in enumerate(groups)}.items())
        nx.set_node_attributes(graph, {i:None for i in graph.nodes}, "id") # remove id

    ###############################
    ### Port graph to jx.Module ###
    ###############################

    global_attrs = graph.graph.copy()
    # try to infer module type from global attrs (exported with `to_graph`)
    return_type = (global_attrs.pop("module").lower() if "module" in global_attrs else None)
    
    # build skeleton of module instances
    nodes = pd.DataFrame((n for i, n in graph.nodes(data=True)))
    module = build_skeleton_module(nodes, return_type)

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

    module.nodes[nodes.columns] = nodes # set column-wise. preserves cols not in nodes.
    module.branch_edges = branch_edges
    module.edges = synapse_edges

    # Add optional attributes if they can be found in nodes
    recordings = pd.DataFrame(nx.get_node_attributes(graph, "recordings"))
    currents = pd.DataFrame(nx.get_node_attributes(graph, "currents"))
    groups = pd.DataFrame(nx.get_node_attributes(graph, "groups").items())
    trainables = pd.DataFrame(
        nx.get_node_attributes(graph, "trainable"), dtype=float
    )

    if not groups.empty and assign_groups:
        groups = groups.explode(1).rename(columns={0: "index", 1: "group"})
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
