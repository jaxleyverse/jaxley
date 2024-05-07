import os
from copy import deepcopy

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd

import jaxley as jx
from jaxley import connect
from jaxley.channels.pospischil import K, Leak, Na
from jaxley.io.graph import (
    compartmentalize_branches,
    from_graph,
    impose_branch_structure,
    to_graph,
)
from jaxley.io.swc import swc_to_graph
from jaxley.synapses import IonotropicSynapse, TestSynapse


def match_container_elements(a, b):
    if type(a) != type(b):
        return False
    if l := len(a) > 0:
        if isinstance(a[0], dict):
            for i in range(l):
                if not match_dicts_and_contents(a[i], b[i]):
                    return False
            return True

        # replace nans by 0 since nan != nan
        return np.all(np.nan_to_num(a) == np.nan_to_num(b))
    return True


def match_dicts_and_contents(a, b):
    if a.keys() != b.keys():  # keys must match
        return False

    for key in a.keys():
        A, B = a[key], b[key]  # unpack by key to ensure same order
        if type(A) != type(B):
            return False
        if isinstance(A, pd.DataFrame):
            A, B = A.to_dict(), B.to_dict()

        if isinstance(A, dict):
            if not match_dicts_and_contents(A, B):
                return False
        elif isinstance(A, (list, np.ndarray, jnp.ndarray)):
            if not match_container_elements(A, B):
                return False
        else:
            both_nan = np.isnan(A) and np.isnan(B) if isinstance(A, float) else False
            if A != B and not both_nan:
                return False
    return True


def get_unique_trainables(indices_set_by_trainables, trainable_params):
    trainables = []
    for inds, params in zip(indices_set_by_trainables, trainable_params):
        inds = inds.flatten().tolist()
        pkey, pvals = next(iter(params.items()))
        # repeat pval to match inds
        pvals = np.repeat(pvals, len(inds) if len(pvals) == 1 else 1, axis=0).tolist()
        pkey = [pkey] * len(pvals)
        trainables += list(zip(inds, pkey, pvals))
    return (
        np.unique(np.stack(trainables), axis=0) if len(trainables) > 0 else np.array([])
    )


# test exporting and re-importing of different modules
def test_graph_import_export_cycle():
    # build a network
    np.random.seed(0)
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(4)])
    cell = jx.Cell([branch for _ in range(5)], parents=jnp.asarray([-1, 0, 1, 2, 2]))
    net = jx.Network([cell] * 3)

    # add synapses
    connect(net[0, 0, 0], net[1, 0, 0], IonotropicSynapse())
    connect(net[0, 0, 1], net[1, 0, 1], IonotropicSynapse())
    connect(net[0, 0, 1], net[1, 0, 1], TestSynapse())

    # add groups
    net.cell(2).add_to_group("cell2")
    net.cell(2).branch(1).add_to_group("cell2branch1")

    # add ion channels
    net.cell(0).insert(Na())
    net.cell(0).insert(Leak())
    net.cell(1).branch(1).insert(Na())
    net.cell(0).insert(K())

    # add recordings
    net.cell(0).branch(0).loc(0.0).record()
    net.cell(0).branch(0).loc(0.0).record("m")

    # add stimuli
    current = jx.step_current(10.0, 80.0, 5.0, 0.025, 100.0)
    net.cell(0).branch(2).loc(0.0).stimulate(current)
    net.cell(1).branch(2).loc(0.0).stimulate(current)

    # add trainables
    net.cell(0).branch(1).make_trainable("Na_gNa")
    net.cell(0).make_trainable("K_gK")
    net.cell(1).branch("all").comp("all").make_trainable("Na_gNa", [0.0, 0.1, 0.2, 0.3])

    # test consistency of exported and re-imported modules
    for module in [net, cell, branch, comp]:
        module.compute_xyz()  # enforces x,y,z in nodes before exporting for later comparison
        module_graph = to_graph(module)  # ensure to_graph works
        re_module = from_graph(module_graph)  # ensure prev exported graph can be read
        re_module_graph = to_graph(
            re_module
        )  # ensure to_graph works on re-imported module

        # ensure modules are equal
        module_dict = deepcopy(module.__dict__)
        re_module_dict = deepcopy(re_module.__dict__)

        # compare edges seperately since, they might be permuted differently
        edges = module_dict.pop("edges").replace(np.nan, 0)
        re_edges = re_module_dict.pop("edges").replace(np.nan, 0)
        assert (
            True if edges.empty and re_edges.empty else (edges == re_edges).all().all()
        ), f"edges do not match for {module}"

        # compare trainables seperately since, they might be permuted differently
        trainable_params = module_dict.pop("trainable_params")
        re_trainable_params = re_module_dict.pop("trainable_params")
        indices_set_by_trainables = module_dict.pop("indices_set_by_trainables")
        re_indices_set_by_trainables = re_module_dict.pop("indices_set_by_trainables")
        trainables = get_unique_trainables(indices_set_by_trainables, trainable_params)
        re_trainables = get_unique_trainables(
            re_indices_set_by_trainables, re_trainable_params
        )
        assert np.all(trainables == re_trainables)

        # match the remaining attributes in the module dicts
        match_dicts_and_contents(module_dict, re_module_dict)

        # ensure graphs are equal
        assert nx.is_isomorphic(
            module_graph,
            re_module_graph,
            node_match=match_dicts_and_contents,
            edge_match=match_dicts_and_contents,
        )


def test_graph_to_jaxley():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(4)])
    cell = jx.Cell([branch for _ in range(5)], parents=jnp.asarray([-1, 0, 1, 2, 2]))
    net = jx.Network([cell] * 3)

    # test if edge graph can pe imported into to jaxley
    sets_of_edges = [[(0, 1), (1, 2), (1, 3)], [(0, 1)]]
    for edges in sets_of_edges:
        edge_graph = nx.DiGraph(edges)
        edge_module = from_graph(edge_graph)

    # test whether swc file can be imported into jaxley
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "morph.swc")
    graph = swc_to_graph(fname)
    swc_module = from_graph(graph)
    for group in ["soma", "apical", "basal"]:
        assert group in swc_module.group_nodes

    # test if exported module can be imported into jaxley
    module_graph = to_graph(net)
    prev_exported_module = from_graph(module_graph)

    # test import at different stages of graph pre-processing
    graph = swc_to_graph(fname)
    graph = impose_branch_structure(graph)
    from_graph(graph)

    graph = compartmentalize_branches(graph)
    from_graph(graph)
