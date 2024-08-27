import os
from copy import deepcopy

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd

import jaxley as jx
from jaxley import connect
from jaxley.channels.pospischil import K, Leak, Na
from jaxley.io.graph import from_graph, make_jaxley_compatible, to_graph
from jaxley.io.swc import swc_to_graph
from jaxley.synapses import IonotropicSynapse, TestSynapse
from jaxley.utils.misc_utils import recursive_compare


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


def compare_modules(m1, m2):
    d1 = deepcopy(m1.__dict__)
    d2 = deepcopy(m2.__dict__)

    # compare edges seperately since, they might be permuted differently
    m1_edges = d1.pop("edges").replace(np.nan, 0)
    m2_edges = d2.pop("edges").replace(np.nan, 0)
    equal_edges = (
        True
        if m1_edges.empty and m2_edges.empty
        else (m1_edges == m2_edges).all().all()
    )

    # compare trainables seperately since, they might be permuted differently
    m1_trainables = d1.pop("trainable_params")
    m2_trainables = d2.pop("trainable_params")
    m1_trainable_inds = d1.pop("indices_set_by_trainables")
    m2_trainable_inds = d2.pop("indices_set_by_trainables")
    m1_trainables = get_unique_trainables(m1_trainable_inds, m1_trainables)
    m2_trainables = get_unique_trainables(m2_trainable_inds, m2_trainables)
    equal_trainables = np.all(m1_trainables == m2_trainables)

    assert equal_trainables
    assert equal_edges
    assert recursive_compare(d1, d2)


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
    net.cell(0).branch(0).loc(0.0).record("Na_m")

    # add stimuli
    current = jx.step_current(0.0, 0.0, 0.0, 0.025, 1.0)
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
        compare_modules(module, re_module)

        # ensure graphs are equal
        assert nx.is_isomorphic(
            module_graph,
            re_module_graph,
            node_match=recursive_compare,
            edge_match=recursive_compare,
        )

        # test if imported module can be simulated
        if isinstance(module, jx.Network):
            jx.integrate(re_module)


# def test_graph_swc_tracer():
#     nseg = 8
#     fname = os.path.join("../tests/swc_files/", "morph.swc")  # n120

#     graph = swc_to_graph(fname)
#     graph = make_jaxley_compatible(graph, nseg=nseg, max_branch_len=2000, ignore_swc_trace_errors=False)
#     cell = from_graph(graph, nseg=nseg, max_branch_len=2000)
#     neuron_cell = import_neuron_morph(fname, nseg=nseg)

#     # remove root branch
#     jaxley_comps = cell.nodes[cell.nodes["branch_index"] != 0].reset_index(drop=True)
#     jaxley_comps["branch_index"] -= 1

#     jx_branch_lens = jaxley_comps.groupby("branch_index")["length"].sum().to_numpy()

#     # match by branch lengths
#     neuron_xyzd = [np.array(s.psection()["morphology"]["pts3d"]) for s in h.allsec()]
#     neuron_branch_lens = np.array([np.sqrt((np.diff(n[:,:3], axis=0)**2).sum(axis=1)).sum() for n in neuron_xyzd])
#     neuron_inds = np.argsort(neuron_branch_lens)
#     jx_inds = np.argsort(jx_branch_lens)

#     errors = pd.DataFrame(columns=["idx_NEURON", "idx_Jaxley", "dxyz", "dl", "dr"])
#     for k in range(len(neuron_inds)):
#         neuron_comp_k = np.array([get_segment_xyzrL(list(h.allsec())[neuron_inds[k]], comp_idx=i) for i in range(nseg)])
#         jx_comp_k = jaxley_comps[jaxley_comps["branch_index"] == jx_inds[k]][["x", "y", "z", "radius", "length"]].to_numpy()
#         dxyz = (((neuron_comp_k[:,:3] - jx_comp_k[:,:3])**2).sum(axis=1)**0.5).max()
#         dl = abs(neuron_comp_k[:,4] - jx_comp_k[:,4]).max()
#         dr = abs(neuron_comp_k[:,3] - jx_comp_k[:,3]).max()
#         errors.loc[k] = [neuron_inds[k], jx_inds[k], dxyz, dl, dr]

#     # allow one error, see https://github.com/jaxleyverse/jaxley/issues/140
#     assert len(errors['dxyz'][errors['dxyz'] > 0.001]) <= 1, "traced coords do not match."
#     assert len(errors['dl'][errors['dl'] > 0.001]) <= 1, "traced lengths do not match."
#     assert len(errors['dr'][errors['dr'] > 0.001]) <= 1, "traced radii do not match."


def test_graph_to_jaxley():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(2)])
    cell = jx.Cell([branch for _ in range(5)], parents=jnp.asarray([-1, 0, 1, 2, 2]))
    net = jx.Network([cell] * 3)

    # test if edge graph can pe imported into to jaxley
    sets_of_edges = [
        [(0, 1), (1, 2), (2, 3)],
        [(0, 1), (1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)],
    ]
    for edges in sets_of_edges:
        edge_graph = nx.DiGraph(edges)
        edge_module = from_graph(edge_graph)

    # test whether swc file can be imported into jaxley
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files/morph.swc")
    graph = swc_to_graph(fname)
    swc_module = from_graph(graph)
    for group in ["soma", "apical", "basal"]:
        assert group in swc_module.group_nodes

    # test import after different stages of graph pre-processing
    graph = swc_to_graph(fname)
    module_imported_directly = from_graph(deepcopy(graph))

    graph = make_jaxley_compatible(deepcopy(graph))
    module_imported_after_preprocessing = from_graph(graph)

    compare_modules(module_imported_directly, module_imported_after_preprocessing)
