import os
from copy import deepcopy

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd
import pytest

import jaxley as jx
from jaxley import connect
from jaxley.channels import HH
from jaxley.channels.pospischil import K, Leak, Na
from jaxley.io.graph import (
    from_graph,
    get_soma_idxs,
    make_jaxley_compatible,
    simulate_swc_trace_errors,
    swc_to_graph,
    to_graph,
    trace_branches,
)
from jaxley.synapses import IonotropicSynapse, TestSynapse
from jaxley.utils.misc_utils import recursive_compare
from tests.helpers import (
    get_segment_xyzrL,
    import_neuron_morph,
    jaxley2neuron_by_group,
    match_stim_loc,
)


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

    m1_synapses = d1.pop("synapses")
    m2_synapses = d2.pop("synapses")
    for syn1, syn2 in zip(m1_synapses, m2_synapses):
        assert recursive_compare(syn1.__dict__, syn2.__dict__)

    m1_channels = d1.pop("channels")
    m2_channels = d2.pop("channels")
    for ch1, ch2 in zip(m1_channels, m2_channels):
        assert recursive_compare(ch1.__dict__, ch2.__dict__)

    # assumes only group inds matter for viewing, otherwise no comparison is possible
    # since i.e.
    # 1) cell.branch(0).add_to_group("soma"); cell.insert(Na())
    # 2) cell.insert(Na()); cell.branch(0).add_to_group("soma")
    # will result in different group_nodes, while cell.nodes are the same
    m1_groups = d1.pop("group_nodes")
    m2_groups = d2.pop("group_nodes")
    assert m1_groups.keys() == m2_groups.keys()

    for g in m1_groups:
        assert np.all(m1_groups[g].index == m1_groups[g].index)

    assert equal_edges
    assert equal_trainables
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


@pytest.mark.parametrize("file", ["morph_single_point_soma.swc", "morph.swc"])
def test_trace_branches(file):
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)
    graph = swc_to_graph(fname)

    source_node = 0
    if len(soma_idxs := get_soma_idxs(graph)) == 1:
        # Setting l = 2*r ensures A_cylinder = 2*pi*r*l = 4*pi*r^2 = A_sphere
        graph.add_edge(soma_idxs[0], soma_idxs[0], l=2 * graph.nodes[soma_idxs[0]]["r"])
        soma_edges = [
            (i, j) for i, j in graph.edges if soma_idxs[0] in [i, j] and i != j
        ]
        # edges connecting nodes to soma are considered part of the soma -> l = 0.
        for i, j in soma_edges:
            graph.edges[i, j]["l"] = 0

    branches = trace_branches(graph, source_node=source_node)
    branches = simulate_swc_trace_errors(graph, branches)

    g = graph.to_undirected()
    nx_branch_lens = np.sort(
        [sum([g.edges[i, j]["l"] for i, j in branch]) for branch in branches]
    )

    h, _ = import_neuron_morph(fname)
    neuron_branch_lens = np.sort([sec.L for sec in h.allsec()])

    errors = np.abs(neuron_branch_lens - nx_branch_lens)
    # one error is expected, see https://github.com/jaxleyverse/jaxley/issues/140
    assert sum(errors > 1e-3) <= 1


@pytest.mark.parametrize("file", ["morph_single_point_soma.swc", "morph.swc"])
def test_from_graph_vs_NEURON(file):
    nseg = 8
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)

    graph = swc_to_graph(fname)
    cell = from_graph(
        graph, nseg=nseg, max_branch_len=2000, ignore_swc_trace_errors=False
    )
    cell._update_nodes_with_xyz()
    h, neuron_cell = import_neuron_morph(fname, nseg=nseg)

    # remove root branch
    jaxley_comps = cell.nodes[
        ~np.isclose(cell.nodes["length"], 0.1 / nseg)
    ].reset_index(drop=True)

    jx_branch_lens = jaxley_comps.groupby("branch_index")["length"].sum().to_numpy()

    # match by branch lengths
    neuron_xyzd = [np.array(s.psection()["morphology"]["pts3d"]) for s in h.allsec()]
    neuron_branch_lens = np.array(
        [
            np.sqrt((np.diff(n[:, :3], axis=0) ** 2).sum(axis=1)).sum()
            for n in neuron_xyzd
        ]
    )
    neuron_inds = np.argsort(neuron_branch_lens)
    jx_inds = np.argsort(jx_branch_lens)

    neuron_df = pd.DataFrame(columns=["neuron_idx", "x", "y", "z", "radius", "length"])
    jx_df = pd.DataFrame(columns=["jx_idx", "x", "y", "z", "radius", "length"])
    for k in range(len(neuron_inds)):
        neuron_comp_k = np.array(
            [
                get_segment_xyzrL(list(h.allsec())[neuron_inds[k]], comp_idx=i)
                for i in range(nseg)
            ]
        )
        # make this a dataframe
        neuron_comp_k = pd.DataFrame(
            neuron_comp_k, columns=["x", "y", "z", "radius", "length"]
        )
        neuron_comp_k["idx"] = neuron_inds[k]
        jx_comp_k = jaxley_comps[jaxley_comps["branch_index"] == jx_inds[k]][
            ["x", "y", "z", "radius", "length"]
        ]
        jx_comp_k["idx"] = jx_inds[k]
        neuron_df = pd.concat([neuron_df, neuron_comp_k], axis=0, ignore_index=True)
        jx_df = pd.concat([jx_df, jx_comp_k], axis=0, ignore_index=True)

    errors = neuron_df["neuron_idx"].to_frame()
    errors["jx_idx"] = jx_df["jx_idx"]
    errors[["x", "y", "z"]] = neuron_df[["x", "y", "z"]] - jx_df[["x", "y", "z"]]
    errors["xyz"] = np.sqrt((errors[["x", "y", "z"]] ** 2).sum(axis=1))
    errors["radius"] = neuron_df["radius"] - jx_df["radius"]
    errors["length"] = neuron_df["length"] - jx_df["length"]

    # one error is expected, see https://github.com/jaxleyverse/jaxley/issues/140
    assert sum(errors.groupby("jx_idx")["xyz"].max() > 1e-3) <= 1
    assert sum(errors.groupby("jx_idx")["radius"].max() > 1e-3) <= 1
    assert sum(errors.groupby("jx_idx")["length"].max() > 1e-3) <= 1


def test_edges_only_to_jaxley():
    # test if edge graph can pe imported into to jaxley
    sets_of_edges = [
        [(0, 1), (1, 2), (2, 3)],
        [(0, 1), (1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)],
    ]
    for edges in sets_of_edges:
        edge_graph = nx.DiGraph(edges)
        edge_module = from_graph(edge_graph)


@pytest.mark.parametrize("file", ["morph_single_point_soma.swc", "morph.swc"])
def test_graph_to_jaxley(file):
    # test whether swc file can be imported into jaxley
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)
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


@pytest.mark.parametrize("file", ["morph_single_point_soma.swc", "morph.swc"])
def test_swc2graph_voltages(file):
    """Check if voltages of SWC recording match.

    To match the branch indices between NEURON and jaxley, we rely on comparing the
    length of the branches.

    It tests whether, on average over time and recordings, the voltage is off by less
    than 1.5 mV.
    """
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)  # n120

    nseg = 8

    i_delay = 2.0
    i_dur = 5.0
    i_amp = 0.25
    t_max = 20.0
    dt = 0.025

    ##################### NEURON ##################
    h, neuron_cell = import_neuron_morph(fname, nseg=nseg)

    ####################### jaxley ##################
    graph = swc_to_graph(fname)
    jx_cell = from_graph(
        graph, nseg=nseg, max_branch_len=2000, ignore_swc_trace_errors=False
    )
    jx_cell._update_nodes_with_xyz()
    jx_cell.insert(HH())

    branch_loc = 0.05
    neuron_inds, jaxley_inds = jaxley2neuron_by_group(
        jx_cell, h.allsec(), loc=branch_loc
    )
    trunk_inds, tuft_inds, basal_inds = [
        jaxley_inds[key] for key in ["trunk", "tuft", "basal"]
    ]
    neuron_trunk_inds, neuron_tuft_inds, neuron_basal_inds = [
        neuron_inds[key] for key in ["trunk", "tuft", "basal"]
    ]

    stim_loc = 0.1
    stim_idx = match_stim_loc(jx_cell, h.soma[0], loc=stim_loc)

    jx_cell.set("axial_resistivity", 1_000.0)
    jx_cell.set("v", -62.0)
    jx_cell.set("HH_m", 0.074901)
    jx_cell.set("HH_h", 0.4889)
    jx_cell.set("HH_n", 0.3644787)

    jx_cell.branch.comp(stim_idx).stimulate(
        jx.step_current(i_delay, i_dur, i_amp, dt, t_max)
    )
    for i in trunk_inds + tuft_inds + basal_inds:
        jx_cell.branch(i).loc(branch_loc).record()

    voltages_jaxley = jx.integrate(jx_cell, delta_t=dt)

    ################### NEURON #################
    stim = h.IClamp(h.soma[0](stim_loc))
    stim.delay = i_delay
    stim.dur = i_dur
    stim.amp = i_amp

    counter = 0
    voltage_recs = {}

    for r in neuron_trunk_inds:
        for i, sec in enumerate(h.allsec()):
            if i == r:
                v = h.Vector()
                v.record(sec(branch_loc)._ref_v)
                voltage_recs[f"v{counter}"] = v
                counter += 1

    for r in neuron_tuft_inds:
        for i, sec in enumerate(h.allsec()):
            if i == r:
                v = h.Vector()
                v.record(sec(branch_loc)._ref_v)
                voltage_recs[f"v{counter}"] = v
                counter += 1

    for r in neuron_basal_inds:
        for i, sec in enumerate(h.allsec()):
            if i == r:
                v = h.Vector()
                v.record(sec(branch_loc)._ref_v)
                voltage_recs[f"v{counter}"] = v
                counter += 1

    for sec in h.allsec():
        sec.insert("hh")
        sec.Ra = 1_000.0

        sec.gnabar_hh = 0.120  # S/cm2
        sec.gkbar_hh = 0.036  # S/cm2
        sec.gl_hh = 0.0003  # S/cm2
        sec.ena = 50  # mV
        sec.ek = -77.0  # mV
        sec.el_hh = -54.3  # mV

    h.dt = dt
    tstop = t_max
    v_init = -62.0

    def initialize():
        h.finitialize(v_init)
        h.fcurrent()

    def integrate():
        while h.t < tstop:
            h.fadvance()

    initialize()
    integrate()
    voltages_neuron = np.asarray([voltage_recs[key] for key in voltage_recs])

    ####################### check ################
    errors = np.mean(np.abs(voltages_jaxley - voltages_neuron), axis=1)

    assert all(errors < 1.5), "voltages do not match."
