# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

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
    add_missing_graph_attrs,
    from_graph,
    make_jaxley_compatible,
    swc_to_graph,
    to_graph,
    trace_branches,
)
from jaxley.synapses import IonotropicSynapse, TestSynapse

# from jaxley.utils.misc_utils import recursive_compare
from tests.helpers import (
    get_segment_xyzrL,
    import_neuron_morph,
    jaxley2neuron_by_group,
    match_stim_loc,
)


# test exporting and re-importing of different modules
def test_graph_import_export_cycle(
    SimpleComp, SimpleBranch, SimpleCell, SimpleNetwork, SimpleMorphCell
):
    # build a network
    np.random.seed(0)
    comp = SimpleComp()
    branch = SimpleBranch(4)
    cell = SimpleCell(5, 4)
    morph_cell = SimpleMorphCell()
    net = SimpleNetwork(3, 5, 4)

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

    # test consistency of exported and re-imported modules
    for module in [net, morph_cell, cell, branch, comp]:
        module.compute_xyz()  # ensure x,y,z in nodes b4 exporting for later comparison
        module_graph = to_graph(module)  # ensure to_graph works
        re_module = from_graph(module_graph)  # ensure prev exported graph can be read
        re_module_graph = to_graph(
            re_module
        )  # ensure to_graph works for re-imported modules

        # TODO: ensure modules are equal
        # compare_modules(module, re_module)

        # TODO: ensure graphs are equal

        # TODO: test if imported module can be simulated
        # if isinstance(module, jx.Network):
        #     jx.integrate(re_module)


@pytest.mark.parametrize("file", ["morph_single_point_soma.swc", "morph.swc"])
def test_trace_branches(file):
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)
    graph = swc_to_graph(fname)

    # pre-processing
    graph = add_missing_graph_attrs(graph)
    graph = trace_branches(graph, None, ignore_swc_trace_errors=False)

    edges = pd.DataFrame([{"u": u, "v": v, **d} for u, v, d in graph.edges(data=True)])
    nx_branch_lens = edges.groupby("branch_index")["l"].sum().to_numpy()
    nx_branch_lens = np.sort(nx_branch_lens)

    # exclude artificial root branch
    if np.isclose(nx_branch_lens[0], 1e-1):
        nx_branch_lens = nx_branch_lens[1:]

    h, _ = import_neuron_morph(fname)
    neuron_branch_lens = np.sort([sec.L for sec in h.allsec()])

    errors = np.abs(neuron_branch_lens - nx_branch_lens)
    # one error is expected, see https://github.com/jaxleyverse/jaxley/issues/140
    assert sum(errors > 1e-3) <= 1


@pytest.mark.parametrize("file", ["morph_single_point_soma.swc", "morph.swc"])
def test_from_graph_vs_NEURON(file):
    ncomp = 8
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)

    graph = swc_to_graph(fname)
    cell = from_graph(
        graph, ncomp=ncomp, max_branch_len=2000, ignore_swc_trace_errors=False
    )
    cell.compute_compartment_centers()
    h, neuron_cell = import_neuron_morph(fname, nseg=ncomp)

    # remove root branch
    jaxley_comps = cell.nodes[
        ~np.isclose(cell.nodes["length"], 0.1 / ncomp)
    ].reset_index(drop=True)

    jx_branch_lens = (
        jaxley_comps.groupby("global_branch_index")["length"].sum().to_numpy()
    )

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
                for i in range(ncomp)
            ]
        )
        # make this a dataframe
        neuron_comp_k = pd.DataFrame(
            neuron_comp_k, columns=["x", "y", "z", "radius", "length"]
        )
        neuron_comp_k["idx"] = neuron_inds[k]
        jx_comp_k = jaxley_comps[jaxley_comps["global_branch_index"] == jx_inds[k]][
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
        assert group in swc_module.groups

    # test import after different stages of graph pre-processing
    graph = swc_to_graph(fname)
    module_imported_directly = from_graph(deepcopy(graph))

    graph = make_jaxley_compatible(deepcopy(graph))
    module_imported_after_preprocessing = from_graph(graph)

    # TODO:
    # compare_modules(module_imported_directly, module_imported_after_preprocessing)


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

    ncomp = 8

    i_delay = 2.0
    i_dur = 5.0
    i_amp = 0.25
    t_max = 20.0
    dt = 0.025

    ##################### NEURON ##################
    h, neuron_cell = import_neuron_morph(fname, nseg=ncomp)

    ####################### jaxley ##################
    graph = swc_to_graph(fname)
    jx_cell = from_graph(
        graph, ncomp=ncomp, max_branch_len=2000, ignore_swc_trace_errors=False
    )
    jx_cell.compute_compartment_centers()
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

    jx_cell.select(stim_idx).stimulate(
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
