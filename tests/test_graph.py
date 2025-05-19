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
    _add_missing_graph_attrs,
    build_compartment_graph,
    from_graph,
    to_swc_graph,
)
from jaxley.modules.base import to_graph
from jaxley.morphology import morph_connect, morph_delete
from jaxley.synapses import IonotropicSynapse, TestSynapse

# from jaxley.utils.misc_utils import recursive_compare
from tests.helpers import (
    equal_both_nan_or_empty_df,
    get_segment_xyzrL,
    import_neuron_morph,
)


# test exporting and re-importing of different modules
def test_graph_import_export_cycle(
    SimpleComp, SimpleBranch, SimpleCell, SimpleNet, SimpleMorphCell
):
    """Export and import a module and check whether .nodes and .edges remain same."""
    np.random.seed(0)
    comp = SimpleComp()
    branch = SimpleBranch(4)
    cell = SimpleCell(5, 4)
    morph_cell = SimpleMorphCell(ncomp=1)
    net = SimpleNet(3, 5, 4)

    # add synapses
    connect(net[0, 0, 0], net[1, 0, 0], IonotropicSynapse())
    connect(net[0, 0, 1], net[1, 0, 1], IonotropicSynapse())
    # connect(net[0, 0, 1], net[1, 0, 1], TestSynapse()) # makes test fail, see warning w. synapses = True

    # add groups
    net.cell(2).add_to_group("cell2")
    net.cell(2).branch(1).add_to_group("cell2branch1")

    # add ion channels
    net.cell(0).insert(Na())
    net.cell(0).insert(Leak())
    net.cell(1).branch(1).insert(Na())
    net.cell(0).insert(K())

    # test consistency of exported and re-imported modules
    for module in [comp, branch, cell, net, morph_cell]:
        module.compute_xyz()  # ensure x,y,z in nodes b4 exporting for later comparison
        module.compute_compartment_centers()

        # ensure to_graph works
        module_graph = to_graph(module, channels=True, synapses=True)

        # ensure prev exported graph can be read
        re_module = from_graph(module_graph, traverse_for_solve_order=False)

        # ensure to_graph works for re-imported modules
        re_module_graph = to_graph(re_module, channels=True, synapses=True)

        # ensure original module and re-imported module are equal
        assert np.all(equal_both_nan_or_empty_df(re_module.nodes, module.nodes))
        assert np.all(equal_both_nan_or_empty_df(re_module.edges, module.edges))
        assert np.all(
            equal_both_nan_or_empty_df(re_module.branch_edges, module.branch_edges)
        )

        for k in module.group_names:
            assert k in re_module.group_names

        for re_xyzr, xyzr in zip(re_module.xyzr, module.xyzr):
            re_xyzr[np.isnan(re_xyzr)] = -1
            xyzr[np.isnan(xyzr)] = -1

            assert np.all(re_xyzr == xyzr)

        re_imported_mechs = re_module.channels + re_module.synapses
        for re_mech, mech in zip(re_imported_mechs, module.channels + module.synapses):
            assert np.all(re_mech.name == mech.name)

        # ensure exported graph and re-exported graph are equal
        node_df = pd.DataFrame(
            [d for i, d in module_graph.nodes(data=True)],
            index=module_graph.nodes,
        )
        node_df = node_df.loc[node_df["type"] != "branchpoint"].sort_index()

        re_node_df = pd.DataFrame(
            [d for i, d in re_module_graph.nodes(data=True)],
            index=re_module_graph.nodes,
        )
        re_node_df = re_node_df.loc[re_node_df["type"] != "branchpoint"].sort_index()
        assert np.all(equal_both_nan_or_empty_df(node_df, re_node_df))

        edges = pd.DataFrame(
            [
                {
                    "pre_index": i,
                    "post_index": j,
                    **module_graph.edges[i, j],
                }
                for (i, j) in module_graph.edges
            ]
        )
        re_edges = pd.DataFrame(
            [
                {
                    "pre_index": i,
                    "post_index": j,
                    **re_module_graph.edges[i, j],
                }
                for (i, j) in re_module_graph.edges
            ]
        )
        assert np.all(equal_both_nan_or_empty_df(edges, re_edges))

        # ignore "externals", "recordings", "trainable_params", "indices_set_by_trainables"
        for k in ["ncomp"]:
            assert module_graph.graph[k] == re_module_graph.graph[k]

        # assume if module can be integrated, so can be comp, cell and branch
        if isinstance(module, jx.Network):
            # test integration of re-imported module
            re_module.select(nodes=0).record(verbose=False)
            jx.integrate(re_module, t_max=0.5)


@pytest.mark.parametrize(
    "file",
    [
        "morph_3_types_single_point_soma.swc",
        "morph_3_types.swc",
        "morph_interrupted_soma.swc",
        "morph_soma_both_ends.swc",
        "morph_somatic_branchpoint.swc",
        "morph_non_somatic_branchpoint.swc",
        "morph_ca1_n120_single_point_soma.swc",
        "morph_ca1_n120.swc",
        "morph_l5pc_with_axon.swc",
        "morph_allen_485574832.swc",
        pytest.param(
            "morph_flywire_t4_720575940626407426.swc",
            marks=pytest.mark.xfail(reason="NEURON throws .hoc error."),
        ),
        pytest.param(
            "morph_retina_20161028_1.swc",
            marks=pytest.mark.xfail(reason="Branch with l=0. Jaxley: 0.1, NEURON: 0.0"),
        ),
    ],
)
def test_trace_branches(file):
    """Test whether all branch lengths match NEURON."""
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)
    swc_graph = to_swc_graph(fname)

    # pre-processing
    comp_graph = build_compartment_graph(
        swc_graph, ncomp=1, ignore_swc_tracing_interruptions=False
    )

    nx_branch_lens = []
    for n in comp_graph.nodes:
        if comp_graph.nodes[n]["type"] == "comp":
            nx_branch_lens.append(comp_graph.nodes[n]["length"])
    nx_branch_lens = np.sort(nx_branch_lens)

    h, _ = import_neuron_morph(fname)
    neuron_branch_lens = np.sort([sec.L for sec in h.allsec()])

    errors = np.abs(neuron_branch_lens - nx_branch_lens)
    assert sum(errors > 1e-3) == 0


@pytest.mark.parametrize(
    "file",
    [
        "morph_ca1_n120_single_point_soma.swc",
        "morph_ca1_n120.swc",
        "morph_l5pc_with_axon.swc",
    ],
)
def test_from_graph_vs_NEURON(file):
    """Check whether comp xyzr match neuron."""
    ncomp = 8
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)

    swc_graph = to_swc_graph(fname)
    comp_graph = build_compartment_graph(
        swc_graph,
        ncomp=ncomp,
        max_len=2000,
        ignore_swc_tracing_interruptions=False,
    )
    cell = from_graph(comp_graph)
    cell.compute_compartment_centers()
    h, neuron_cell = import_neuron_morph(fname, ncomp=ncomp)

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

    assert sum(errors.groupby("jx_idx")["xyz"].max() > 1e-3) == 0
    assert sum(errors.groupby("jx_idx")["radius"].max() > 1e-3) == 0
    assert sum(errors.groupby("jx_idx")["length"].max() > 1e-3) == 0


def test_edges_only_to_jaxley():
    """Test if edge graph can pe imported into to jaxley."""
    sets_of_edges = [
        [(0, 1), (1, 2), (2, 3)],
        [(0, 1), (1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)],
    ]
    for edges in sets_of_edges:
        graph = nx.Graph(edges)
        swc_graph = _add_missing_graph_attrs(graph)
        comp_graph = build_compartment_graph(swc_graph, ncomp=1, min_radius=1.0)
        edge_module = from_graph(comp_graph)


@pytest.mark.parametrize("ncomp", [1, 3])
def test_morph_delete(ncomp: int):
    """Test correctness of `nodes` and voltages after `morph_delete`."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=ncomp)
    cell = jx.Cell(branch, parents=[-1, 0, 0])
    cell.branch(0).set("length", 50.0)
    cell = morph_delete(cell.branch(2))
    cell.insert(HH())

    cell2 = jx.Cell(branch, parents=[-1, 0])
    cell2.branch(0).set("length", 50.0)
    cell2.insert(HH())

    cell[0, 0].record()
    cell[0, 0].stimulate(0.1 * jnp.ones((100,)))
    cell2[0, 0].record()
    cell2[0, 0].stimulate(0.1 * jnp.ones((100,)))

    v1 = jx.integrate(cell)
    v2 = jx.integrate(cell2)
    assert np.max(np.abs(v1 - v2)) < 1e-8, "voltages do not match."

    # Drop xyz because the first cell had branches that form a "star", so even
    # after deleting a branch we do not expect xyz to be a straight line.
    assert np.all(
        equal_both_nan_or_empty_df(
            cell.nodes.drop(columns=["x", "y", "z"]),
            cell2.nodes.drop(columns=["x", "y", "z"]),
        )
    )


@pytest.mark.parametrize("ncomp", [1, 3])
def test_morph_attach(ncomp: int):
    """Test correctness of `nodes` and voltages after `morph_attach`."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=ncomp)
    cell = jx.Cell(branch, parents=[-1, 0])
    cell.insert(Leak())
    stub = jx.Cell(branch, parents=[-1])
    stub.set("length", 80.0)
    stub.insert(HH())
    cell = morph_connect(cell.branch(1).loc(0.0), stub.branch(0).loc(0.0))

    cell2 = jx.Cell(branch, parents=[-1, 0, 0])
    cell2.branch(2).set("length", 80.0)
    cell2.branch(2).insert(HH())
    cell2.branch([0, 1]).insert(Leak())

    cell[0, 0].record()
    cell[0, 0].stimulate(0.1 * jnp.ones((100,)))
    cell2[0, 0].record()
    cell2[0, 0].stimulate(0.1 * jnp.ones((100,)))

    v1 = jx.integrate(cell)
    v2 = jx.integrate(cell2)
    assert np.max(np.abs(v1 - v2)) < 1e-8, "voltages do not match."

    # Drop xyz because the first cell had branches that form a "star", so even
    # after deleting a branch we do not expect xyz to be a straight line.
    assert np.all(
        equal_both_nan_or_empty_df(
            cell.nodes.drop(columns=["x", "y", "z"]),
            cell2.nodes.drop(columns=["x", "y", "z"]),
        )
    )


@pytest.mark.parametrize("ncomp", [1, 2])
def test_morph_edit_swc(ncomp: int):
    """Check whether we get NaN after having deleted and added things to SWC."""
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", "morph_l5pc_with_axon.swc")
    cell = jx.read_swc(fname, ncomp=ncomp, backend="graph")
    cell = morph_delete(cell.axon)
    cell = morph_delete(cell.apical)

    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=ncomp)
    stub = jx.Cell(branch, parents=[-1])
    stub.set("length", 100.0)
    stub.add_to_group("stub")  # To more easily find the stub later.

    # Implicitly also tests whether it can be combined with groups (`.soma`), and
    # whether branchpoint nodes _and_ tip nodes work (branchpoint node for `cell`, tip
    # for `stub`).
    cell = morph_connect(cell.soma.branch(0).loc(1.0), stub.branch(0).loc(0.0))

    # Modify a bit and run a simulation.
    cell.stub.set_ncomp(4)
    cell.branch(3).set_ncomp(2)

    # Channels and initialization.
    cell.soma.insert(HH())
    cell.insert(Leak())
    cell.set("v", -65.0)
    cell.init_states()

    # Simulation.
    cell[0, 0].record("v")
    cell.stub.branch(0).comp(3).record("v")
    cell.soma.branch(0).comp(0).stimulate(jx.step_current(10.0, 5.0, 0.2, 0.025, 100.0))
    v = jx.integrate(cell)

    assert np.invert(np.any(np.isnan(v))), "Found NaN"


def test_trim_dendrites_of_swc():
    """This function tests whether we can successfully trim dendrites.

    It is just an API test and does not check for correctness.

    When the morphology is being trimmed, it deletes node [0] which had caused issues
    at some point.
    """

    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", "morph_ca1_n120.swc")
    swc_graph = to_swc_graph(fname)
    comp_graph = build_compartment_graph(swc_graph, ncomp=1)

    # Next, we loop over all nodes. We want to keep nodes only if they made any of the
    # following conditions:
    # - if a node has more than one neighbor (`degree > 1`),
    # - if its compartment length is > 250 $\mu$m, or
    # - if it is a soma.
    nodes_to_keep = []
    for node in comp_graph.nodes:
        degree = comp_graph.in_degree(node) + comp_graph.out_degree(node)

        condition1 = degree > 1
        condition2 = comp_graph.nodes[node]["length"] > 250.0
        condition3 = "soma" in comp_graph.nodes[node]["groups"]
        if condition1 or condition2 or condition3:
            nodes_to_keep.append(node)

    comp_graph = nx.subgraph(comp_graph, nodes_to_keep)
    cell = from_graph(comp_graph)
    cell.delete_recordings()
    cell.delete_stimuli()
    cell.soma.branch(0).comp(0).record()
    cell.soma.branch(0).comp(0).stimulate(
        jx.step_current(10.0, 20.0, 0.1, 0.025, 100.0)
    )
    v = jx.integrate(cell)
    assert np.invert(np.any(np.isnan(v))), "Found a NaN in the voltage."
