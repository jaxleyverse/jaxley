# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
import os

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from jaxley_mech.channels.pospischil import K, Leak, Na

import jaxley as jx
from jaxley import connect
from jaxley.channels import HH
from jaxley.io.graph import (
    build_compartment_graph,
    from_graph,
    nx_to_swc,
    swc_to_nx,
    swc_to_pandas,
    to_graph,
)
from jaxley.morphology import morph_connect, morph_delete
from jaxley.synapses import IonotropicSynapse
from tests.helpers import equal_both_nan_or_empty_df, import_neuron_morph

# `build_compartment_graph()` asserts on these before it gets to trace anything.
NEEDS_MIN_RADIUS = pytest.mark.xfail(
    raises=AssertionError, reason="Radius 0.0 in SWC file, needs `min_radius`."
)

# The five listed files have soma has branches leaving from both ends. Since a Module stores only parent/child relations
# (comb_parents, _par_inds), it never records which end of the parent a child attached to. So _branchpoints_and_tips_of
# can only place the branchpoints at the parent's last point then graph.py emits a tip at the root's start.
ROOT_JUNCTION_DIFFERS = {
    "morph_ca1_n120.swc",
    "morph_interrupted_soma.swc",
    "morph_non_somatic_branchpoint.swc",
    "morph_soma_both_ends.swc",
    "morph_somatic_branchpoint.swc",
}

MORPHOLOGIES = [
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
    pytest.param("morph_flywire_t4_720575940626407426.swc", marks=NEEDS_MIN_RADIUS),
    pytest.param("morph_retina_20161028_1.swc", marks=NEEDS_MIN_RADIUS),
]

# The same files without the marks, for tests that do not compartmentalize and so are not
# bothered by a zero radius.
ALL_MORPHOLOGIES = [f if isinstance(f, str) else f.values[0] for f in MORPHOLOGIES]


def swc_path(file: str) -> str:
    return os.path.join(os.path.dirname(__file__), "swc_files", file)


def branchpoints_and_tips(graph: nx.Graph) -> list:
    """Node names of a compartment graph that are not compartments."""
    return [
        n for n, d in graph.nodes(data=True) if pd.isna(d.get("global_branch_index"))
    ]


def node_frame(graph: nx.Graph) -> pd.DataFrame:
    """Node attributes of a graph, as a DataFrame indexed by node name."""
    return pd.DataFrame.from_dict(
        dict(graph.nodes(data=True)), orient="index"
    ).sort_index()


def edge_frame(graph: nx.Graph) -> pd.DataFrame:
    """Edge attributes of a graph, indexed by the edge with its ends in ascending order."""
    df = nx.to_pandas_edgelist(graph)
    if df.empty:
        return df
    lo = np.minimum(df["source"], df["target"])
    hi = np.maximum(df["source"], df["target"])
    df["source"], df["target"] = lo, hi
    return df.set_index(["source", "target"]).sort_index()


def build_test_net(SimpleNet):
    """A network with synapses, groups and channels, to exercise every exported attr."""
    net = SimpleNet(3, 5, 4)
    connect(net[0, 0, 0], net[1, 0, 0], IonotropicSynapse())
    connect(net[0, 0, 1], net[1, 0, 1], IonotropicSynapse())

    net.cell(2).add_to_group("cell2")
    net.cell(2).branch(1).add_to_group("cell2branch1")

    net.cell(0).insert(Na())
    net.cell(0).insert(Leak())
    net.cell(1).branch(1).insert(Na())
    net.cell(0).insert(K())
    return net


@pytest.mark.parametrize("file", ALL_MORPHOLOGIES)
def test_swc_nx_round_trip(file):
    """`swc_to_nx` and `nx_to_swc` must be inverse at the graph level.

    Failure cases:
    - `swc_to_nx` maps every `id` outside `relevant_ids` to 0, so a file using other ids
      (`morph_flywire_t4_*` has 0, 1, 5, 6) cannot round-trip its `id` column.
    - `nx_to_swc` writes `p` from a BFS rooted at the lowest node index, so a file with a
      second root (`morph_multiple_roots.swc`) comes back with that root re-parented.
    """
    swc = swc_to_pandas(swc_path(file))
    graph = swc_to_nx(swc)
    re_graph = swc_to_nx(nx_to_swc(graph))

    assert sorted(graph.nodes) == sorted(re_graph.nodes)
    assert [type(n) for n in sorted(graph.nodes)] == [
        type(n) for n in sorted(re_graph.nodes)
    ], "node labels changed type, which silently splits int and float labels"
    edges = lambda g: sorted(map(tuple, map(sorted, g.edges)))
    assert edges(graph) == edges(re_graph)
    for node in graph.nodes:
        assert graph.nodes[node] == re_graph.nodes[node], node


def test_module_graph_module_round_trip(
    SimpleComp, SimpleBranch, SimpleCell, SimpleNet, SimpleMorphCell
):
    """`from_graph(to_graph(module))` must reproduce the module exactly."""
    np.random.seed(0)
    modules = [
        SimpleComp(),
        SimpleBranch(4),
        SimpleCell(5, 4),
        build_test_net(SimpleNet),
        SimpleMorphCell(ncomp=1),
    ]

    for module in modules:
        re_module = from_graph(to_graph(module))

        # `equal_both_nan_or_empty_df` is insensitive to column order,
        # which the synapse columns do not preserve.
        assert np.all(equal_both_nan_or_empty_df(re_module.nodes, module.nodes))
        assert np.all(equal_both_nan_or_empty_df(re_module.edges, module.edges))
        assert np.all(
            equal_both_nan_or_empty_df(re_module.branch_edges, module.branch_edges)
        )
        assert np.all(
            equal_both_nan_or_empty_df(re_module._comp_edges, module._comp_edges)
        )
        # Only the branchpoint names, not their coordinates: on a module built from scratch
        # `compute_compartment_centers()` overwrites `_branchpoints` x/y/z with the mean of
        # the neighbouring compartment centers, whereas the re-imported module keeps the
        # branch end the branchpoint actually sits on. See `_branchpoints_and_tips_of`.
        assert re_module._branchpoints.index.equals(module._branchpoints.index)

        for k in module.group_names:
            assert k in re_module.group_names

        assert len(re_module.xyzr) == len(module.xyzr)
        for re_xyzr, xyzr in zip(re_module.xyzr, module.xyzr):
            assert np.allclose(re_xyzr, xyzr, equal_nan=True)

        re_mechs = re_module.channels + re_module.synapses
        for re_mech, mech in zip(re_mechs, module.channels + module.synapses):
            assert re_mech.name == mech.name

        # Assume that if the network integrates, so do comp, branch and cell.
        if isinstance(module, jx.Network):
            re_module.select(nodes=0).record(verbose=False)
            jx.integrate(re_module, t_max=0.5)


def test_module_graph_round_trip_preserves_synapse_dtypes(SimpleNet):
    """The synapse index columns must not change to `object` on the way through a graph.

    The compartment edges carry no synapse indices, so those columns hold a mix of `pd.NA`
    and integers while in the graph. Only the synapse rows survive the import, so they have
    to be cast back.
    """
    net = build_test_net(SimpleNet)
    re_net = from_graph(to_graph(net))

    for col in ["global_edge_index", "index_within_type", "type_ind"]:
        assert re_net.edges[col].dtype == net.edges[col].dtype, col
        assert re_net.edges[col].equals(net.edges[col]), col


@pytest.mark.parametrize("channels", [True, False])
@pytest.mark.parametrize("synapses", [True, False])
def test_to_graph(SimpleNet, channels, synapses):
    """Check what `to_graph` exports, and that `channels`/`synapses` can be left out."""
    net = build_test_net(SimpleNet)
    graph = to_graph(net, channels=channels, synapses=synapses)

    non_comps = branchpoints_and_tips(graph)
    tips = [n for n in non_comps if graph.degree(n) == 1]
    assert graph.number_of_nodes() - len(non_comps) == len(net.nodes)
    assert len(non_comps) - len(tips) == len(net._branchpoints)

    # Every compartment-to-compartment edge of the module survives the export.
    comp_edges = net._comp_edges[net._comp_edges["type"] == 0]
    exported = {frozenset(e) for e in graph.edges}
    for source, sink in comp_edges[["source", "sink"]].to_numpy():
        assert frozenset((int(source), int(sink))) in exported, (source, sink)

    for attr in ["xyzr", "channels", "synapses", "group_names", "pumps"]:
        assert attr in graph.graph, attr
    assert graph.graph["module"] == "network"

    assert (len(graph.graph["channels"]) > 0) == channels
    assert (len(graph.graph["synapses"]) > 0) == synapses

    # A channel owns one boolean column plus one per parameter.
    channel_cols = {c._name for c in net.channels}
    channel_cols |= {p for c in net.channels for p in c.channel_params}
    assert bool(set(node_frame(graph).columns) & channel_cols) == channels

    # A synapse edge is the only edge with `synapse == True`.
    has_synapse_edge = any(d.get("synapse") for _, _, d in graph.edges(data=True))
    assert has_synapse_edge == synapses

    re_net = from_graph(graph)
    assert len(re_net.nodes) == len(net.nodes)
    assert (len(re_net.edges) > 0) == synapses


def test_graph_re_export(SimpleComp, SimpleBranch, SimpleCell, SimpleNet):
    """Exporting an imported graph must give back the very same graph."""
    np.random.seed(0)
    modules = [
        SimpleComp(),
        SimpleBranch(4),
        SimpleCell(5, 4),
        build_test_net(SimpleNet),
    ]

    for module in modules:
        graph = to_graph(module)
        re_graph = to_graph(from_graph(graph))

        # `graphs_equal` also compares `G.graph`, which holds channel objects and the
        # `xyzr` list. Those do not compare with `==`, so check them by hand and strip them
        # before comparing the nodes and edges.
        assert set(graph.graph) == set(re_graph.graph)
        for attr in ["channels", "synapses", "pumps"]:
            assert [type(m) for m in graph.graph[attr]] == [
                type(m) for m in re_graph.graph[attr]
            ], attr
        assert graph.graph["group_names"] == re_graph.graph["group_names"]
        assert len(graph.graph["xyzr"]) == len(re_graph.graph["xyzr"])
        for a, b in zip(graph.graph["xyzr"], re_graph.graph["xyzr"]):
            assert np.allclose(a, b, equal_nan=True)

        assert sorted(graph.nodes) == sorted(re_graph.nodes)
        assert equal_both_nan_or_empty_df(node_frame(graph), node_frame(re_graph))
        assert equal_both_nan_or_empty_df(edge_frame(graph), edge_frame(re_graph))


@pytest.mark.parametrize("ncomp", [1, 3])
@pytest.mark.parametrize("file", MORPHOLOGIES)
def test_to_graph_rebuilds_compartment_graph(file, ncomp):
    """`to_graph()` must rebuild the graph the module was imported from.

    The branchpoints and tips are not stored on the module; they are rebuilt from `xyzr`.
    So this checks that nothing is lost by not storing them.
    """
    fname = swc_path(file)
    comp_graph = build_compartment_graph(
        swc_to_nx(swc_to_pandas(fname)),
        ncomp=ncomp,
        max_len=2_000.0,
        ignore_swc_tracing_interruptions=False,
    )
    cell = jx.read_swc(
        fname,
        ncomp=ncomp,
        backend="graph",
        max_branch_len=2_000.0,
        ignore_swc_tracing_interruptions=False,
    )
    graph = to_graph(cell)

    assert graph.number_of_nodes() == comp_graph.number_of_nodes()
    assert graph.number_of_edges() == comp_graph.number_of_edges()
    assert nx.is_forest(graph)

    # The degrees only agree where the root branch does not carry a branchpoint at its
    # start. Asserted either way, so that closing that gap fails here and gets noticed.
    degrees = lambda g: sorted(dict(g.degree()).values())
    if file in ROOT_JUNCTION_DIFFERS:
        assert degrees(graph) != degrees(comp_graph), "root junction gap seems fixed"
    else:
        assert degrees(graph) == degrees(comp_graph)

    # Compartments are the nodes with a `branch_index`, everything else is a branchpoint
    # or a tip. Tips are the ones with a single neighbour, and the rest must be exactly the
    # branchpoints that the module tracks in `_branchpoints`.
    non_comps = branchpoints_and_tips(graph)
    tips = [n for n in non_comps if graph.degree(n) == 1]
    assert graph.number_of_nodes() - len(non_comps) == len(cell.nodes)
    assert len(non_comps) - len(tips) == len(cell._branchpoints)


@pytest.mark.parametrize("ncomp", [1, 3])
def test_to_graph_after_editing_the_morphology(ncomp):
    """Branchpoints and tips are derived, so they must follow an edited branch structure.

    Anything that renumbers compartments or changes which branches have children can put the
    derived nodes out of step with `_comp_edges`.
    """
    branch = jx.Branch(jx.Compartment(), ncomp=ncomp)
    stub = jx.Cell(branch, parents=[-1])

    edited = {
        "morph_delete": morph_delete(jx.Cell(branch, parents=[-1, 0, 0]).branch(2)),
        "morph_connect": morph_connect(
            jx.Cell(branch, parents=[-1, 0]).branch(1).loc(0.0), stub.branch(0).loc(0.0)
        ),
    }
    cell = jx.Cell(branch, parents=[-1, 0, 0])
    cell.branch(1).set_ncomp(5)
    edited["set_ncomp"] = cell

    for name, module in edited.items():
        graph = to_graph(module)
        non_comps = branchpoints_and_tips(graph)
        tips = [n for n in non_comps if graph.degree(n) == 1]

        assert nx.is_forest(graph), name
        assert nx.is_connected(graph), name
        assert graph.number_of_nodes() - len(non_comps) == len(module.nodes), name
        assert len(non_comps) - len(tips) == len(module._branchpoints), name

        re_module = from_graph(graph)
        assert np.all(equal_both_nan_or_empty_df(re_module.nodes, module.nodes)), name
        assert np.all(
            equal_both_nan_or_empty_df(re_module._comp_edges, module._comp_edges)
        ), name


# NOTE: comp length, radius, area and volume are checked against NEURON for every backend
# in `test_swc.py::test_swc_morph_params_vs_neuron`, and the comp centers in
# `test_swc.py::test_comp_centers_agree_across_backends`. Both compare per attribute or by
# nearest center, which is why there is no such test here: pairing jaxley branches to
# NEURON sections by sorted branch length is unstable on real reconstructions and ends up
# comparing unrelated branches.


@pytest.mark.parametrize("file", MORPHOLOGIES)
def test_trace_branches(file):
    """Test whether all branch lengths match NEURON."""
    fname = swc_path(file)

    # These two are traced with an interruption that NEURON does not split on.
    ignore_swc_interrupts = file in [
        "morph_somatic_branchpoint.swc",
        "morph_non_somatic_branchpoint.swc",
    ]
    comp_graph = build_compartment_graph(
        swc_to_nx(swc_to_pandas(fname)),
        ncomp=1,
        ignore_swc_tracing_interruptions=ignore_swc_interrupts,
    )

    # With `ncomp=1` every compartment is a whole branch. Branchpoints and tips are marked
    # by a NaN `branch_index` and carry no length.
    nx_branch_lens = np.sort(
        [
            comp_graph.nodes[n]["length"]
            for n in comp_graph.nodes
            if pd.notna(comp_graph.nodes[n]["global_branch_index"])
        ]
    )

    h, _ = import_neuron_morph(fname)
    neuron_branch_lens = np.sort([sec.L for sec in h.allsec()])

    errors = np.abs(neuron_branch_lens - nx_branch_lens)
    assert sum(errors > 1e-3) == 0


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
    fname = swc_path("morph_l5pc_with_axon.swc")
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
    fname = swc_path("morph_ca1_n120.swc")
    comp_graph = build_compartment_graph(swc_to_nx(swc_to_pandas(fname)), ncomp=1)

    # Next, we loop over all nodes. We want to keep nodes only if they made any of the
    # following conditions:
    # - if a node has more than one neighbor (`degree > 1`),
    # - if its compartment length is > 250 $\mu$m, or
    # - if it is a soma.
    nodes_to_keep = []
    for node in comp_graph.nodes:
        degree = comp_graph.degree(node)

        condition1 = degree > 1
        condition2 = comp_graph.nodes[node]["length"] > 250.0
        condition3 = comp_graph.nodes[node]["id"] == 1
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
