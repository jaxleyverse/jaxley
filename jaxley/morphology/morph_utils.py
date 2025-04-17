# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import networkx as nx
import numpy as np

from jaxley.io.graph import connect_graphs, from_graph, to_graph


def morph_delete(module_view) -> "Cell":
    """Deletes part of a morphology.

    Args:
        module_view: View of a `jx.Module`. Defines the compartments to be deleted.

    Returns:
        A cell in which specified compartments are deleted.

    Example:
    --------

    ::

        cell = jx.read_swc("path_to_swc_file.swc", ncomp=1)
        cell = morph_delete(cell.axon)
    """
    # If the user did not run `compute_xyz` or `compute_compartment_centers`, we run
    # it automatically.
    if np.isnan(module_view.base.xyzr[0][0, 0]):
        module_view.base.compute_xyz()
    if "x" not in module_view.base.nodes.columns:
        module_view.base.compute_compartment_centers()

    comps_to_delete = module_view.nodes.index
    comp_graph = to_graph(module_view.base)

    nodes_to_keep = []
    for node in comp_graph.nodes:
        # We keep all `branchpoints`. The ones that have degree <2 will automatically
        # be trimmed away `_remove_branch_points_at_tips`, which is run during
        # `from_graph()`.
        if (
            comp_graph.nodes[node]["type"] == "branchpoint"
            or comp_graph.nodes[node]["comp_index"] not in comps_to_delete
        ):
            nodes_to_keep.append(node)

    comp_graph = nx.subgraph(comp_graph, nodes_to_keep)
    cell = from_graph(comp_graph)
    return cell


def morph_connect(module_view1, module_view2) -> "Cell":
    """Combine two morphologies into a single cell.

    Args:
        module_view1: The view of a ``jx.Cell()``. Must have been created with a
            command ending on ``loc(0.0)`` or ``loc(1.0)``. For example, the following
            are valid:
            ``cell.branch(0).loc(0.0)``, ``cell.branch(5).loc(0.0)``,
            ``cell.branch(5).loc(1.0)``.
            But those are not valid:
            ``cell.branch(0).comp(0)`` (uses ``.comp``), ``cell.branch(5).loc(0.9)``
            (does not use ``loc(0.0)`` or ``loc(1.0)``).
        module_view2: The view of a ``jx.Cell()``. Must follow the same rules as
            ``module_view1``.

    Returns:
        A ``jx.Cell`` which is made up of both input cells.

    Example:
    --------

    ::

        cell = jx.read_swc("path_to_swc_file.swc", ncomp=1)
        stub = jx.Cell()
        cell = morph_connect(cell.branch(0).loc(0.0), stub.branch(0).loc(0.0))
    """
    # If the user did not run `compute_xyz` or `compute_compartment_centers`, we run
    # it automatically.
    for view in [module_view1, module_view2]:
        if np.isnan(view.base.xyzr[0][0, 0]):
            view.base.compute_xyz()
        if "x" not in view.base.nodes.columns:
            view.base.compute_compartment_centers()

    graph1 = to_graph(module_view1.base)
    graph2 = to_graph(module_view2.base)

    comps = []
    for view in [module_view1, module_view2]:
        global_comp = int(view.nodes.index.to_numpy()[0])
        if len(view._comp_edges) == 0:
            # Connect to the tip of a branch.
            comps.append(global_comp)
        else:
            # Connect to a branchpoint.
            # Branchpoint counters are stricly higher than compartment counters,
            # so we can simply take the `max()` to get the branchpoint index.
            comps.append(int(view._comp_edges["sink"].max()))

    combined_graph = connect_graphs(graph1, graph2, comps[0], comps[1])
    cell = from_graph(combined_graph)
    return cell
