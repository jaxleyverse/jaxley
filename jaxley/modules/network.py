# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import itertools
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union
from warnings import warn

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import vmap
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from jaxley.modules.base import Module
from jaxley.modules.cell import Cell
from jaxley.utils.cell_utils import (
    compute_children_and_parents,
    convert_point_process_to_distributed,
    loc_of_index,
)
from jaxley.utils.misc_utils import concat_and_ignore_empty, cumsum_leading_zero
from jaxley.utils.solver_utils import dhs_group_comps_into_levels
from jaxley.utils.syn_utils import gather_synapes


class Network(Module):
    """A network made up of multiple cells, connected by synapses.

    This class defines a network of cells. These cells can later on be connected with
    synapses via `jx.connect`.
    """

    network_params: Dict = {}
    network_states: Dict = {}

    def __init__(
        self,
        cells: List[Cell],
        vectorize_cells: Optional[bool] = None,
    ):
        """Initialize network of cells and synapses.

        Args:
            cells: A list of cells that make up the network.
            vectorize_cells: Whether to vectorize the voltage updates across cells.
                If `None`, then we default to `True` on GPU and to `False` on CPU.
                Manually setting this to `True` on CPU has two effects: (1) The voltage
                equations for each cell are solved in parallel. (2) The
                `net._solver_device` is set to `gpu`, regardless of which device you
                are on. Because of this, the voltage update is unrolled, which leads
                to higher compile time, but potentially lower runtime. Manually
                setting this to `False` on GPU will typically lead to very poor
                performance, and will only work if you had previously run
                `cell._init_solver_jaxley_dhs_solve()` for every cell in the network.
                We note that channels are always vectorized, regardless of the value of
                `vectorize_cells`.
        """
        super().__init__()
        if vectorize_cells is not None:
            self._solver_device = "gpu" if vectorize_cells else "cpu"

        for cell in cells:
            self.xyzr += deepcopy(cell.xyzr)

        self._cells_list = cells
        self.ncomp_per_branch = np.concatenate(
            [cell.ncomp_per_branch for cell in cells]
        )
        self.ncomp = int(np.max(self.ncomp_per_branch))
        self.cumsum_ncomp = cumsum_leading_zero(self.ncomp_per_branch)
        self._internal_node_inds = np.arange(self.cumsum_ncomp[-1])
        self._append_params_and_states(self.network_params, self.network_states)

        self.nbranches_per_cell = [cell.total_nbranches for cell in cells]
        self.total_nbranches = sum(self.nbranches_per_cell)
        self._cumsum_nbranches = cumsum_leading_zero(self.nbranches_per_cell)

        self.nodes = pd.concat([c.nodes for c in cells], ignore_index=True)
        self.nodes["global_comp_index"] = np.arange(self.cumsum_ncomp[-1])
        self.nodes["global_branch_index"] = np.repeat(
            np.arange(self.total_nbranches), self.ncomp_per_branch
        ).tolist()
        self.nodes["global_cell_index"] = list(
            itertools.chain(
                *[[i] * int(cell.cumsum_ncomp[-1]) for i, cell in enumerate(cells)]
            )
        )
        offset = 0
        node_indices = []
        self._internal_node_inds = []
        for cell in cells:
            node_indices.append(cell.nodes.index + offset)
            self._internal_node_inds.append(cell._internal_node_inds + offset)
            offset += cell._n_nodes
        self.nodes.index = np.concatenate(node_indices)
        self._internal_node_inds = np.concatenate(self._internal_node_inds)

        parents = [cell.comb_parents for cell in cells]
        self.comb_parents = jnp.concatenate(
            [p.at[1:].add(self._cumsum_nbranches[i]) for i, p in enumerate(parents)]
        )

        # Two columns: `parent_branch_index` and `child_branch_index`. One row per
        # branch, apart from those branches which do not have a parent (i.e.
        # -1 in parents). For every branch, tracks the global index of that branch
        # (`child_branch_index`) and the global index of its parent
        # (`parent_branch_index`).
        self.branch_edges = pd.DataFrame(
            dict(
                parent_branch_index=self.comb_parents[self.comb_parents != -1],
                child_branch_index=np.where(self.comb_parents != -1)[0],
            )
        )

        # For morphology indexing of both `jax.sparse` and the custom `jaxley` solvers.
        self._par_inds, self._child_inds, self._child_belongs_to_branchpoint = (
            compute_children_and_parents(self.branch_edges)
        )

        # `nbranchpoints` in each cell == cell._par_inds (because `par_inds` are
        # unique).
        nbranchpoints = jnp.asarray([len(cell._par_inds) for cell in cells])
        self._cumsum_nbranchpoints_per_cell = cumsum_leading_zero(nbranchpoints)

        # Channels.
        self._gather_channels_from_constituents(cells)

        self.initialize()
        del self._cells_list

    def __repr__(self):
        return (
            f"{type(self).__name__} with {len(self.channels)} different channels "
            f"and {len(self.synapses)} synapses. Use `.nodes` or `.edges` for details."
        )

    def _init_comp_graph(self):
        """Initialize attributes concerning the compartment graph.

        In particular, it initializes:
        - `_comp_edges`
        - `_branchpoints`
        - `_n_nodes`
        - `_off_diagonal_inds`

        In principle, we could do this by traversing the entire `graph`. However,
        it is faster to just collect the graph attributes of each `cell` and
        concatenate them (which is what we do here).
        """
        # Compartment edges, branchpoints, internal_node_inds.
        offset = 0
        self._comp_edges = []
        self._branchpoints = []
        for cell in self._cells_list:
            # Compartment edges
            edges = cell._comp_edges.copy()
            edges[["source", "sink"]] += offset
            self._comp_edges.append(edges)

            # Branchpoints.
            branchpoints = cell._branchpoints.copy()
            branchpoints.index = branchpoints.index + offset
            self._branchpoints.append(branchpoints)

            offset += cell._n_nodes  # Compartment-offset.
        self._comp_edges = pd.concat(self._comp_edges, ignore_index=True)
        self._branchpoints = pd.concat(self._branchpoints)
        self._n_nodes = offset

        # off_diagonal_inds
        sources = np.asarray(self._comp_edges["source"].to_list())
        sinks = np.asarray(self._comp_edges["sink"].to_list())
        self._off_diagonal_inds = jnp.stack([sources, sinks]).astype(int)

    def _init_solver_jaxley_dhs_solve(self, *args, **kwargs) -> None:
        """Create module attributes for indexing with the `jaxley.dhs` voltage volver.

        This function first generates the networkX `comp_graph`, then traverses it
        to identify the solve order, and then pre-computes the relevant attributes used
        for re-ordering compartments during the voltage solve with `jaxley.dhs`.
        """
        offset = 0
        lower_and_upper_offset = 0

        dhs_map_dict = {}
        dhs_inv_map_to_node_order = []
        dhs_map_to_node_order = []
        dhs_map_to_node_order_lower = []
        dhs_map_to_node_order_upper = []
        dhs_node_order = []
        dhs_parent_lookup = []

        for cell in self._cells_list:
            dhs_map_dict.update(
                {
                    k + offset: v + offset
                    for k, v in cell._dhs_solve_indexer["map_dict"].items()
                }
            )
            dhs_inv_map_to_node_order.append(
                cell._dhs_solve_indexer["inv_map_to_solve_order"] + offset
            )
            dhs_map_to_node_order.append(
                cell._dhs_solve_indexer["map_to_solve_order"] + offset
            )

            dhs_map_to_node_order_lower.append(
                cell._dhs_solve_indexer["map_to_solve_order_lower"]
                + lower_and_upper_offset
            )
            dhs_map_to_node_order_upper.append(
                cell._dhs_solve_indexer["map_to_solve_order_upper"]
                + lower_and_upper_offset
            )

            # For point neurons, node order is empty. In that case [:, :2] would
            # fail. Thus, we need the if-case `len(node_order) == 0`.
            node_order = cell._dhs_solve_indexer["node_order"]
            if self._solver_device == "cpu" or len(node_order) == 0:
                # On CPU, we want to solve all cells sequentially for minimal compile
                # time. Because of this, we also add `offset` to the `level`.
                dhs_node_order.append(node_order + offset)
            else:
                # Only :2 because node_order contains [node, parent, level], and we
                # do not want to offset the level when solving cells in parallel.
                dhs_node_order.append(node_order.at[:, :2].add(offset))

            # Discard the last one because it is a [-1] which just absorbs all
            # compartments that are already finished with their recursion. We append
            # such a compartment after this loop again.
            parent_lookup = cell._dhs_solve_indexer["parent_lookup"].copy()[:-1]
            parent_lookup[1:] += offset
            dhs_parent_lookup.append(parent_lookup)

            offset += cell._n_nodes  # Compartment-offset.
            lower_and_upper_offset += (cell._n_nodes - 1) * 2

        self._dhs_solve_indexer = {}
        self._dhs_solve_indexer["inv_map_to_solve_order"] = np.concatenate(
            dhs_inv_map_to_node_order
        )
        self._dhs_solve_indexer["map_to_solve_order"] = np.concatenate(
            dhs_map_to_node_order
        )
        self._dhs_solve_indexer["map_to_solve_order_lower"] = np.concatenate(
            dhs_map_to_node_order_lower
        )
        self._dhs_solve_indexer["map_to_solve_order_upper"] = np.concatenate(
            dhs_map_to_node_order_upper
        )
        self._dhs_solve_indexer["node_order"] = np.concatenate(dhs_node_order, axis=0)
        self._dhs_solve_indexer["parent_lookup"] = np.concatenate(
            dhs_parent_lookup + [[-1]]
        )
        self._dhs_solve_indexer["node_order_grouped"] = dhs_group_comps_into_levels(
            self._dhs_solve_indexer["node_order"]
        )
        self._init_view()

    def _step_synapse(
        self,
        states: Dict,
        syn_channels: List,
        params: Dict,
        delta_t: float,
        edges: pd.DataFrame,
    ) -> Tuple[Dict, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Perform one step of the synapses and obtain their currents."""
        states = self._step_synapse_state(states, syn_channels, params, delta_t, edges)
        states, current_terms = self._synapse_currents(
            states, syn_channels, params, delta_t, edges
        )
        return states, current_terms

    def _step_synapse_state(
        self,
        states: Dict,
        syn_channels: List,
        params: Dict,
        delta_t: float,
        edges: pd.DataFrame,
    ) -> Dict:
        voltages = states["v"]

        grouped_syns = edges.groupby("type", sort=False, group_keys=False)
        pre_syn_inds = grouped_syns["pre_index"].apply(list)
        post_syn_inds = grouped_syns["post_index"].apply(list)
        synapse_names = list(grouped_syns.indices.keys())

        for i, synapse_type in enumerate(syn_channels):
            assert (
                synapse_names[i] == synapse_type._name
            ), "Mixup in the ordering of synapses. Please create an issue on Github."
            synapse_param_names = list(synapse_type.synapse_params.keys())
            synapse_state_names = list(synapse_type.synapse_states.keys())

            synapse_params = {}
            for p in synapse_param_names:
                synapse_params[p] = params[p]
            synapse_states = {}
            for s in synapse_state_names:
                synapse_states[s] = states[s]

            pre_inds = np.asarray(pre_syn_inds[synapse_names[i]])
            post_inds = np.asarray(post_syn_inds[synapse_names[i]])

            # State updates.
            states_updated = synapse_type.update_states(
                synapse_states,
                delta_t,
                voltages[pre_inds],
                voltages[post_inds],
                synapse_params,
            )

            # Rebuild state.
            for key, val in states_updated.items():
                states[key] = val

        return states

    def _synapse_currents(
        self,
        states: Dict,
        syn_channels: List,
        params: Dict,
        delta_t: float,
        edges: pd.DataFrame,
    ) -> Tuple[Dict, Tuple[jnp.ndarray, jnp.ndarray]]:
        voltages = states["v"]

        grouped_syns = edges.groupby("type", sort=False, group_keys=False)
        pre_syn_inds = grouped_syns["pre_index"].apply(list)
        post_syn_inds = grouped_syns["post_index"].apply(list)
        synapse_names = list(grouped_syns.indices.keys())

        syn_voltage_terms = jnp.zeros_like(voltages)
        syn_constant_terms = jnp.zeros_like(voltages)
        # Run with two different voltages that are `diff` apart to infer the slope and
        # offset.
        diff = 1e-3
        for i, synapse_type in enumerate(syn_channels):
            assert (
                synapse_names[i] == synapse_type._name
            ), "Mixup in the ordering of synapses. Please create an issue on Github."
            synapse_param_names = list(synapse_type.synapse_params.keys())
            synapse_state_names = list(synapse_type.synapse_states.keys())

            synapse_params = {}
            for p in synapse_param_names:
                synapse_params[p] = params[p]
            synapse_states = {}
            for s in synapse_state_names:
                synapse_states[s] = states[s]

            # Get pre and post indexes of the current synapse type.
            pre_inds = np.asarray(pre_syn_inds[synapse_names[i]])
            post_inds = np.asarray(post_syn_inds[synapse_names[i]])

            # Compute slope and offset of the current through every synapse.
            pre_v_and_perturbed = jnp.stack(
                [voltages[pre_inds], voltages[pre_inds] + diff]
            )
            post_v_and_perturbed = jnp.stack(
                [voltages[post_inds], voltages[post_inds] + diff]
            )
            synapse_currents = vmap(
                synapse_type.compute_current, in_axes=(None, 0, 0, None)
            )(
                synapse_states,
                pre_v_and_perturbed,
                post_v_and_perturbed,
                synapse_params,
            )
            synapse_currents_dist = convert_point_process_to_distributed(
                synapse_currents, params["area"][post_inds]
            )

            # Split into voltage and constant terms.
            voltage_term = (synapse_currents_dist[1] - synapse_currents_dist[0]) / diff
            constant_term = (
                synapse_currents_dist[0] - voltage_term * voltages[post_inds]
            )

            # Gather slope and offset for every postsynaptic compartment.
            gathered_syn_currents = gather_synapes(
                len(voltages),
                post_inds,
                voltage_term,
                constant_term,
            )
            syn_voltage_terms += gathered_syn_currents[0]
            syn_constant_terms -= gathered_syn_currents[1]

            # Add the synaptic currents through every compartment as state.
            # `post_syn_currents` is a `jnp.ndarray` of as many elements as there are
            # compartments in the network.
            # `[0]` because we only use the non-perturbed voltage.
            states[f"i_{synapse_type._name}"] = synapse_currents[0]

        return states, (syn_voltage_terms, syn_constant_terms)

    def arrange_in_layers(
        self,
        layers: List[int],
        within_layer_offset: float = 500.0,
        between_layer_offset: float = 1500.0,
        vertical_layers: bool = False,
    ):
        """Arrange the cells in the network to form layers.

        Moves the cells in the network to arrange them into layers.

        Args:
            layers: List of integers specifying the number of cells in each layer.
            within_layer_offset: Offset between cells within the same layer.
            between_layer_offset: Offset between layers.
            vertical_layers: If True, layers are arranged vertically.
        """
        assert (
            np.sum(layers) == self.shape[0]
        ), "The number of cells in the layers must match the number of cells in the network."
        cells_in_layers = [
            list(range(sum(layers[:i]), sum(layers[: i + 1])))
            for i in range(len(layers))
        ]

        for l, cell_inds in enumerate(cells_in_layers):
            layer = self.cell(cell_inds)
            for i, cell in enumerate(layer.cells):
                if vertical_layers:
                    x_offset = (i - (len(cell_inds) - 1) / 2) * within_layer_offset
                    y_offset = (len(layers) - 1 - l) * between_layer_offset
                else:
                    x_offset = l * between_layer_offset
                    y_offset = (i - (len(cell_inds) - 1) / 2) * within_layer_offset

                cell.move_to(x=x_offset, y=y_offset, z=0)

    def vis(
        self,
        detail: str = "full",
        ax: Optional[Axes] = None,
        color: str = "k",
        synapse_color: str = "b",
        dims: Tuple[int] = (0, 1),
        cell_plot_kwargs: Dict = {},
        synapse_plot_kwargs: Dict = {},
        synapse_scatter_kwargs: Dict = {},
        **kwargs,  # absorb add. kwargs, i.e. to enable net.cell(0).vis(type="line")
    ) -> Axes:
        """Visualize the module.

        Args:
            detail: Either of [point, full]. `point` visualizes every neuron in the
                network as a dot.
                `full` plots the full morphology of every neuron. It requires that
                `compute_xyz()` has been run.
            color: The color in which cells are plotted.
            synapse_color: The color in which synapses are plotted.
            dims: Which dimensions to plot. 1=x, 2=y, 3=z coordinate. Must be a tuple of
                two of them.
            cell_plot_kwargs: Keyword arguments passed to the plotting function for
                cell morphologies. Only takes effect for `detail='full'`.
            synapse_plot_kwargs: Keyword arguments passed to the plotting function for
                syanpses.
            synapse_scatter_kwargs: Keyword arguments passed to the scatter function for
                syanpse terminals.
        """
        xyz0 = self.cell(0).xyzr[0][:, :3]
        same_xyz = []
        for cell in self.cells:
            cell_coords = cell.xyzr[0][:, :3]
            same_xyz.append(
                xyz0.shape == cell_coords.shape and np.all(xyz0 == cell_coords)
            )
        if np.all(same_xyz):
            warn(
                "Same coordinates for all cells. Consider using `net.cell(i).move()`, "
                "`net.cell(i).move_to()` or `net.arrange_in_layers()` to move them."
            )

        if ax is None:
            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_subplot(111) if len(dims) < 3 else plt.axes(projection="3d")

        # detail="point" -> pos taken to be the mean of all traced points on the cell.
        cell_to_point_xyz = lambda cell: np.mean(np.vstack(cell.xyzr)[:, :3], axis=0)

        dims_np = np.asarray(dims)
        if detail == "point":
            for cell in self.cells:
                pos = cell_to_point_xyz(cell)[dims_np]
                ax.scatter(*pos, color=color, **cell_plot_kwargs)
        elif detail == "full":
            ax = super().vis(dims=dims, color=color, ax=ax, **cell_plot_kwargs)
        else:
            raise ValueError("detail must be in {full, point}.")

        # Plot the synapses.
        nodes = self.nodes
        for i, edge in self.edges.iterrows():
            prepost_locs = []
            for prepost in ["pre", "post"]:
                loc, comp = edge[[prepost + "_locs", prepost + "_index"]]
                branch = nodes.loc[comp, "global_branch_index"]
                cell = nodes.loc[comp, "global_cell_index"]
                branch_xyz = self.xyzr[branch][:, :3]

                xyz_loc = branch_xyz
                if detail == "point":
                    xyz_loc = cell_to_point_xyz(self.cell(cell))
                elif len(branch_xyz) == 2:
                    # If only start and end point of a branch are traced, perform a
                    # linear interpolation to get the synpase location.
                    xyz_loc = branch_xyz[0] + (branch_xyz[1] - branch_xyz[0]) * loc
                else:
                    # If densely traced, use intermediate trace values for synapse loc.
                    middle_ind = int((len(branch_xyz) - 1) * loc)
                    xyz_loc = xyz_loc[middle_ind]

                prepost_locs.append(xyz_loc)
            prepost_locs = np.stack(prepost_locs).T
            ax.plot(*prepost_locs[dims_np], color=synapse_color, **synapse_plot_kwargs)
            ax.scatter(
                *prepost_locs[dims_np, 1], color=synapse_color, **synapse_scatter_kwargs
            )

        return ax

    def _infer_synapse_type_ind(self, synapse_name):
        syn_names = self.base.synapse_names
        is_new_type = False if synapse_name in syn_names else True
        type_ind = len(syn_names) if is_new_type else syn_names.index(synapse_name)
        return type_ind, is_new_type

    def _update_synapse_state_names(self, synapse_type):
        # (Potentially) update variables that track meta information about synapses.
        self.base.synapse_names.append(synapse_type._name)
        self.base.synapse_param_names += list(synapse_type.synapse_params.keys())
        self.base.synapse_state_names += list(synapse_type.synapse_states.keys())
        self.base.synapses.append(synapse_type)

    def _append_multiple_synapses(self, pre_nodes, post_nodes, synapse_type):
        # Add synapse types to the module and infer their unique identifier.
        synapse_name = synapse_type._name
        synapse_current_name = f"i_{synapse_name}"
        type_ind, is_new = self._infer_synapse_type_ind(synapse_name)
        if is_new:  # synapse is not known
            self._update_synapse_state_names(synapse_type)
            self.base.synapse_current_names.append(synapse_current_name)
            index_within_type = list(range(0, len(pre_nodes)))
        else:
            # Figure out how many synapses of this type already exist
            n_existing = len(self.base.edges[self.base.edges.type == synapse_name])
            index_within_type = list(range(n_existing, n_existing + len(pre_nodes)))

        index = len(self.base.edges)
        indices = [idx for idx in range(index, index + len(pre_nodes))]
        global_edge_index = pd.DataFrame({"global_edge_index": indices})
        index_within_type = pd.DataFrame({"index_within_type": index_within_type})
        post_loc = loc_of_index(
            post_nodes["global_comp_index"].to_numpy(),
            post_nodes["global_branch_index"].to_numpy(),
            self.ncomp_per_branch,
        )
        pre_loc = loc_of_index(
            pre_nodes["global_comp_index"].to_numpy(),
            pre_nodes["global_branch_index"].to_numpy(),
            self.ncomp_per_branch,
        )

        # Define new synapses. Each row is one synapse.
        pre_nodes = pd.DataFrame({"pre_index": pre_nodes.index})
        post_nodes = pd.DataFrame({"post_index": post_nodes.index})
        new_rows = pd.concat(
            [
                global_edge_index,
                index_within_type,
                pre_nodes.reset_index(drop=True),
                post_nodes.reset_index(drop=True),
            ],
            axis=1,
        )
        new_rows["type"] = synapse_name
        new_rows["type_ind"] = type_ind
        new_rows["pre_locs"] = pre_loc
        new_rows["post_locs"] = post_loc
        self.base.edges = concat_and_ignore_empty(
            [self.base.edges, new_rows], ignore_index=True, axis=0
        )
        self._add_params_to_edges(synapse_type, indices)
        self.base.edges["controlled_by_param"] = 0
        self._edges_in_view = self.edges.index.to_numpy()

    def _add_params_to_edges(self, synapse_type, indices):
        # Add parameters and states to the `.edges` table.
        for key, param_val in synapse_type.synapse_params.items():
            self.base.edges.loc[indices, key] = param_val

        # Update synaptic state array.
        for key, state_val in synapse_type.synapse_states.items():
            self.base.edges.loc[indices, key] = state_val
