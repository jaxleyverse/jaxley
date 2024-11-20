# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import itertools
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd
from jax import vmap
from matplotlib.axes import Axes

from jaxley.modules.base import Module
from jaxley.modules.cell import Cell
from jaxley.utils.cell_utils import (
    build_branchpoint_group_inds,
    compute_children_and_parents,
    convert_point_process_to_distributed,
    loc_of_index,
    merge_cells,
)
from jaxley.utils.misc_utils import concat_and_ignore_empty, cumsum_leading_zero
from jaxley.utils.solver_utils import (
    JaxleySolveIndexer,
    comp_edges_to_indices,
    remap_index_to_masked,
)
from jaxley.utils.syn_utils import gather_synapes


class Network(Module):
    """Network class.

    This class defines a network of cells that can be connected with synapses.
    """

    network_params: Dict = {}
    network_states: Dict = {}

    def __init__(
        self,
        cells: List[Cell],
    ):
        """Initialize network of cells and synapses.

        Args:
            cells: A list of cells that make up the network.
        """
        super().__init__()
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
        self._update_local_indices()
        self._init_view()

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

        # `nbranchpoints` in each cell == cell._par_inds (because `par_inds` are unique).
        nbranchpoints = jnp.asarray([len(cell._par_inds) for cell in cells])
        self._cumsum_nbranchpoints_per_cell = cumsum_leading_zero(nbranchpoints)

        # Channels.
        self._gather_channels_from_constituents(cells)

        self._initialize()
        del self._cells_list

    def __repr__(self):
        return f"{type(self).__name__} with {len(self.channels)} different channels and {len(self.synapses)} synapses. Use `.nodes` or `.edges` for details."

    def _init_morph_jaxley_spsolve(self):
        branchpoint_group_inds = build_branchpoint_group_inds(
            len(self._par_inds),
            self._child_belongs_to_branchpoint,
            self.cumsum_ncomp[-1],
        )
        children_in_level = merge_cells(
            self._cumsum_nbranches,
            self._cumsum_nbranchpoints_per_cell,
            [cell._solve_indexer.children_in_level for cell in self._cells_list],
            exclude_first=False,
        )
        parents_in_level = merge_cells(
            self._cumsum_nbranches,
            self._cumsum_nbranchpoints_per_cell,
            [cell._solve_indexer.parents_in_level for cell in self._cells_list],
            exclude_first=False,
        )
        padded_cumsum_ncomp = cumsum_leading_zero(
            np.concatenate(
                [np.diff(cell._solve_indexer.cumsum_ncomp) for cell in self._cells_list]
            )
        )

        # Generate mapping to dealing with the masking which allows using the custom
        # sparse solver to deal with different ncomp per branch.
        remapped_node_indices = remap_index_to_masked(
            self._internal_node_inds,
            self.nodes,
            padded_cumsum_ncomp,
            self.ncomp_per_branch,
        )
        self._solve_indexer = JaxleySolveIndexer(
            cumsum_ncomp=padded_cumsum_ncomp,
            branchpoint_group_inds=branchpoint_group_inds,
            children_in_level=children_in_level,
            parents_in_level=parents_in_level,
            root_inds=self._cumsum_nbranches[:-1],
            remapped_node_indices=remapped_node_indices,
        )

    def _init_morph_jax_spsolve(self):
        """Initialize the morphology for networks.

        The reason that this function is a bit involved for a `Network` is that Jaxley
        considers branchpoint nodes to be at the very end of __all__ nodes (i.e. the
        branchpoints of the first cell are even after the compartments of the second
        cell. The reason for this is that, otherwise, `cumsum_ncomp` becomes tricky).

        To achieve this, we first loop over all compartments and append them, and then
        loop over all branchpoints and append those. The code for building the indices
        from the `comp_edges` is identical to `jx.Cell`.

        Explanation of `self._comp_eges['type']`:
        `type == 0`: compartment <--> compartment (within branch)
        `type == 1`: branchpoint --> parent-compartment
        `type == 2`: branchpoint --> child-compartment
        `type == 3`: parent-compartment --> branchpoint
        `type == 4`: child-compartment --> branchpoint
        """
        self._cumsum_ncomp_per_cell = cumsum_leading_zero(
            jnp.asarray([cell.cumsum_ncomp[-1] for cell in self.cells])
        )
        self._comp_edges = pd.DataFrame()

        # Add all the internal nodes.
        for offset, cell in zip(self._cumsum_ncomp_per_cell, self._cells_list):
            condition = cell._comp_edges["type"].to_numpy() == 0
            rows = cell._comp_edges[condition]
            self._comp_edges = pd.concat(
                [self._comp_edges, [offset, offset, 0] + rows], ignore_index=True
            )

        # All branchpoint-to-compartment nodes.
        start_branchpoints = self.cumsum_ncomp[-1]  # Index of the first branchpoint.
        for offset, offset_branchpoints, cell in zip(
            self._cumsum_ncomp_per_cell,
            self._cumsum_nbranchpoints_per_cell,
            self._cells_list,
        ):
            offset_within_cell = cell.cumsum_ncomp[-1]
            condition = cell._comp_edges["type"].isin([1, 2])
            rows = cell._comp_edges[condition]
            self._comp_edges = pd.concat(
                [
                    self._comp_edges,
                    [
                        start_branchpoints - offset_within_cell + offset_branchpoints,
                        offset,
                        0,
                    ]
                    + rows,
                ],
                ignore_index=True,
            )

        # All compartment-to-branchpoint nodes.
        for offset, offset_branchpoints, cell in zip(
            self._cumsum_ncomp_per_cell,
            self._cumsum_nbranchpoints_per_cell,
            self._cells_list,
        ):
            offset_within_cell = cell.cumsum_ncomp[-1]
            condition = cell._comp_edges["type"].isin([3, 4])
            rows = cell._comp_edges[condition]
            self._comp_edges = pd.concat(
                [
                    self._comp_edges,
                    [
                        offset,
                        start_branchpoints - offset_within_cell + offset_branchpoints,
                        0,
                    ]
                    + rows,
                ],
                ignore_index=True,
            )

        # Convert comp_edges to the index format required for `jax.sparse` solvers.
        n_nodes, data_inds, indices, indptr = comp_edges_to_indices(self._comp_edges)
        self._n_nodes = n_nodes
        self._data_inds = data_inds
        self._indices_jax_spsolve = indices
        self._indptr_jax_spsolve = indptr

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
        pre_syn_inds = grouped_syns["pre_global_comp_index"].apply(list)
        post_syn_inds = grouped_syns["post_global_comp_index"].apply(list)
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
        pre_syn_inds = grouped_syns["pre_global_comp_index"].apply(list)
        post_syn_inds = grouped_syns["post_global_comp_index"].apply(list)
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
                synapse_currents,
                params["radius"][post_inds],
                params["length"][post_inds],
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
            states[f"{synapse_type._name}_current"] = synapse_currents[0]

        return states, (syn_voltage_terms, syn_constant_terms)

    def vis(
        self,
        detail: str = "full",
        ax: Optional[Axes] = None,
        col: str = "k",
        synapse_col: str = "b",
        dims: Tuple[int] = (0, 1),
        type: str = "line",
        layers: Optional[List] = None,
        morph_plot_kwargs: Dict = {},
        synapse_plot_kwargs: Dict = {},
        synapse_scatter_kwargs: Dict = {},
        networkx_options: Dict = {},
        layer_kwargs: Dict = {},
    ) -> Axes:
        """Visualize the module.

        Args:
            detail: Either of [point, full]. `point` visualizes every neuron in the
                network as a dot (and it uses `networkx` to obtain cell positions).
                `full` plots the full morphology of every neuron. It requires that
                `compute_xyz()` has been run and allows for indivual neurons to be
                moved with `.move()`.
            col: The color in which cells are plotted. Only takes effect if
                `detail='full'`.
            type: Either `line` or `scatter`. Only takes effect if `detail='full'`.
            synapse_col: The color in which synapses are plotted. Only takes effect if
                `detail='full'`.
            dims: Which dimensions to plot. 1=x, 2=y, 3=z coordinate. Must be a tuple of
                two of them.
            layers: Allows to plot the network in layers. Should provide the number of
                neurons in each layer, e.g., [5, 10, 1] would be a network with 5 input
                neurons, 10 hidden layer neurons, and 1 output neuron.
            morph_plot_kwargs: Keyword arguments passed to the plotting function for
                cell morphologies. Only takes effect for `detail='full'`.
            synapse_plot_kwargs: Keyword arguments passed to the plotting function for
                syanpses. Only takes effect for `detail='full'`.
            synapse_scatter_kwargs: Keyword arguments passed to the scatter function
                for the end point of synapses. Only takes effect for `detail='full'`.
            networkx_options: Options passed to `networkx.draw()`. Only takes effect if
                `detail='point'`.
            layer_kwargs: Only used if `layers` is specified and if `detail='full'`.
                Can have the following entries: `within_layer_offset` (float),
                `between_layer_offset` (float), `vertical_layers` (bool).
        """
        if detail == "point":
            graph = self._build_graph(layers)

            if layers is not None:
                pos = nx.multipartite_layout(graph, subset_key="layer")
                nx.draw(graph, pos, with_labels=True, **networkx_options)
            else:
                nx.draw(graph, with_labels=True, **networkx_options)
        elif detail == "full":
            if layers is not None:
                # Assemble cells in the network into layers.
                global_counter = 0
                layers_config = {
                    "within_layer_offset": 500.0,
                    "between_layer_offset": 1500.0,
                    "vertical_layers": False,
                }
                layers_config.update(layer_kwargs)
                for layer_ind, num_in_layer in enumerate(layers):
                    for ind_within_layer in range(num_in_layer):
                        if layers_config["vertical_layers"]:
                            x_offset = (
                                ind_within_layer - (num_in_layer - 1) / 2
                            ) * layers_config["within_layer_offset"]
                            y_offset = (len(layers) - 1 - layer_ind) * layers_config[
                                "between_layer_offset"
                            ]
                        else:
                            x_offset = layer_ind * layers_config["between_layer_offset"]
                            y_offset = (
                                ind_within_layer - (num_in_layer - 1) / 2
                            ) * layers_config["within_layer_offset"]

                        self.cell(global_counter).move_to(x=x_offset, y=y_offset, z=0)
                        global_counter += 1
            ax = super().vis(
                dims=dims,
                col=col,
                ax=ax,
                type=type,
                morph_plot_kwargs=morph_plot_kwargs,
            )

            pre_locs = self.edges["pre_locs"].to_numpy()
            post_locs = self.edges["post_locs"].to_numpy()
            pre_comp = self.edges["pre_global_comp_index"].to_numpy()
            nodes = self.nodes.set_index("global_comp_index")
            pre_branch = nodes.loc[pre_comp, "global_branch_index"].to_numpy()
            post_comp = self.edges["post_global_comp_index"].to_numpy()
            post_branch = nodes.loc[post_comp, "global_branch_index"].to_numpy()

            dims_np = np.asarray(dims)

            for pre_loc, post_loc, pre_b, post_b in zip(
                pre_locs, post_locs, pre_branch, post_branch
            ):
                pre_coord = self.xyzr[pre_b]
                if len(pre_coord) == 2:
                    # If only start and end point of a branch are traced, perform a
                    # linear interpolation to get the synpase location.
                    pre_coord = pre_coord[0] + (pre_coord[1] - pre_coord[0]) * pre_loc
                else:
                    # If densely traced, use intermediate trace values for synapse loc.
                    middle_ind = int((len(pre_coord) - 1) * pre_loc)
                    pre_coord = pre_coord[middle_ind]

                post_coord = self.xyzr[post_b]
                if len(post_coord) == 2:
                    # If only start and end point of a branch are traced, perform a
                    # linear interpolation to get the synpase location.
                    post_coord = (
                        post_coord[0] + (post_coord[1] - post_coord[0]) * post_loc
                    )
                else:
                    # If densely traced, use intermediate trace values for synapse loc.
                    middle_ind = int((len(post_coord) - 1) * post_loc)
                    post_coord = post_coord[middle_ind]

                coords = np.stack([pre_coord[dims_np], post_coord[dims_np]]).T
                ax.plot(
                    coords[0],
                    coords[1],
                    c=synapse_col,
                    **synapse_plot_kwargs,
                )
                ax.scatter(
                    post_coord[dims_np[0]],
                    post_coord[dims_np[1]],
                    c=synapse_col,
                    **synapse_scatter_kwargs,
                )
        else:
            raise ValueError("detail must be in {full, point}.")

        return ax

    def _build_graph(self, layers: Optional[List] = None, **options):
        graph = nx.DiGraph()

        def build_extents(*subset_sizes):
            return nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))

        if layers is not None:
            extents = build_extents(*layers)
            layers = [range(start, end) for start, end in extents]
            for i, layer in enumerate(layers):
                graph.add_nodes_from(layer, layer=i)
        else:
            graph.add_nodes_from(range(len(self._cells_in_view)))

        pre_comp = self.edges["pre_global_comp_index"].to_numpy()
        nodes = self.nodes.set_index("global_comp_index")
        pre_cell = nodes.loc[pre_comp, "global_cell_index"].to_numpy()
        post_comp = self.edges["post_global_comp_index"].to_numpy()
        post_cell = nodes.loc[post_comp, "global_cell_index"].to_numpy()

        inds = np.stack([pre_cell, post_cell]).T
        graph.add_edges_from(inds)

        return graph

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
        synapse_current_name = f"{synapse_name}_current"
        type_ind, is_new = self._infer_synapse_type_ind(synapse_name)
        if is_new:  # synapse is not known
            self._update_synapse_state_names(synapse_type)
            self.base.synapse_current_names.append(synapse_current_name)

        index = len(self.base.edges)
        indices = [idx for idx in range(index, index + len(pre_nodes))]
        global_edge_index = pd.DataFrame({"global_edge_index": indices})
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
        pre_nodes = pre_nodes[["global_comp_index"]]
        pre_nodes.columns = ["pre_global_comp_index"]
        post_nodes = post_nodes[["global_comp_index"]]
        post_nodes.columns = ["post_global_comp_index"]
        new_rows = pd.concat(
            [
                global_edge_index,
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
