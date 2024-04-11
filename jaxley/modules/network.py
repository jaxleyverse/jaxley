import itertools
from copy import deepcopy
from itertools import chain
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd
from jax import vmap
from matplotlib.axes import Axes

from jaxley.connection import Connectivity
from jaxley.modules.base import GroupView, Module, View
from jaxley.modules.branch import Branch
from jaxley.modules.cell import Cell, CellView
from jaxley.utils.cell_utils import (
    convert_point_process_to_distributed,
    flip_comp_indices,
    merge_cells,
    get_local_indices,
)
from jaxley.utils.syn_utils import gather_synapes, prepare_syn


class Network(Module):
    """Network."""

    network_params: Dict = {}
    network_states: Dict = {}

    def __init__(
        self,
        cells: List[Cell],
        connectivities: List[Connectivity] = [],
    ):
        """Initialize network of cells and synapses.

        Args:
            cells: _description_
            connectivities: _description_
        """
        super().__init__()
        for cell in cells:
            self.xyzr += deepcopy(cell.xyzr)

        self.cells = cells
        self.nseg = cells[0].nseg

        self.synapses = [connectivity.synapse_type for connectivity in connectivities]

        # TODO(@michaeldeistler): should we also track this for channels?
        self.synapse_names = [c.synapse_type._name for c in connectivities]
        self.synapse_param_names = list(
            chain.from_iterable(
                [list(c.synapse_type.synapse_params.keys()) for c in connectivities]
            )
        )
        self.synapse_state_names = list(
            chain.from_iterable(
                [list(c.synapse_type.synapse_states.keys()) for c in connectivities]
            )
        )

        # Two columns: `parent_branch_index` and `child_branch_index`. One row per
        # branch, apart from those branches which do not have a parent (i.e.
        # -1 in parents). For every branch, tracks the global index of that branch
        # (`child_branch_index`) and the global index of its parent
        # (`parent_branch_index`). Needed at `init_syns()`.
        self.branch_edges: Optional[pd.DataFrame] = None

        self.initialize()
        self.init_syns(connectivities)

        self.nodes = pd.concat([c.nodes for c in cells], ignore_index=True)
        self._append_params_and_states(self.network_params, self.network_states)
        self.nodes["comp_index"] = np.arange(self.nseg * self.total_nbranches).tolist()
        self.nodes["branch_index"] = (
            np.arange(self.nseg * self.total_nbranches) // self.nseg
        ).tolist()
        self.nodes["cell_index"] = list(
            itertools.chain(
                *[[i] * (self.nseg * b) for i, b in enumerate(self.nbranches_per_cell)]
            )
        )

        # Channels.
        self._gather_channels_from_constituents(cells)
        self.initialized_conds = False

    def __getattr__(self, key):
        # Ensure that hidden methods such as `__deepcopy__` still work.
        if key.startswith("__"):
            return super().__getattribute__(key)

        if key == "cell":
            view = deepcopy(self.nodes)
            view["global_comp_index"] = view["comp_index"]
            view["global_branch_index"] = view["branch_index"]
            view["global_cell_index"] = view["cell_index"]
            return CellView(self, view)
        elif key in self.synapse_names:
            type_index = self.synapse_names.index(key)
            return SynapseView(self, self.edges, key, self.synapses[type_index])
        elif key in self.group_nodes:
            inds = self.group_nodes[key].index.values
            view = self.nodes.loc[inds]
            view["global_comp_index"] = view["comp_index"]
            view["global_branch_index"] = view["branch_index"]
            view["global_cell_index"] = view["cell_index"]
            return GroupView(self, view)
        else:
            raise KeyError(f"Key {key} not recognized.")

    @property
    def shape(self):
        local_idcs = get_local_indices(self.nodes)
        return tuple(local_idcs.nunique())
    
    def init_morph(self):
        self.nbranches_per_cell = [cell.total_nbranches for cell in self.cells]
        self.total_nbranches = sum(self.nbranches_per_cell)
        self.cumsum_nbranches = jnp.cumsum(jnp.asarray([0] + self.nbranches_per_cell))

        parents = [cell.comb_parents for cell in self.cells]
        self.comb_parents = jnp.concatenate(
            [p.at[1:].add(self.cumsum_nbranches[i]) for i, p in enumerate(parents)]
        )
        self.comb_branches_in_each_level = merge_cells(
            self.cumsum_nbranches,
            [cell.comb_branches_in_each_level for cell in self.cells],
            exclude_first=False,
        )

        self.initialized_morph = True

    def init_conds(self, params):
        """Given an axial resisitivity, set the coupling conductances."""
        nbranches = self.total_nbranches
        nseg = self.nseg
        parents = self.comb_parents

        axial_resistivity = jnp.reshape(params["axial_resistivity"], (nbranches, nseg))
        radiuses = jnp.reshape(params["radius"], (nbranches, nseg))
        lengths = jnp.reshape(params["length"], (nbranches, nseg))

        conds = vmap(Branch.init_branch_conds, in_axes=(0, 0, 0, None))(
            axial_resistivity, radiuses, lengths, self.nseg
        )
        coupling_conds_fwd = conds[0]
        coupling_conds_bwd = conds[1]
        summed_coupling_conds = conds[2]

        par_inds = self.branch_edges["parent_branch_index"].to_numpy()
        child_inds = self.branch_edges["child_branch_index"].to_numpy()

        conds = vmap(Cell.init_cell_conds, in_axes=(0, 0, 0, 0, 0, 0))(
            axial_resistivity[child_inds, -1],
            axial_resistivity[par_inds, 0],
            radiuses[child_inds, -1],
            radiuses[par_inds, 0],
            lengths[child_inds, -1],
            lengths[par_inds, 0],
        )
        branch_conds_fwd = jnp.zeros((nbranches))
        branch_conds_bwd = jnp.zeros((nbranches))
        branch_conds_fwd = branch_conds_fwd.at[child_inds].set(conds[0])
        branch_conds_bwd = branch_conds_bwd.at[child_inds].set(conds[1])

        summed_coupling_conds = Cell.update_summed_coupling_conds(
            summed_coupling_conds,
            child_inds,
            branch_conds_fwd,
            branch_conds_bwd,
            parents,
        )

        cond_params = {
            "coupling_conds_fwd": coupling_conds_fwd,
            "coupling_conds_bwd": coupling_conds_bwd,
            "summed_coupling_conds": summed_coupling_conds,
            "branch_conds_fwd": branch_conds_fwd,
            "branch_conds_bwd": branch_conds_bwd,
        }

        return cond_params

    def init_syns(self, connectivities):
        global_pre_comp_inds = []
        global_post_comp_inds = []
        global_pre_branch_inds = []
        global_post_branch_inds = []
        pre_locs = []
        post_locs = []
        pre_branch_inds = []
        post_branch_inds = []
        pre_cell_inds = []
        post_cell_inds = []
        for i, connectivity in enumerate(connectivities):
            pre_cell_inds_, pre_inds, post_cell_inds_, post_inds = prepare_syn(
                connectivity.conns, self.nseg
            )
            # Global compartment indizes.
            global_pre_comp_inds = (
                self.cumsum_nbranches[pre_cell_inds_] * self.nseg + pre_inds
            )
            global_post_comp_inds = (
                self.cumsum_nbranches[post_cell_inds_] * self.nseg + post_inds
            )
            global_pre_branch_inds = [
                self.cumsum_nbranches[c.pre_cell_ind] + c.pre_branch_ind
                for c in connectivity.conns
            ]
            global_post_branch_inds = [
                self.cumsum_nbranches[c.post_cell_ind] + c.post_branch_ind
                for c in connectivity.conns
            ]
            # Local compartment inds.
            pre_locs = np.asarray([c.pre_loc for c in connectivity.conns])
            post_locs = np.asarray([c.post_loc for c in connectivity.conns])
            # Local branch inds.
            pre_branch_inds = np.asarray([c.pre_branch_ind for c in connectivity.conns])
            post_branch_inds = np.asarray(
                [c.post_branch_ind for c in connectivity.conns]
            )
            pre_cell_inds = pre_cell_inds_
            post_cell_inds = post_cell_inds_

            self.edges = pd.concat(
                [
                    self.edges,
                    pd.DataFrame(
                        dict(
                            pre_locs=pre_locs,
                            pre_branch_index=pre_branch_inds,
                            pre_cell_index=pre_cell_inds,
                            post_locs=post_locs,
                            post_branch_index=post_branch_inds,
                            post_cell_index=post_cell_inds,
                            type=connectivity.synapse_type._name,
                            type_ind=i,
                            global_pre_comp_index=global_pre_comp_inds,
                            global_post_comp_index=global_post_comp_inds,
                            global_pre_branch_index=global_pre_branch_inds,
                            global_post_branch_index=global_post_branch_inds,
                        )
                    ),
                ],
                ignore_index=True,
            )

        # Add parameters and states to the `.edges` table.
        index = 0
        for i, connectivity in enumerate(connectivities):
            for key in connectivity.synapse_type.synapse_params:
                param_val = connectivity.synapse_type.synapse_params[key]
                indices = np.arange(index, index + len(connectivity.conns))
                self.edges.loc[indices, key] = param_val

            for key in connectivity.synapse_type.synapse_states:
                state_val = connectivity.synapse_type.synapse_states[key]
                indices = np.arange(index, index + len(connectivity.conns))
                self.edges.loc[indices, key] = state_val

            index += len(connectivity.conns)

        self.branch_edges = pd.DataFrame(
            dict(
                parent_branch_index=self.comb_parents[self.comb_parents != -1],
                child_branch_index=np.where(self.comb_parents != -1)[0],
            )
        )

        self.initialized_syns = True

    def _step_synapse(
        self,
        states,
        syn_channels,
        params,
        delta_t,
        edges: pd.DataFrame,
    ):
        """Perform one step of the synapses and obtain their currents."""
        states = self._step_synapse_state(states, syn_channels, params, delta_t, edges)
        states, current_terms = self._synapse_currents(
            states, syn_channels, params, delta_t, edges
        )
        return states, current_terms

    def _step_synapse_state(
        self, states, syn_channels, params, delta_t, edges: pd.DataFrame
    ):
        voltages = states["v"]

        grouped_syns = edges.groupby("type", sort=False, group_keys=False)
        pre_syn_inds = grouped_syns["global_pre_comp_index"].apply(list)
        post_syn_inds = grouped_syns["global_post_comp_index"].apply(list)
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

            pre_inds = flip_comp_indices(pre_inds, self.nseg)  # See #305
            post_inds = flip_comp_indices(post_inds, self.nseg)  # See #305

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
        self, states, syn_channels, params, delta_t, edges: pd.DataFrame
    ):
        voltages = states["v"]

        grouped_syns = edges.groupby("type", sort=False, group_keys=False)
        pre_syn_inds = grouped_syns["global_pre_comp_index"].apply(list)
        post_syn_inds = grouped_syns["global_post_comp_index"].apply(list)
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

            pre_inds = flip_comp_indices(pre_inds, self.nseg)  # See #305
            post_inds = flip_comp_indices(post_inds, self.nseg)  # See #305

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
    ) -> None:
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

                        self.cell(global_counter).move(x=x_offset, y=y_offset, z=0)
                        global_counter += 1
            ax = self._vis(
                dims=dims,
                col=col,
                ax=ax,
                type=type,
                view=self.nodes,
                morph_plot_kwargs=morph_plot_kwargs,
            )

            pre_locs = self.edges["pre_locs"].to_numpy()
            post_locs = self.edges["post_locs"].to_numpy()
            pre_branch = self.edges["global_pre_branch_index"].to_numpy()
            post_branch = self.edges["global_post_branch_index"].to_numpy()

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
            graph.add_nodes_from(range(len(self.cells)))

        pre_cell = self.edges["pre_cell_index"].to_numpy()
        post_cell = self.edges["post_cell_index"].to_numpy()

        inds = np.stack([pre_cell, post_cell]).T
        graph.add_edges_from(inds)

        return graph


class SynapseView(View):
    """SynapseView."""

    def __init__(self, pointer, view, key, synapse: "jx.Synapse"):
        self.synapse = synapse
        view = deepcopy(view[view["type"] == key])
        view = view.assign(controlled_by_param=0)

        # Used for `.set()`.
        view["global_index"] = view.index.values
        # Used for `__call__()`.
        view["index"] = list(range(len(view)))
        # Because `make_trainable` needs to access the rows of `jaxedges` (which does
        # not contain `NaNa` rows) we need to reset the index here. We undo this for
        # `.set()`. `.index.values` is used for `make_trainable`.
        view = view.reset_index(drop=True)

        super().__init__(pointer, view)

    def __call__(self, index: int):
        self.view["controlled_by_param"] = self.view.index.values
        return self.adjust_view("index", index)

    def show(
        self,
        *,
        indices: bool = True,
        params: bool = True,
        states: bool = True,
    ):
        """Show synapses."""
        printable_nodes = deepcopy(self.view[["type", "type_ind"]])

        if indices:
            names = [
                "pre_locs",
                "pre_branch_index",
                "pre_cell_index",
                "post_locs",
                "post_branch_index",
                "post_cell_index",
            ]
            printable_nodes[names] = self.view[names]

        if params:
            for key in self.synapse.synapse_params.keys():
                printable_nodes[key] = self.view[key]

        if states:
            for key in self.synapse.synapse_states.keys():
                printable_nodes[key] = self.view[key]

        printable_nodes["controlled_by_param"] = self.view["controlled_by_param"]
        return printable_nodes

    def set(self, key: str, val: float):
        """Set parameters of the pointer."""
        synapse_index = self.view["type_ind"].values[0]
        synapse_type = self.pointer.synapses[synapse_index]
        synapse_param_names = list(synapse_type.synapse_params.keys())
        synapse_state_names = list(synapse_type.synapse_states.keys())

        assert (
            key in synapse_param_names or key in synapse_state_names
        ), f"{key} does not exist in synapse of type {synapse_type._name}."

        # Reset index to global index because we are writing to `self.edges`.
        self.view = self.view.set_index("global_index", drop=False)
        self.pointer._set(key, val, self.view, self.pointer.edges)

    def _assert_key_in_params_or_states(self, key):
        synapse_index = self.view["type_ind"].values[0]
        synapse_type = self.pointer.synapses[synapse_index]
        synapse_param_names = list(synapse_type.synapse_params.keys())
        synapse_state_names = list(synapse_type.synapse_states.keys())

        assert (
            key in synapse_param_names or key in synapse_state_names
        ), f"{key} does not exist in synapse of type {synapse_type._name}."

    def make_trainable(
        self,
        key: str,
        init_val: Optional[Union[float, list]] = None,
        verbose: bool = True,
    ):
        """Make a parameter trainable."""
        self._assert_key_in_params_or_states(key)
        # Use `.index.values` for indexing because we are memorizing the indices for
        # `jaxedges`.
        self.pointer._make_trainable(self.view, key, init_val, verbose=verbose)

    def data_set(
        self,
        key: str,
        val: Union[float, jnp.ndarray],
        param_state: Optional[List[Dict]] = None,
    ):
        """Set parameter of module (or its view) to a new value within `jit`."""
        self._assert_key_in_params_or_states(key)
        return self.pointer._data_set(key, val, self.view, param_state=param_state)

    def record(self, state: str = "v"):
        """Record a state."""
        assert (
            state in self.pointer.synapse_state_names[self.view["type_ind"].values[0]]
        ), f"State {state} does not exist in synapse of type {self.view['type'].values[0]}."

        view = deepcopy(self.view)
        view["state"] = state

        recording_view = view[["state"]]
        recording_view = recording_view.assign(rec_index=view.index)
        self.pointer._record(recording_view)
