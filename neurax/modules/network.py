import itertools
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import vmap

from neurax.connection import Connectivity
from neurax.modules.base import Module, View
from neurax.modules.branch import Branch
from neurax.modules.cell import Cell, CellView
from neurax.utils.cell_utils import merge_cells
from neurax.utils.syn_utils import postsyn_voltage_updates, prepare_syn


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
        self._init_params_and_state(self.network_params, self.network_states)
        self._append_to_params_and_state(cells)
        for cell in cells:
            self._append_to_channel_params_and_state(cell)
        self._append_synapses_to_params_and_state(connectivities)

        self.cells = cells
        self.connectivities = connectivities
        self.conns = [connectivity.synapse_type for connectivity in connectivities]
        self.nseg = cells[0].nseg
        self.synapse_names = [type(c.synapse_type).__name__ for c in connectivities]
        self.synapse_param_names = [
            c.synapse_type.synapse_params.keys() for c in connectivities
        ]
        self.synapse_state_names = [
            c.synapse_type.synapse_states.keys() for c in connectivities
        ]

        self.initialize()
        self.initialized_conds = False

    def _append_synapses_to_params_and_state(self, connectivities):
        for connectivity in connectivities:
            for key in connectivity.synapse_type.synapse_params:
                param_vals = jnp.asarray(
                    [
                        connectivity.synapse_type.synapse_params[key]
                        for _ in connectivity.conns
                    ]
                )
                self.syn_params[key] = param_vals
            for key in connectivity.synapse_type.synapse_states:
                state_vals = jnp.asarray(
                    [
                        connectivity.synapse_type.synapse_states[key]
                        for _ in connectivity.conns
                    ]
                )
                self.syn_states[key] = state_vals

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
            return SynapseView(self, self.syn_edges, key)
        elif key in self.group_views:
            return self.group_views[key]
        else:
            raise KeyError(f"Key {key} not recognized.")

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

        # Indexing.
        self.nodes = pd.DataFrame(
            dict(
                comp_index=np.arange(self.nseg * self.total_nbranches).tolist(),
                branch_index=(
                    np.arange(self.nseg * self.total_nbranches) // self.nseg
                ).tolist(),
                cell_index=list(
                    itertools.chain(
                        *[
                            [i] * (self.nseg * b)
                            for i, b in enumerate(self.nbranches_per_cell)
                        ]
                    )
                ),
            )
        )

        # Channel indexing.
        for i, cell in enumerate(self.cells):
            for channel in cell.channels:
                name = type(channel).__name__
                comp_inds = deepcopy(cell.channel_nodes[name]["comp_index"].to_numpy())
                branch_inds = deepcopy(
                    cell.channel_nodes[name]["branch_index"].to_numpy()
                )
                comp_inds += self.nseg * self.cumsum_nbranches[i]
                branch_inds += self.cumsum_nbranches[i]
                index = pd.DataFrame.from_dict(
                    dict(
                        comp_index=comp_inds,
                        branch_index=branch_inds,
                        cell_index=[i] * len(comp_inds),
                    )
                )
                self._append_to_channel_nodes(index, channel)

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
            axial_resistivity[par_inds, 0],
            axial_resistivity[child_inds, -1],
            radiuses[par_inds, 0],
            radiuses[child_inds, -1],
            lengths[par_inds, 0],
            lengths[child_inds, -1],
        )
        summed_coupling_conds = Cell.update_summed_coupling_conds(
            summed_coupling_conds,
            child_inds,
            conds[0],
            conds[1],
            parents,
        )

        branch_conds_fwd = jnp.zeros((nbranches))
        branch_conds_bwd = jnp.zeros((nbranches))
        branch_conds_fwd = branch_conds_fwd.at[child_inds].set(conds[0])
        branch_conds_bwd = branch_conds_bwd.at[child_inds].set(conds[1])

        cond_params = {
            "coupling_conds_fwd": coupling_conds_fwd,
            "coupling_conds_bwd": coupling_conds_bwd,
            "summed_coupling_conds": summed_coupling_conds,
            "branch_conds_fwd": branch_conds_fwd,
            "branch_conds_bwd": branch_conds_bwd,
        }

        return cond_params

    def init_syns(self):
        pre_comp_inds = []
        post_comp_inds = []
        for connectivity in self.connectivities:
            pre_cell_inds, pre_inds, post_cell_inds, post_inds = prepare_syn(
                connectivity.conns, self.nseg
            )
            pre_comp_inds.append(
                self.cumsum_nbranches[pre_cell_inds] * self.nseg + pre_inds
            )
            post_comp_inds.append(
                self.cumsum_nbranches[post_cell_inds] * self.nseg + post_inds
            )

        # Prepare synapses.
        self.syn_edges = pd.DataFrame(
            columns=["pre_comp_index", "post_comp_index", "type", "type_ind"]
        )
        for i, connectivity in enumerate(self.connectivities):
            self.syn_edges = pd.concat(
                [
                    self.syn_edges,
                    pd.DataFrame(
                        dict(
                            pre_comp_index=pre_comp_inds[i],
                            post_comp_index=post_comp_inds[i],
                            type=type(connectivity.synapse_type).__name__,
                            type_ind=i,
                        )
                    ),
                ],
            )
        self.syn_edges["index"] = list(self.syn_edges.index)

        self.branch_edges = pd.DataFrame(
            dict(
                parent_branch_index=self.comb_parents[self.comb_parents != -1],
                child_branch_index=np.where(self.comb_parents != -1)[0],
            )
        )

        self.initialized_syns = True

    @staticmethod
    def _step_synapse(
        u,
        syn_channels,
        params,
        delta_t,
        edges: pd.DataFrame,
    ):
        """Perform one step of the synapses and obtain their currents."""
        voltages = u["voltages"]

        grouped_syns = edges.groupby("type", sort=False, group_keys=False)
        pre_syn_inds = grouped_syns["pre_comp_index"].apply(list)
        post_syn_inds = grouped_syns["post_comp_index"].apply(list)
        synapse_names = list(grouped_syns.indices.keys())

        syn_voltage_terms = jnp.zeros_like(voltages)
        syn_constant_terms = jnp.zeros_like(voltages)
        new_syn_states = []
        for i, synapse_type in enumerate(syn_channels):
            assert (
                synapse_names[i] == type(synapse_type).__name__
            ), "Mixup in the ordering of synapses. Please create an issue on Github."
            synapse_states, synapse_current_terms = synapse_type.step(
                u, delta_t, voltages, params, np.asarray(pre_syn_inds[i])
            )
            synapse_current_terms = postsyn_voltage_updates(
                voltages,
                np.asarray(post_syn_inds[i]),
                *synapse_current_terms,
            )
            syn_voltage_terms += synapse_current_terms[0]
            syn_constant_terms += synapse_current_terms[1]
            new_syn_states.append(synapse_states)

        return new_syn_states, syn_voltage_terms, syn_constant_terms


class SynapseView(View):
    """SynapseView."""

    def __init__(self, pointer, view, key):
        view = view[view["type"] == key]
        view = view.assign(controlled_by_param=view.index)
        super().__init__(pointer, view)

    def __call__(self, index: int):
        return self.adjust_view("index", index)

    def show(
        self,
        *,
        indices: bool = True,
        params: bool = True,
        states: bool = True,
    ):
        """Show synapses."""
        ind_of_params = self.view.index.values
        nodes = deepcopy(self.view)

        if not indices:
            for key in nodes:
                nodes = nodes.drop(key, axis=1)

        if params:
            for key, val in self.pointer.syn_params.items():
                nodes[key] = val[ind_of_params]

        if states:
            for key, val in self.pointer.syn_states.items():
                nodes[key] = val[ind_of_params]

        return nodes

    def adjust_view(self, key: str, index: float):
        """Update view."""
        if index != "all":
            self.view = self.view[self.view[key] == index]
        return self

    def set_params(self, key: str, val: float):
        """Set parameters of the pointer."""
        assert (
            key in self.pointer.synapse_param_names[self.view["type_ind"].values[0]]
        ), f"Parameter {key} does not exist in synapse of type {self.view['type'].values[0]}."
        self.pointer._set_params(key, val, self.view)

    def set_states(self, key: str, val: float):
        """Set parameters of the pointer."""
        assert (
            key in self.pointer.synapse_state_names[self.view["type_ind"].values[0]]
        ), f"State {key} does not exist in synapse of type {self.view['type'].values[0]}."
        self.pointer._set_states(key, val, self.view)

    def get_params(self, key: str):
        """Return parameters."""
        assert (
            key in self.pointer.synapse_param_names[self.view["type_ind"].values[0]]
        ), f"Parameter {key} does not exist in synapse of type {self.view['type'].values[0]}."
        self.pointer._get_params(key, self.view)

    def get_states(self, key: str):
        """Return states."""
        assert (
            key in self.pointer.synapse_state_names[self.view["type_ind"].values[0]]
        ), f"State {key} does not exist in synapse of type {self.view['type'].values[0]}."
        self.pointer._get_states(key, self.view)

    def make_trainable(self, key: str, init_val: Optional[Union[float, list]] = None):
        """Make a parameter trainable."""
        assert (
            key in self.pointer.synapse_param_names[self.view["type_ind"].values[0]]
        ), f"Parameter {key} does not exist in synapse of type {self.view['type'].values[0]}."
        self.pointer._make_trainable(self.view, key, init_val)
