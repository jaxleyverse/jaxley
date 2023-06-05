import itertools
from typing import Callable, Dict, List, Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import vmap

from neurax.connection import Connection
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
        self, cells: List[Cell], connectivities: List[List[Connection]],
    ):
        """Initialize network of cells and synapses.

        Args:
            cells (List[Cell]): _description_
            conns (List[List[Connection]]): _description_
        """
        super().__init__()
        self._init_params_and_state(self.network_params, self.network_states)
        self._append_to_params_and_state(cells)
        self._append_synapses_to_params_and_state(connectivities)

        self.cells = cells
        self.connectivities = connectivities
        self.conns = [connectivity.synapse_type for connectivity in connectivities]
        self.nseg = cells[0].nseg
        self.channels = cells[0].channels

        self.initialized_morph = False
        self.initialized_syns = False
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
        if key == "cell":
            return CellView(self, self.nodes)
        elif key == "synapse":
            return SynapseView(self, self.syn_edges)
        else:
            raise KeyError("Only synapse and cell as keys allowed.")

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
            summed_coupling_conds, child_inds, conds[0], conds[1], parents,
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
        self.syn_edges = pd.DataFrame()
        for i, connectivity in enumerate(self.connectivities):
            self.syn_edges = pd.concat(
                [
                    self.syn_edges,
                    pd.DataFrame(
                        dict(
                            pre_comp_index=pre_comp_inds[i],
                            post_comp_index=post_comp_inds[i],
                            type=type(connectivity.synapse_type).__name__,
                        )
                    ),
                ]
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
        u, syn_channels, params, delta_t, edges: pd.DataFrame,
    ):
        """Perform one step of the synapses and obtain their currents."""
        voltages = u["voltages"]

        pre_syn_inds = edges.groupby("type")["pre_comp_index"].apply(list)
        post_syn_inds = edges.groupby("type")["post_comp_index"].apply(list)

        syn_voltage_terms = jnp.zeros_like(voltages)
        syn_constant_terms = jnp.zeros_like(voltages)
        new_syn_states = []
        for i, list_of_synapses in enumerate(syn_channels):
            synapse_states, synapse_current_terms = list_of_synapses.step(
                u, delta_t, voltages, params, np.asarray(pre_syn_inds[i])
            )
            synapse_current_terms = postsyn_voltage_updates(
                voltages, np.asarray(post_syn_inds[i]), *synapse_current_terms,
            )
            syn_voltage_terms += synapse_current_terms[0]
            syn_constant_terms += synapse_current_terms[1]
            new_syn_states.append(synapse_states)

        return new_syn_states, syn_voltage_terms, syn_constant_terms


class SynapseView(View):
    """SynapseView."""

    def __init__(self, pointer, view):
        view = view.assign(controlled_by_param=view.index)
        super().__init__(pointer, view)

    def __call__(self, index: int):
        # TODO. Problem is that the `type` value of `pointer.syn_edges` is a string.
        # Thus, in `adjust_view`, we cannot subtract entries from each other.
        assert (
            index == "all"
        ), "Not implemented to set parameters for individual synapses."
        return super().adjust_view("index", index)

