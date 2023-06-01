from typing import Dict, List, Optional, Callable
import itertools

import numpy as np
from jax import vmap
import jax.numpy as jnp
import pandas as pd

from neurax.modules.base import Module, View
from neurax.modules.cell import Cell, CellView
from neurax.modules.branch import Branch
from neurax.channels import Channel
from neurax.connection import Connection
from neurax.utils.cell_utils import (
    merge_cells,
    _compute_index_of_kid,
    cum_indizes_of_kids,
)
from neurax.utils.syn_utils import postsyn_voltage_updates
from neurax.utils.syn_utils import prepare_presyn, prepare_postsyn


class Network(Module):
    """Network."""

    network_params: Dict = {}
    network_states: Dict = {}

    def __init__(
        self,
        cells: List[Cell],
        connectivities: List[List[Connection]],
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
                self.params[key] = param_vals
            for key in connectivity.synapse_type.synapse_states:
                state_vals = jnp.asarray(
                    [
                        connectivity.synapse_type.synapse_states[key]
                        for _ in connectivity.conns
                    ]
                )
                self.states[key] = state_vals

    def __getattr__(self, key):
        assert key == "cell"
        return CellView(self, self.nodes)

    def init_morph(self):
        self.nbranches_per_cell = [cell.total_nbranches for cell in self.cells]
        self.total_nbranches = sum(self.nbranches_per_cell)
        self.cumsum_nbranches = jnp.cumsum(jnp.asarray([0] + self.nbranches_per_cell))
        self.max_num_kids = 4
        # for c in self.cells:
        #     assert (
        #         self.max_num_kids == c.max_num_kids
        #     ), "Different max_num_kids between cells."

        parents = [cell.comb_parents for cell in self.cells]
        self.comb_parents = jnp.concatenate(
            [p.at[1:].add(self.cumsum_nbranches[i]) for i, p in enumerate(parents)]
        )
        self.comb_parents_in_each_level = merge_cells(
            self.cumsum_nbranches,
            [cell.comb_parents_in_each_level for cell in self.cells],
        )
        self.comb_branches_in_each_level = merge_cells(
            self.cumsum_nbranches,
            [cell.comb_branches_in_each_level for cell in self.cells],
            exclude_first=False,
        )

        # Prepare indizes for solve
        comb_ind_of_kids = jnp.concatenate(
            [
                jnp.asarray(_compute_index_of_kid(cell.comb_parents))
                for cell in self.cells
            ]
        )
        # Defined only for forward Euler:
        self.comb_cum_kid_inds = cum_indizes_of_kids(
            [comb_ind_of_kids], self.max_num_kids, reset_at=[-1, 0]
        )[0]
        comb_ind_of_kids_in_each_level = [
            comb_ind_of_kids[bil] for bil in self.comb_branches_in_each_level
        ]
        self.comb_cum_kid_inds_in_each_level = cum_indizes_of_kids(
            comb_ind_of_kids_in_each_level, self.max_num_kids, reset_at=[0]
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

    def init_conds(self):
        """Given an axial resisitivity, set the coupling conductances."""
        # Initially, the coupling conductances are set to `None`. They have to be set
        # by calling `.set_axial_resistivities()`.
        nbranches = self.total_nbranches
        nseg = self.nseg
        parents = self.comb_parents

        axial_resistivity = jnp.reshape(
            self.params["axial_resistivity"], (nbranches, nseg)
        )
        radiuses = jnp.reshape(self.params["radius"], (nbranches, nseg))
        lengths = jnp.reshape(self.params["length"], (nbranches, nseg))

        conds = vmap(Branch.init_branch_conds, in_axes=(0, 0, 0, None))(
            axial_resistivity, radiuses, lengths, self.nseg
        )
        self.coupling_conds_fwd = conds[0]
        self.coupling_conds_bwd = conds[1]
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
        self.summed_coupling_conds = Cell.update_summed_coupling_conds(
            summed_coupling_conds,
            child_inds,
            conds[0],
            conds[1],
            parents,
        )

        self.branch_conds_fwd = jnp.zeros((nbranches * nseg))
        self.branch_conds_bwd = jnp.zeros((nbranches * nseg))
        self.branch_conds_fwd = self.branch_conds_fwd.at[child_inds].set(conds[0])
        self.branch_conds_bwd = self.branch_conds_bwd.at[child_inds].set(conds[1])

        self.initialized_conds = True

    def init_syns(self):
        pre_comp_inds = []
        post_comp_inds = []
        self.post_grouped_inds = []
        self.post_grouped_syns = []
        for connectivity in self.connectivities:
            pre_syn_cell_inds, pre_syn_inds = prepare_presyn(
                connectivity.conns, self.nseg
            )
            pre_comp_inds.append(
                self.cumsum_nbranches[pre_syn_cell_inds] * self.nseg + pre_syn_inds
            )
            grouped_post_inds, grouped_syns, post_syn = prepare_postsyn(
                connectivity.conns, self.nseg
            )
            post_comp_inds.append(
                self.cumsum_nbranches[post_syn[:, 0]] * self.nseg + post_syn[:, 1]
            )
            self.post_grouped_inds.append(grouped_post_inds)
            self.post_grouped_syns.append(grouped_syns)

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
        cumsum_num_branches,
        syn_inds,
        grouped_post_syn_inds,
        grouped_post_syns,
        nseg,
    ):
        """Perform one step of the synapses and obtain their currents."""
        voltages = u["voltages"]

        syn_voltage_terms = jnp.zeros_like(voltages)
        syn_constant_terms = jnp.zeros_like(voltages)
        new_syn_states = []
        for i, list_of_synapses in enumerate(syn_channels):
            synapse_states, synapse_current_terms = list_of_synapses.step(
                u, delta_t, voltages, params, syn_inds
            )
            synapse_current_terms = postsyn_voltage_updates(
                nseg,
                cumsum_num_branches,
                voltages,
                grouped_post_syn_inds[i],
                grouped_post_syns[i],
                *synapse_current_terms,
            )
            syn_voltage_terms += synapse_current_terms[0]
            syn_constant_terms += synapse_current_terms[1]
            new_syn_states.append(synapse_states)

        return new_syn_states, syn_voltage_terms, syn_constant_terms
