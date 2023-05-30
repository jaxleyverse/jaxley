from typing import Dict, List, Optional, Callable
import itertools

import numpy as np
import jax.numpy as jnp
import pandas as pd

from neurax.modules.base import Module, View
from neurax.modules.cell import Cell, CellView
from neurax.channels import Channel
from neurax.connection import Connection
from neurax.cell import merge_cells, _compute_index_of_kid, cum_indizes_of_kids
from neurax.utils.syn_utils import prepare_presyn, prepare_postsyn
from neurax.utils.syn_utils import postsyn_voltage_updates


class Network(Module):
    network_params: Dict = {}
    network_states: Dict = {}

    def __init__(
        self,
        cells: List[Cell],
        conns: List[List[Connection]],
    ):
        """Initialize network of cells and synapses.

        Args:
            cells (List[Cell]): _description_
            conns (List[List[Connection]]): _description_
        """
        super().__init__()
        self._init_params_and_state(self.network_params, self.network_states)
        self._append_to_params_and_state(cells)
        self._append_synapses_to_params_and_state(conns)

        self.cells = cells
        self.conns = conns
        self.nseg = cells[0].nseg
        self.channels = cells[0].channels

        self.initialized_morph = False
        self.initialized_syns = False
        self.initialized_conds = True

    def _append_synapses_to_params_and_state(self, conns):
        for conn in conns:
            for key in conn[0].synapse_params:
                param_vals = jnp.asarray([c.synapse_params[key] for c in conn])
                self.params[key] = param_vals
            for key in conn[0].synapse_states:
                state_vals = jnp.asarray([c.synapse_states[key] for c in conn])
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
        # Initially, the coupling conductances are set to `None`. They have to be set
        # by calling `.set_axial_resistivities()`.
        self.coupling_conds_fwd = jnp.concatenate(
            [c.coupling_conds_fwd for c in self.cells]
        )
        self.coupling_conds_bwd = jnp.concatenate(
            [c.coupling_conds_bwd for c in self.cells]
        )
        self.branch_conds_fwd = jnp.concatenate(
            [c.branch_conds_fwd for c in self.cells]
        )
        self.branch_conds_bwd = jnp.concatenate(
            [c.branch_conds_bwd for c in self.cells]
        )
        self.summed_coupling_conds = jnp.concatenate(
            [c.summed_coupling_conds for c in self.cells]
        )
        self.initialized_conds = True

    def init_syns(self):
        pre_comp_inds = []
        post_comp_inds = []
        self.post_grouped_inds = []
        self.post_grouped_syns = []
        for conn in self.conns:
            pre_syn_cell_inds, pre_syn_inds = prepare_presyn(conn, self.nseg)
            pre_comp_inds.append(
                self.cumsum_nbranches[pre_syn_cell_inds] * self.nseg + pre_syn_inds
            )
            grouped_post_inds, grouped_syns, post_syn = prepare_postsyn(conn, self.nseg)
            post_comp_inds.append(
                self.cumsum_nbranches[post_syn[:, 0]] * self.nseg + post_syn[:, 1]
            )
            self.post_grouped_inds.append(grouped_post_inds)
            self.post_grouped_syns.append(grouped_syns)

        # Prepare synapses.
        self.edges = pd.DataFrame()
        for i in range(len(self.conns)):
            self.edges = pd.concat(
                [
                    self.edges,
                    pd.DataFrame(
                        dict(
                            pre_comp_index=pre_comp_inds[i],
                            post_comp_index=post_comp_inds[i],
                            type="glutamate",
                        )
                    ),
                ]
            )

    def _step_synapse(
        self,
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
            synapse_states, synapse_current_terms = list_of_synapses[0].step(
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
