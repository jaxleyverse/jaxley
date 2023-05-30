from typing import Dict, List, Optional, Callable
import itertools

import numpy as np
import jax.numpy as jnp
import pandas as pd

from neurax.modules.base import Module, View
from neurax.modules.cell import Cell, CellView
from neurax.connection import Connection
from neurax.network import Connectivity
from neurax.stimulus import get_external_input
from neurax.cell import merge_cells, _compute_index_of_kid, cum_indizes_of_kids
from neurax.utils.syn_utils import postsyn_voltage_updates
from neurax.synapses.synapse import Synapse
from neurax.utils.syn_utils import prepare_presyn, prepare_postsyn


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

        self.coupling_conds_fwd = jnp.asarray([])
        self.coupling_conds_bwd = jnp.asarray([])
        self.branch_conds_fwd = jnp.asarray([])
        self.branch_conds_bwd = jnp.asarray([])
        self.summed_coupling_conds = jnp.asarray([])

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
        self.nbranches = [cell.nbranches for cell in self.cells]
        self.cumsum_nbranches = jnp.cumsum(jnp.asarray([0] + self.nbranches))
        self.max_num_kids = 4
        # for c in self.cells:
        #     assert (
        #         self.max_num_kids == c.max_num_kids
        #     ), "Different max_num_kids between cells."

        parents = [cell.parents for cell in self.cells]
        self.comb_parents = jnp.concatenate(
            [p.at[1:].add(self.cumsum_nbranches[i]) for i, p in enumerate(parents)]
        )
        self.comb_parents_in_each_level = merge_cells(
            self.cumsum_nbranches,
            [cell.parents_in_each_level for cell in self.cells],
        )
        self.comb_branches_in_each_level = merge_cells(
            self.cumsum_nbranches,
            [cell.branches_in_each_level for cell in self.cells],
            exclude_first=False,
        )

        # Prepare indizes for solve
        comb_ind_of_kids = jnp.concatenate(
            [jnp.asarray(_compute_index_of_kid(cell.parents)) for cell in self.cells]
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
                comp_index=np.arange(self.nseg * sum(self.nbranches)).tolist(),
                branch_index=(
                    np.arange(self.nseg * sum(self.nbranches)) // self.nseg
                ).tolist(),
                cell_index=list(
                    itertools.chain(
                        *[[i] * (self.nseg * b) for i, b in enumerate(self.nbranches)]
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

    def step(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        i_inds: jnp.ndarray,
        i_current: jnp.ndarray,
    ):
        """Step for a single compartment.

        Args:
            u: The full state vector, including states of all channels and the voltage.
            dt: Time step.

        Returns:
            Next state. Same shape as `u`.
        """
        nbranches = sum(self.nbranches)
        nseg = self.nseg

        if self.branch_conds_fwd is None:
            self.init_branch_conds()

        voltages = u["voltages"]

        # Parameters have to go in here.
        new_channel_states, (v_terms, const_terms) = self.step_channels(
            u, dt, self.cells[0].branches[0].compartments[0].channels, self.params
        )

        # External input.
        i_ext = get_external_input(
            voltages, i_inds, i_current, self.params["radius"], self.params["length"]
        )

        # Step of the synapse.
        new_syn_states, syn_voltage_terms, syn_constant_terms = self.step_synapse(
            u,
            self.conns,
            self.params,
            dt,
            self.cumsum_nbranches,
            self.edges["pre_comp_index"].to_numpy(),
            self.post_grouped_inds,
            self.post_grouped_syns,
            self.nseg,
        )

        new_voltages = self.step_voltages(
            voltages=jnp.reshape(voltages, (nbranches, nseg)),
            voltage_terms=jnp.reshape(v_terms, (nbranches, nseg)),
            constant_terms=jnp.reshape(const_terms, (nbranches, nseg))
            + jnp.reshape(i_ext, (nbranches, nseg)),
            coupling_conds_bwd=jnp.reshape(
                self.coupling_conds_bwd, (nbranches, nseg - 1)
            ),
            coupling_conds_fwd=jnp.reshape(
                self.coupling_conds_fwd, (nbranches, nseg - 1)
            ),
            summed_coupling_conds=jnp.reshape(
                self.summed_coupling_conds, (nbranches, nseg)
            ),
            branch_cond_fwd=self.branch_conds_fwd,
            branch_cond_bwd=self.branch_conds_bwd,
            num_branches=sum(self.nbranches),
            parents=self.comb_parents,
            kid_inds_in_each_level=self.comb_cum_kid_inds_in_each_level,
            max_num_kids=4,
            parents_in_each_level=self.comb_parents_in_each_level,
            branches_in_each_level=self.comb_branches_in_each_level,
            tridiag_solver="thomas",
            delta_t=dt,
        )
        final_state = new_channel_states[0]
        for s in new_syn_states:
            for key in s:
                final_state[key] = s[key]
        final_state["voltages"] = new_voltages.flatten(order="C")
        return final_state

    def step_synapse(
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
