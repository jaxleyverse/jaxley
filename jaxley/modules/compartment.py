from typing import Callable, Dict, List, Optional

import jax.numpy as jnp
import pandas as pd

from jaxley.channels import Channel
from jaxley.modules.base import Module, View
from jaxley.utils.cell_utils import index_of_loc, loc_of_index


class Compartment(Module):
    compartment_params: Dict = {
        "length": 10.0,
        "radius": 1.0,
        "axial_resistivity": 5_000.0,
    }
    compartment_states: Dict = {"voltages": -70.0}

    def __init__(self):
        super().__init__()
        self._init_params_and_state(self.compartment_params, self.compartment_states)

        self.nseg = 1
        self.total_nbranches = 1
        self.nbranches_per_cell = [1]
        self.cumsum_nbranches = jnp.asarray([0, 1])

        # Indexing.
        self.nodes = pd.DataFrame(
            dict(comp_index=[0], branch_index=[0], cell_index=[0])
        )
        self.branch_edges = pd.DataFrame(
            dict(parent_branch_index=[], child_branch_index=[])
        )
        self.initialize()
        self.initialized_conds = True

    def init_conds(self, params):
        cond_params = {
            "branch_conds_fwd": jnp.asarray([]),
            "branch_conds_bwd": jnp.asarray([]),
            "coupling_conds_fwd": jnp.asarray([[]]),
            "coupling_conds_bwd": jnp.asarray([[]]),
            "summed_coupling_conds": jnp.asarray([[0.0]]),
        }
        return cond_params


class CompartmentView(View):
    """CompartmentView."""

    def __init__(self, pointer, view):
        view = view.assign(controlled_by_param=view.comp_index)
        super().__init__(pointer, view)

    def __call__(self, loc: float):
        if loc != "all":
            assert (
                loc >= 0.0 and loc <= 1.0
            ), "Compartments must be indexed by a continuous value between 0 and 1."

        index = index_of_loc(0, loc, self.pointer.nseg) if loc != "all" else "all"
        return super().adjust_view("comp_index", index)

    def connect(self, post, synapse_type):
        synapse_name = type(synapse_type).__name__
        if synapse_name not in self.pointer.synapse_names:
            new_synapse_type = True
        else:
            new_synapse_type = False

        if new_synapse_type:
            max_ind = self.pointer.syn_edges["type_ind"].max() + 1
            type_ind = 0 if jnp.isnan(max_ind) else max_ind
        else:
            type_ind = self.pointer.syn_edges.query(f"type == '{synapse_name}'")[
                "type_ind"
            ].to_numpy()[0]

        pre_comp = loc_of_index(
            self.view["global_comp_index"].to_numpy(), self.pointer.nseg
        )
        post_comp = loc_of_index(
            post.view["global_comp_index"].to_numpy(), self.pointer.nseg
        )
        self.pointer.syn_edges = pd.concat(
            [
                self.pointer.syn_edges,
                pd.DataFrame(
                    dict(
                        pre_locs=pre_comp,
                        post_locs=post_comp,
                        pre_branch_index=self.view["branch_index"].to_numpy(),
                        post_branch_index=post.view["branch_index"].to_numpy(),
                        pre_cell_index=self.view["cell_index"].to_numpy(),
                        post_cell_index=post.view["cell_index"].to_numpy(),
                        type=synapse_name,
                        type_ind=type_ind,
                        global_pre_comp_index=self.view["global_comp_index"].to_numpy(),
                        global_post_comp_index=post.view[
                            "global_comp_index"
                        ].to_numpy(),
                        global_pre_branch_index=self.view[
                            "global_branch_index"
                        ].to_numpy(),
                        global_post_branch_index=post.view[
                            "global_branch_index"
                        ].to_numpy(),
                    )
                ),
            ],
            ignore_index=True,
        )
        self.pointer.syn_edges["index"] = list(self.pointer.syn_edges.index)

        for key in synapse_type.synapse_params:
            param_vals = jnp.asarray([synapse_type.synapse_params[key]])
            if new_synapse_type:
                self.pointer.syn_params[key] = param_vals
            else:
                self.pointer.syn_params[key] = jnp.concatenate(
                    [self.pointer.syn_params[key], param_vals]
                )

        for key in synapse_type.synapse_states:
            state_vals = jnp.asarray([synapse_type.synapse_states[key]])
            if new_synapse_type:
                self.pointer.syn_states[key] = state_vals
            else:
                self.pointer.syn_states[key] = jnp.concatenate(
                    [self.pointer.syn_states[key], state_vals]
                )

        if new_synapse_type:
            self.pointer.synapse_names.append(type(synapse_type).__name__)
            self.pointer.synapse_param_names.append(synapse_type.synapse_params.keys())
            self.pointer.synapse_state_names.append(synapse_type.synapse_states.keys())
            self.pointer.syn_classes.append(synapse_type)
