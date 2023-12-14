from typing import Callable, Dict, List, Optional

import jax.numpy as jnp
import pandas as pd

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

        self.nseg = 1
        self.total_nbranches = 1
        self.nbranches_per_cell = [1]
        self.cumsum_nbranches = jnp.asarray([0, 1])

        # Indexing.
        self.nodes = pd.DataFrame(
            dict(comp_index=[0], branch_index=[0], cell_index=[0])
        )
        for param_name, param_value in self.compartment_params.items():
            self.nodes[param_name] = param_value
        for state_name, state_value in self.compartment_states.items():
            self.nodes[state_name] = state_value
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

    def connect(self, post: "CompartmentView", synapse_type):
        """Connect two compartments with a chemical synapse.

        High-level strategy:

        We need to first check if the network already has a type of this synapse, else
        we need to register it as a new synapse in a bunch of dictionaries which track
        synapse parameters, state and meta information.

        Next, we register the new connection in the synapse dataframe (`.syn_edges`).
        Then, we update synapse parameter and state arrays with the new connection.
        Finally, we update synapse meta information.
        """
        synapse_name = type(synapse_type).__name__
        is_new_type = True if synapse_name not in self.pointer.synapse_names else False

        if is_new_type:
            # New type: index for the synapse type is one more than the currently
            # highest index.
            max_ind = self.pointer.syn_edges["type_ind"].max() + 1
            type_ind = 0 if jnp.isnan(max_ind) else max_ind
        else:
            # Not a new type: search for the index that this type has previously had.
            type_ind = self.pointer.syn_edges.query(f"type == '{synapse_name}'")[
                "type_ind"
            ].to_numpy()[0]

        # The `syn_edges` dataframe expects the compartment as continuous `loc`, not
        # as discrete compartment index (because the continuous `loc` is used for
        # plotting). Below, we cast the compartment index to its (rough) location.
        pre_comp = loc_of_index(
            self.view["global_comp_index"].to_numpy(), self.pointer.nseg
        )
        post_comp = loc_of_index(
            post.view["global_comp_index"].to_numpy(), self.pointer.nseg
        )

        # Update edges.
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

        # We add a column called index which is used by `adjust_view` of the
        # `SynapseView` (see `network.py`).
        self.pointer.syn_edges["index"] = list(self.pointer.syn_edges.index)

        # Update synaptic parameter array.
        for key in synapse_type.synapse_params:
            param_vals = jnp.asarray([synapse_type.synapse_params[key]])
            if is_new_type:
                # Register parameter array for new synapse type.
                self.pointer.syn_params[key] = param_vals
            else:
                # Append to synaptic parameter array.
                self.pointer.syn_params[key] = jnp.concatenate(
                    [self.pointer.syn_params[key], param_vals]
                )

        # Update synaptic state array.
        for key in synapse_type.synapse_states:
            state_vals = jnp.asarray([synapse_type.synapse_states[key]])
            if is_new_type:
                # Register parameter array for new synapse type.
                self.pointer.syn_states[key] = state_vals
            else:
                # Append to synaptic parameter array.
                self.pointer.syn_states[key] = jnp.concatenate(
                    [self.pointer.syn_states[key], state_vals]
                )

        # (Potentially) update variables that track meta information about synapses.
        if is_new_type:
            self.pointer.synapse_names.append(type(synapse_type).__name__)
            self.pointer.synapse_param_names.append(
                list(synapse_type.synapse_params.keys())
            )
            self.pointer.synapse_state_names.append(
                list(synapse_type.synapse_states.keys())
            )
            self.pointer.syn_classes.append(synapse_type)
