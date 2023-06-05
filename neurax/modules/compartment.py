from typing import Callable, Dict, List, Optional

import jax.numpy as jnp
import pandas as pd

from neurax.channels import Channel
from neurax.modules.base import Module, View
from neurax.utils.cell_utils import index_of_loc


class Compartment(Module):
    compartment_params: Dict = {
        "length": 10.0,
        "radius": 1.0,
        "axial_resistivity": 5_000.0,
    }
    compartment_states: Dict = {"voltages": -70.0}

    def __init__(self, channels: List[Channel]):
        super().__init__()
        self._init_params_and_state(self.compartment_params, self.compartment_states)

        # Insert channel parameters.
        for channel in channels:
            for key in channel.channel_params:
                self.params[key] = jnp.asarray(
                    [channel.channel_params[key]]
                )  # should be atleast1d TODO

        # Insert channel states.
        for channel in channels:
            for key in channel.channel_states:
                self.states[key] = jnp.asarray(
                    [channel.channel_states[key]]
                )  # should be atleast1d TODO

        self.channels = channels
        self.initialized_morph = True
        self.initialized_conds = True
        self.initialized_syns = True

        self.nseg = 1
        self.total_nbranches = 1
        self.nbranches_per_cell = [1]
        self.cumsum_nbranches = jnp.asarray([0, 1])

        # Indexing.
        self.nodes = pd.DataFrame(
            dict(comp_index=[0], branch_index=[0], cell_index=[0])
        )
        # Synapse indexing.
        self.syn_edges = pd.DataFrame(
            dict(pre_comp_index=[], post_comp_index=[], type="")
        )
        self.branch_edges = pd.DataFrame(
            dict(parent_branch_index=[], child_branch_index=[])
        )

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
