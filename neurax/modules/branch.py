from typing import Callable, Dict, List, Optional
from copy import deepcopy

import jax.numpy as jnp
import numpy as np
import pandas as pd

from neurax.modules.base import Module, View, GroupView
from neurax.modules.compartment import Compartment, CompartmentView
from neurax.utils.cell_utils import compute_coupling_cond


class Branch(Module):
    branch_params: Dict = {}
    branch_states: Dict = {}

    def __init__(self, compartments: List[Compartment]):
        super().__init__()
        self._init_params_and_state(self.branch_params, self.branch_states)
        self._append_to_params_and_state(compartments)
        for comp in compartments:
            self._append_to_channel_params_and_state(comp)

        self.nseg = len(compartments)
        self.total_nbranches = 1
        self.nbranches_per_cell = [1]
        self.cumsum_nbranches = jnp.asarray([0, 1])

        self.initialized_morph = True
        self.initialized_conds = False
        self.initialized_syns = True

        # Indexing.
        self.nodes = pd.DataFrame(
            dict(
                comp_index=np.arange(self.nseg).tolist(),
                branch_index=[0] * self.nseg,
                cell_index=[0] * self.nseg,
            )
        )

        # Channel indexing.
        for i, comp in enumerate(compartments):
            index = pd.DataFrame.from_dict(
                dict(comp_index=[i], branch_index=[0], cell_index=[0])
            )
            for channel in comp.channels:
                self._append_to_channel_nodes(index, channel)

        # Synapse indexing.
        self.syn_edges = pd.DataFrame(
            dict(pre_comp_index=[], post_comp_index=[], type="")
        )
        self.branch_edges = pd.DataFrame(
            dict(parent_branch_index=[], child_branch_index=[])
        )

    def __getattr__(self, key):
        # Ensure that hidden methods such as `__deepcopy__` still work.
        if key.startswith("__"):
            return super().__getattribute__(key)

        if key == "comp":
            view = deepcopy(self.nodes)
            view["original_comp_index"] = view["comp_index"]
            view["original_branch_index"] = view["branch_index"]
            view["original_cell_index"] = view["cell_index"]
            return CompartmentView(self, view)
        elif key in self.group_views:
            return self.group_views[key]
        else:
            raise KeyError(f"Key {key} not recognized.")

    def init_conds(self, params):
        conds = self.init_branch_conds(
            params["axial_resistivity"], params["radius"], params["length"], self.nseg
        )
        cond_params = {
            "branch_conds_fwd": jnp.asarray([]),
            "branch_conds_bwd": jnp.asarray([]),
        }
        cond_params["coupling_conds_fwd"] = conds[0]
        cond_params["coupling_conds_bwd"] = conds[1]
        cond_params["summed_coupling_conds"] = conds[2]

        return cond_params

    @staticmethod
    def init_branch_conds(axial_resistivity, radiuses, lengths, nseg):
        """Given an axial resisitivity, set the coupling conductances."""

        # Compute coupling conductance for segments within a branch.
        # `radius`: um
        # `r_a`: ohm cm
        # `length_single_compartment`: um
        # `coupling_conds`: S * um / cm / um^2 = S / cm / um
        rad1 = radiuses[1:]
        rad2 = radiuses[:-1]
        l1 = lengths[1:]
        l2 = lengths[:-1]
        r_a1 = axial_resistivity[1:]
        r_a2 = axial_resistivity[:-1]
        coupling_conds_bwd = compute_coupling_cond(rad2, rad1, r_a1, l2, l1)
        coupling_conds_fwd = compute_coupling_cond(rad1, rad2, r_a2, l1, l2)

        # Convert (S / cm / um) -> (mS / cm^2)
        coupling_conds_fwd *= 10**7
        coupling_conds_bwd *= 10**7

        # Compute the summed coupling conductances of each compartment.
        summed_coupling_conds = jnp.zeros((nseg))
        summed_coupling_conds = summed_coupling_conds.at[1:].add(coupling_conds_fwd)
        summed_coupling_conds = summed_coupling_conds.at[:-1].add(coupling_conds_bwd)
        return coupling_conds_fwd, coupling_conds_bwd, summed_coupling_conds


class BranchView(View):
    def __init__(self, pointer, view):
        view = view.assign(controlled_by_param=view.branch_index)
        super().__init__(pointer, view)

    def __call__(self, index: float):
        self.allow_make_trainable = True
        return super().adjust_view("branch_index", index)

    def __getattr__(self, key):
        assert key == "comp"
        return CompartmentView(self.pointer, self.view)
