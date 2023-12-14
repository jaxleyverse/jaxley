from copy import deepcopy
from typing import Callable, Dict, List, Optional, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd

from jaxley.modules.base import GroupView, Module, View
from jaxley.modules.compartment import Compartment, CompartmentView
from jaxley.utils.cell_utils import compute_coupling_cond


class Branch(Module):
    branch_params: Dict = {}
    branch_states: Dict = {}

    def __init__(
        self,
        compartments: Union[Compartment, List[Compartment]],
        nseg: Optional[int] = None,
    ):
        super().__init__()
        assert (
            isinstance(compartments, Compartment) or nseg is None
        ), "If `compartments` is a list then you cannot set `nseg`."
        assert (
            isinstance(compartments, List) or nseg is not None
        ), "If `compartments` is not a list then you have to set `nseg`."
        if isinstance(compartments, Compartment):
            compartment_list = [compartments for _ in range(nseg)]
        else:
            compartment_list = compartments
        # Compartments are currently defined in reverse. See also #30. This `.reverse`
        # is needed to make `tests/test_composability_of_modules.py` pass.
        compartment_list.reverse()

        self.nseg = len(compartment_list)
        self.total_nbranches = 1
        self.nbranches_per_cell = [1]
        self.cumsum_nbranches = jnp.asarray([0, 1])

        # Indexing.
        # TODO: need to take care of setting the `HH` column to False, not NaN.
        self.nodes = pd.concat([c.nodes for c in compartment_list], ignore_index=True)
        self._append_params_and_states(self.branch_params, self.branch_states)
        self.nodes["comp_index"] = np.arange(self.nseg).tolist()
        self.nodes["branch_index"] = [0] * self.nseg
        self.nodes["cell_index"] = [0] * self.nseg

        # Channels.
        for comp in compartment_list:
            for channel in comp.channels:
                self.channels.append(channel)
        # Setting columns of channel names to `False` instead of `NaN`.
        for channel in self.channels:
            name = type(channel).__name__
            self.nodes[name] = self.nodes[name].notna()

        # Synapse indexing.
        self.syn_edges = pd.DataFrame(
            dict(global_pre_comp_index=[], global_post_comp_index=[], type="")
        )
        self.branch_edges = pd.DataFrame(
            dict(parent_branch_index=[], child_branch_index=[])
        )
        self.initialize()
        self.initialized_conds = False

    def __getattr__(self, key):
        # Ensure that hidden methods such as `__deepcopy__` still work.
        if key.startswith("__"):
            return super().__getattribute__(key)

        if key == "comp":
            view = deepcopy(self.nodes)
            view["global_comp_index"] = view["comp_index"]
            view["global_branch_index"] = view["branch_index"]
            view["global_cell_index"] = view["cell_index"]
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
        r1 = radiuses[:-1]
        r2 = radiuses[1:]
        r_a1 = axial_resistivity[:-1]
        r_a2 = axial_resistivity[1:]
        l1 = lengths[:-1]
        l2 = lengths[1:]
        coupling_conds_bwd = compute_coupling_cond(r1, r2, r_a1, r_a2, l1, l2)
        coupling_conds_fwd = compute_coupling_cond(r2, r1, r_a2, r_a1, l2, l1)

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
        new_view = super().adjust_view("branch_index", index)
        new_view.view["comp_index"] -= new_view.view["comp_index"].iloc[0]
        return new_view

    def __getattr__(self, key):
        assert key == "comp"
        return CompartmentView(self.pointer, self.view)
