from typing import Dict, List, Optional, Callable

import jax.numpy as jnp
import pandas as pd
import numpy as np

from neurax.modules.base import Module, View
from neurax.modules.compartment import Compartment, CompartmentView
from neurax.utils.cell_utils import compute_coupling_cond


class Branch(Module):
    branch_params: Dict = {}
    branch_states: Dict = {}

    def __init__(self, compartments: List[Compartment]):
        super().__init__()
        self._init_params_and_state(self.branch_params, self.branch_states)
        self._append_to_params_and_state(compartments)

        self.nseg = len(compartments)
        self.total_nbranches = 1
        self.nbranches_per_cell = [1]
        self.cumsum_nbranches = jnp.asarray([0, 1])
        self.channels = compartments[0].channels

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
        # Synapse indexing.
        self.syn_edges = pd.DataFrame(
            dict(pre_comp_index=[], post_comp_index=[], type="")
        )
        self.branch_edges = pd.DataFrame(
            dict(parent_branch_index=[], child_branch_index=[])
        )

    def __getattr__(self, key):
        assert key == "comp"
        return CompartmentView(self, self.nodes)

    def init_conds(self):
        conds = self.init_branch_conds(
            self.params["axial_resistivity"],
            self.params["radius"],
            self.params["length"],
            self.nseg,
        )
        self.coupling_conds_fwd = conds[0]
        self.coupling_conds_bwd = conds[1]
        self.summed_coupling_conds = conds[2]
        self.initialized_conds = True

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
        super().__init__(pointer, view)

    def __call__(self, loc: float):
        return super().adjust_view("branch_index", loc)

    def __getattr__(self, key):
        assert key == "comp"
        return CompartmentView(self.pointer, self.view)
