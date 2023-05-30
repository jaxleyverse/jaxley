from typing import Dict, List, Optional, Callable

import jax.numpy as jnp
import pandas as pd
import numpy as np

from neurax.modules.base import Module, View
from neurax.modules.compartment import Compartment, CompartmentView


class Branch(Module):
    branch_params: Dict = {}
    branch_states: Dict = {}

    def __init__(self, compartments: List[Compartment]):
        super().__init__()
        self._init_params_and_state(self.branch_params, self.branch_states)
        self._append_to_params_and_state(compartments)

        self.compartments = compartments
        self.nseg = len(compartments)
        self.total_nbranches = 1
        self.nbranches_per_cell = [1]
        self.cumsum_nbranches = jnp.asarray([0, 1])
        self.channels = compartments[0].channels

        self.initialized_morph = True

        # Indexing.
        self.nodes = pd.DataFrame(
            dict(
                comp_index=np.arange(self.nseg).tolist(),
                branch_index=[0] * self.nseg,
                cell_index=[0] * self.nseg,
            )
        )
        # Synapse indexing.
        self.edges = pd.DataFrame(dict(pre_comp_index=[], post_comp_index=[], type=""))

    def __getattr__(self, key):
        assert key == "comp"
        return CompartmentView(self, self.nodes)

    def init_branch_conds(self):
        """Given an axial resisitivity, set the coupling conductances."""
        axial_resistivity = self.params["axial_resistivity"]
        radiuses = self.params["radius"]
        lengths = self.params["length"]

        def compute_coupling_cond(rad1, rad2, r_a, l1, l2):
            return rad1 * rad2**2 / r_a / (rad2**2 * l1 + rad1**2 * l2) / l1

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
        self.coupling_conds_bwd = compute_coupling_cond(rad2, rad1, r_a1, l2, l1)
        self.coupling_conds_fwd = compute_coupling_cond(rad1, rad2, r_a2, l1, l2)

        # Convert (S / cm / um) -> (mS / cm^2)
        self.coupling_conds_fwd *= 10**7
        self.coupling_conds_bwd *= 10**7

        # Compute the summed coupling conductances of each compartment.
        self.summed_coupling_conds = jnp.zeros((self.nseg))
        self.summed_coupling_conds = self.summed_coupling_conds.at[1:].add(
            self.coupling_conds_fwd
        )
        self.summed_coupling_conds = self.summed_coupling_conds.at[:-1].add(
            self.coupling_conds_bwd
        )
        self.initialized_conds = True


class BranchView(View):
    def __init__(self, pointer, view):
        super().__init__(pointer, view)

    def __call__(self, index):
        return super().adjust_view("branch_index", index)

    def __getattr__(self, key):
        assert key == "comp"
        return CompartmentView(self.pointer, self.view)
