from typing import Dict, List, Optional, Callable
import numpy as np
import jax.numpy as jnp
import pandas as pd

from neurax.modules.base import Module, View
from neurax.modules.branch import Branch, BranchView
from neurax.cell import (
    _compute_num_kids,
    compute_levels,
    compute_branches_in_level,
    _compute_index_of_kid,
    cum_indizes_of_kids,
)


class Cell(Module):
    """Cell."""

    cell_params: Dict = {}
    cell_states: Dict = {}

    def __init__(self, branches: List[Branch], parents: List):
        super().__init__()
        self._init_params_and_state(self.cell_params, self.cell_states)
        self._append_to_params_and_state(branches)

        self.branches = branches
        self.nseg = branches[0].nseg
        self.total_nbranches = len(branches)
        self.nbranches_per_cell = [len(branches)]
        self.comb_parents = jnp.asarray(parents)
        self.cumsum_nbranches = jnp.asarray([0, len(branches)])
        self.channels = branches[0].channels
        self.max_num_kids = 4

        # Indexing.
        self.nodes = pd.DataFrame(
            dict(
                comp_index=np.arange(self.nseg * self.total_nbranches).tolist(),
                branch_index=(
                    np.arange(self.nseg * self.total_nbranches) // self.nseg
                ).tolist(),
                cell_index=[0] * (self.nseg * self.total_nbranches),
            )
        )
        # Synapse indexing.
        self.edges = pd.DataFrame(dict(pre_comp_index=[], post_comp_index=[], type=""))
        
        self.coupling_conds_bwd = jnp.asarray([b.coupling_conds_bwd for b in branches])
        self.coupling_conds_fwd = jnp.asarray([b.coupling_conds_fwd for b in branches])
        self.summed_coupling_conds = jnp.asarray(
            [b.summed_coupling_conds for b in branches]
        )
        self.initialized_morph = False
        self.initialized_conds = False

    def __getattr__(self, key):
        assert key == "branch"
        return BranchView(self, self.nodes)

    def init_morph(self):
        parents = self.comb_parents

        self.num_kids = jnp.asarray(_compute_num_kids(parents))
        self.levels = compute_levels(parents)
        self.comb_branches_in_each_level = compute_branches_in_level(self.levels)

        self.comb_parents_in_each_level = [
            jnp.unique(parents[c]) for c in self.comb_branches_in_each_level
        ]

        ind_of_kids = jnp.asarray(_compute_index_of_kid(parents))
        ind_of_kids_in_each_level = [
            ind_of_kids[bil] for bil in self.comb_branches_in_each_level
        ]
        self.comb_cum_kid_inds_in_each_level = cum_indizes_of_kids(
            ind_of_kids_in_each_level, max_num_kids=4, reset_at=[0]
        )
        self.initialized_morph = True

    def init_branch_conds(self):
        """Given an axial resisitivity, set the coupling conductances."""
        nbranches = self.total_nbranches
        nseg = self.nseg
        parents = self.comb_parents

        axial_resistivity = jnp.reshape(
            self.params["axial_resistivity"], (nbranches, nseg)
        )
        radiuses = jnp.reshape(self.params["radius"], (nbranches, nseg))
        lengths = jnp.reshape(self.params["length"], (nbranches, nseg))

        def compute_coupling_cond(rad1, rad2, r_a, l1, l2):
            return rad1 * rad2**2 / r_a / (rad2**2 * l1 + rad1**2 * l2) / l1

        # Compute coupling conductance for segments within a branch.
        # `radius`: um
        # `r_a`: ohm cm
        # `length_single_compartment`: um
        # `coupling_conds`: S * um / cm / um^2 = S / cm / um
        # Compute coupling conductance for segments at branch points.
        rad1 = radiuses[jnp.arange(1, nbranches), -1]
        rad2 = radiuses[parents[jnp.arange(1, nbranches)], 0]
        l1 = lengths[jnp.arange(1, nbranches), -1]
        l2 = lengths[parents[jnp.arange(1, nbranches)], 0]
        r_a1 = axial_resistivity[jnp.arange(1, nbranches), -1]
        r_a2 = axial_resistivity[parents[jnp.arange(1, nbranches)], 0]
        self.branch_conds_bwd = compute_coupling_cond(rad2, rad1, r_a1, l2, l1)
        self.branch_conds_fwd = compute_coupling_cond(rad1, rad2, r_a2, l1, l2)

        # Convert (S / cm / um) -> (mS / cm^2)
        self.branch_conds_fwd *= 10**7
        self.branch_conds_bwd *= 10**7

        for b in range(1, nbranches):
            self.summed_coupling_conds = self.summed_coupling_conds.at[b, -1].add(
                self.branch_conds_fwd[b - 1]
            )
            self.summed_coupling_conds = self.summed_coupling_conds.at[
                parents[b], 0
            ].add(self.branch_conds_bwd[b - 1])

        self.branch_conds_fwd = jnp.concatenate(
            [jnp.asarray([0.0]), self.branch_conds_fwd]
        )
        self.branch_conds_bwd = jnp.concatenate(
            [jnp.asarray([0.0]), self.branch_conds_bwd]
        )
        self.initialized_conds = True


class CellView(View):
    def __init__(self, pointer, view):
        super().__init__(pointer, view)

    def __call__(self, index):
        return super().adjust_view("cell_index", index)

    def __getattr__(self, key):
        assert key == "branch"
        return BranchView(self.pointer, self.view)
