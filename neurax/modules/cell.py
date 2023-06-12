from typing import Callable, Dict, List, Optional
from copy import deepcopy

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import vmap
from jax.lax import ScatterDimensionNumbers, scatter_add

from neurax.modules.base import Module, View
from neurax.modules.branch import Branch, BranchView
from neurax.utils.cell_utils import (
    compute_branches_in_level,
    compute_coupling_cond,
    compute_levels,
)


class Cell(Module):
    """Cell."""

    cell_params: Dict = {}
    cell_states: Dict = {}

    def __init__(self, branches: List[Branch], parents: List):
        super().__init__()
        self._init_params_and_state(self.cell_params, self.cell_states)
        self._append_to_params_and_state(branches)
        self._append_to_channel_params_and_state(branches)

        self.nseg = branches[0].nseg
        self.total_nbranches = len(branches)
        self.nbranches_per_cell = [len(branches)]
        self.comb_parents = jnp.asarray(parents)
        self.cumsum_nbranches = jnp.asarray([0, len(branches)])

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

        # Channel indexing.
        for i, branch in enumerate(branches):
            for channel in branch.channels:
                name = type(channel).__name__
                comp_inds = deepcopy(
                    branch.channel_nodes[name]["comp_index"].to_numpy()
                )
                comp_inds += self.nseg * i
                index = pd.DataFrame.from_dict(
                    dict(
                        comp_index=comp_inds,
                        branch_index=[i] * len(comp_inds),
                        cell_index=[0] * len(comp_inds),
                    )
                )
                self._append_to_channel_nodes(index, channel)

        # Synapse indexing.
        self.syn_edges = pd.DataFrame(
            dict(pre_comp_index=[], post_comp_index=[], type="")
        )
        self.branch_edges = pd.DataFrame(
            dict(
                parent_branch_index=self.comb_parents[1:],
                child_branch_index=np.arange(1, self.total_nbranches),
            )
        )

        self.initialized_morph = False
        self.initialized_conds = False
        self.initialized_syns = True

    def __getattr__(self, key: str):
        assert key == "branch"
        return BranchView(self, self.nodes)

    def init_morph(self):
        """Initialize morphology."""
        parents = self.comb_parents

        levels = compute_levels(parents)
        self.comb_branches_in_each_level = compute_branches_in_level(levels)

        self.initialized_morph = True

    def init_conds(self, params):
        """Given an axial resisitivity, set the coupling conductances."""
        nbranches = self.total_nbranches
        nseg = self.nseg
        parents = self.comb_parents

        axial_resistivity = jnp.reshape(params["axial_resistivity"], (nbranches, nseg))
        radiuses = jnp.reshape(params["radius"], (nbranches, nseg))
        lengths = jnp.reshape(params["length"], (nbranches, nseg))

        conds = vmap(Branch.init_branch_conds, in_axes=(0, 0, 0, None))(
            axial_resistivity, radiuses, lengths, self.nseg
        )
        coupling_conds_fwd = conds[0]
        coupling_conds_bwd = conds[1]
        summed_coupling_conds = conds[2]

        par_inds = self.branch_edges["parent_branch_index"].to_numpy()
        child_inds = self.branch_edges["child_branch_index"].to_numpy()

        conds = vmap(self.init_cell_conds, in_axes=(0, 0, 0, 0, 0, 0))(
            axial_resistivity[child_inds, -1],
            axial_resistivity[par_inds, 0],
            radiuses[child_inds, -1],
            radiuses[par_inds, 0],
            lengths[child_inds, -1],
            lengths[par_inds, 0],
        )
        summed_coupling_conds = self.update_summed_coupling_conds(
            summed_coupling_conds,
            child_inds,
            conds[0],
            conds[1],
            parents,
        )

        branch_conds_fwd = jnp.zeros((nbranches))
        branch_conds_bwd = jnp.zeros((nbranches))
        branch_conds_fwd = branch_conds_fwd.at[child_inds].set(conds[0])
        branch_conds_bwd = branch_conds_bwd.at[child_inds].set(conds[1])

        cond_params = {
            "coupling_conds_fwd": coupling_conds_fwd,
            "coupling_conds_bwd": coupling_conds_bwd,
            "summed_coupling_conds": summed_coupling_conds,
            "branch_conds_fwd": branch_conds_fwd,
            "branch_conds_bwd": branch_conds_bwd,
        }
        return cond_params

    @staticmethod
    def init_cell_conds(ra_parent, ra_child, r_parent, r_child, l_parent, l_child):
        """Initializes the cell conductances, i.e., the ones at branch points.

        This method is used via vmap. Inputs should be scalar.

        `radius`: um
        `r_a`: ohm cm
        `length_single_compartment`: um
        `coupling_conds`: S * um / cm / um^2 = S / cm / um
        """
        branch_conds_fwd = compute_coupling_cond(
            r_child, r_parent, ra_parent, l_child, l_parent
        )
        branch_conds_bwd = compute_coupling_cond(
            r_parent, r_child, ra_child, l_parent, l_child
        )

        # Convert (S / cm / um) -> (mS / cm^2)
        branch_conds_fwd *= 10**7
        branch_conds_bwd *= 10**7

        return branch_conds_fwd, branch_conds_bwd

    @staticmethod
    def update_summed_coupling_conds(
        summed_conds, child_inds, conds_fwd, conds_bwd, parents
    ):
        """Perform updates on `summed_coupling_conds`.

        Args:
            summed_conds: shape [num_branches, nseg]
            child_inds: shape [num_branches - 1]
            conds_fwd: shape [num_branches - 1]
            conds_bwd: shape [num_branches - 1]
            parents: shape [num_branches]
        """

        summed_conds = summed_conds.at[child_inds, -1].add(conds_bwd[child_inds - 1])

        dnums = ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0, 1),
            scatter_dims_to_operand_dims=(0, 1),
        )
        summed_conds = scatter_add(
            summed_conds,
            jnp.stack([parents[child_inds], jnp.zeros_like(parents[child_inds])]).T,
            conds_fwd[child_inds - 1],
            dnums,
        )
        return summed_conds


class CellView(View):
    """CellView."""

    def __init__(self, pointer, view):
        view = view.assign(controlled_by_param=view.cell_index)
        super().__init__(pointer, view)

    def __call__(self, index: float):
        if index == "all":
            self.allow_make_trainable = False
        return super().adjust_view("cell_index", index)

    def __getattr__(self, key):
        assert key == "branch"
        return BranchView(self.pointer, self.view)
