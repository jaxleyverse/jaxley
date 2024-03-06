from copy import deepcopy
from typing import Callable, Dict, List, Optional, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import vmap
from jax.lax import ScatterDimensionNumbers, scatter_add

from jaxley.modules.base import GroupView, Module, View
from jaxley.modules.branch import Branch, BranchView, Compartment
from jaxley.utils.cell_utils import (
    compute_branches_in_level,
    compute_coupling_cond,
    compute_levels,
)
from jaxley.utils.swc import swc_to_jaxley


class Cell(Module):
    """Cell."""

    cell_params: Dict = {}
    cell_states: Dict = {}

    def __init__(
        self,
        branches: Union[Branch, List[Branch]],
        parents: List,
        xyzr: Optional[List[np.ndarray]] = None,
    ):
        """Initialize a cell.

        Args:
            branches:
            parents:
            xyzr: For every branch, the x, y, and z coordinates and the radius at the
                traced coordinates. Note that this is the full tracing (from SWC), not
                the stick representation coordinates.
        """
        super().__init__()
        assert isinstance(branches, Branch) or len(parents) == len(
            branches
        ), "If `branches` is a list then you have to provide equally many parents, i.e. len(branches) == len(parents)."
        if isinstance(branches, Branch):
            branch_list = [branches for _ in range(len(parents))]
        else:
            branch_list = branches

        if xyzr is not None:
            assert len(xyzr) == len(parents)
            self.xyzr = xyzr
        else:
            # For every branch (`len(parents)`), we have a start and end point (`2`) and
            # a (x,y,z,r) coordinate for each of them (`4`).
            # Since `xyzr` is only inspected at `.vis()` and because it depends on the
            # (potentially learned) length of every compartment, we only populate
            # self.xyzr at `.vis()`.
            self.xyzr = [float("NaN") * np.zeros((2, 4)) for _ in range(len(parents))]

        self.nseg = branch_list[0].nseg
        self.total_nbranches = len(branch_list)
        self.nbranches_per_cell = [len(branch_list)]
        self.comb_parents = jnp.asarray(parents)
        self.cumsum_nbranches = jnp.asarray([0, len(branch_list)])

        # Indexing.
        self.nodes = pd.concat([c.nodes for c in branch_list], ignore_index=True)
        self._append_params_and_states(self.cell_params, self.cell_states)
        self.nodes["comp_index"] = np.arange(self.nseg * self.total_nbranches).tolist()
        self.nodes["branch_index"] = (
            np.arange(self.nseg * self.total_nbranches) // self.nseg
        ).tolist()
        self.nodes["cell_index"] = [0] * (self.nseg * self.total_nbranches)

        # Channels.
        self._gather_channels_from_constituents(branch_list)

        # Synapse indexing.
        self.syn_edges = pd.DataFrame(
            dict(global_pre_comp_index=[], global_post_comp_index=[], type="")
        )
        self.branch_edges = pd.DataFrame(
            dict(
                parent_branch_index=self.comb_parents[1:],
                child_branch_index=np.arange(1, self.total_nbranches),
            )
        )

        self.initialize()
        self.init_syns(None)
        self.initialized_conds = False

    def __getattr__(self, key):
        # Ensure that hidden methods such as `__deepcopy__` still work.
        if key.startswith("__"):
            return super().__getattribute__(key)

        if key == "branch":
            view = deepcopy(self.nodes)
            view["global_comp_index"] = view["comp_index"]
            view["global_branch_index"] = view["branch_index"]
            view["global_cell_index"] = view["cell_index"]
            return BranchView(self, view)
        elif key in self.group_nodes:
            inds = self.group_nodes[key].index.values
            view = self.nodes.loc[inds]
            view["global_comp_index"] = view["comp_index"]
            view["global_branch_index"] = view["branch_index"]
            view["global_cell_index"] = view["cell_index"]
            return GroupView(self, view)
        else:
            raise KeyError(f"Key {key} not recognized.")

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
        branch_conds_fwd = jnp.zeros((nbranches))
        branch_conds_bwd = jnp.zeros((nbranches))
        branch_conds_fwd = branch_conds_fwd.at[child_inds].set(conds[0])
        branch_conds_bwd = branch_conds_bwd.at[child_inds].set(conds[1])

        summed_coupling_conds = self.update_summed_coupling_conds(
            summed_coupling_conds,
            child_inds,
            branch_conds_fwd,
            branch_conds_bwd,
            parents,
        )

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
            r_child, r_parent, ra_child, ra_parent, l_child, l_parent
        )
        branch_conds_bwd = compute_coupling_cond(
            r_parent, r_child, ra_parent, ra_child, l_parent, l_child
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

        summed_conds = summed_conds.at[child_inds, -1].add(conds_bwd[child_inds])

        dnums = ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0, 1),
            scatter_dims_to_operand_dims=(0, 1),
        )
        summed_conds = scatter_add(
            summed_conds,
            jnp.stack([parents[child_inds], jnp.zeros_like(parents[child_inds])]).T,
            conds_fwd[child_inds],
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
        new_view = super().adjust_view("cell_index", index)
        new_view.view["comp_index"] -= new_view.view["comp_index"].iloc[0]
        new_view.view["branch_index"] -= new_view.view["branch_index"].iloc[0]
        return new_view

    def __getattr__(self, key):
        assert key == "branch"
        return BranchView(self.pointer, self.view)

    def fully_connect(self, post_cell_view, synapse_type):
        """Returns a list of `Connection`s which build a fully connected layer.

        Connections are from branch 0 location 0 to a randomly chosen branch and loc.
        """
        pre_cell_inds = np.unique(self.view["cell_index"].to_numpy())
        post_cell_inds = np.unique(post_cell_view.view["cell_index"].to_numpy())

        for pre_ind in pre_cell_inds:
            for post_ind in post_cell_inds:
                num_branches_post = self.pointer.nbranches_per_cell[post_ind]
                branch_pre = 0
                loc_pre = 0.0
                rand_branch_post = np.random.randint(0, num_branches_post)
                rand_loc_post = np.random.rand()
                pre = self.pointer.cell(pre_ind).branch(branch_pre).comp(loc_pre)
                post = (
                    self.pointer.cell(post_ind)
                    .branch(rand_branch_post)
                    .comp(rand_loc_post)
                )
                pre.connect(post, synapse_type)

    def sparse_connect(self, post_cell_view, p, synapse_type):
        """Returns a list of `Connection`s forming a sparse, randomly connected layer.

        Connections are from branch 0 location 0 to a randomly chosen branch and loc.
        """
        pre_cell_inds = np.unique(self.view["cell_index"].to_numpy())
        post_cell_inds = np.unique(post_cell_view.view["cell_index"].to_numpy())

        num_pre = len(pre_cell_inds)
        num_post = len(post_cell_inds)
        num_connections = np.random.binomial(num_pre * num_post, p)
        pre_syn_neurons = np.random.choice(pre_cell_inds, size=num_connections)
        post_syn_neurons = np.random.choice(post_cell_inds, size=num_connections)

        for pre_ind, post_ind in zip(pre_syn_neurons, post_syn_neurons):
            num_branches_post = self.pointer.nbranches_per_cell[post_ind]
            branch_pre = 0
            loc_pre = 0.0
            rand_branch_post = np.random.randint(0, num_branches_post)
            rand_loc_post = np.random.rand()

            pre = self.pointer.cell(pre_ind).branch(branch_pre).comp(loc_pre)
            post = (
                self.pointer.cell(post_ind).branch(rand_branch_post).comp(rand_loc_post)
            )
            pre.connect(post, synapse_type)

    def rotate(self, degrees: float, rotation_axis: str = "xy"):
        """Rotate jaxley modules clockwise. Used only for visualization.

        Args:
            degrees: How many degrees to rotate the module by.
            rotation_axis: Either of {`xy` | `xz` | `yz`}.
        """
        self.pointer._rotate(
            degrees=degrees, rotation_axis=rotation_axis, view=self.view
        )


def read_swc(
    fname: str,
    nseg: int,
    max_branch_len: float = 300.0,
    min_radius: Optional[float] = None,
):
    """Reads SWC file into a `jx.Cell`."""
    parents, pathlengths, radius_fns, _, coords_of_branches = swc_to_jaxley(
        fname, max_branch_len=max_branch_len, sort=True, num_lines=None
    )
    nbranches = len(parents)

    non_split = 1 / nseg
    range_ = np.linspace(non_split / 2, 1 - non_split / 2, nseg)

    comp = Compartment().initialize()
    branch = Branch([comp for _ in range(nseg)]).initialize()
    cell = Cell(
        [branch for _ in range(nbranches)], parents=parents, xyzr=coords_of_branches
    )

    radiuses = np.flip(
        np.asarray([radius_fns[b](range_) for b in range(len(parents))]), axis=1
    )
    radiuses_each = radiuses.flatten(order="C")
    if min_radius is None:
        assert np.all(
            radiuses_each > 0.0
        ), "Radius 0.0 in SWC file. Set `read_swc(..., min_radius=...)`."
    else:
        radiuses_each[radiuses_each < min_radius] = min_radius

    lengths_each = np.repeat(pathlengths, nseg) / nseg

    cell.set("length", lengths_each)
    cell.set("radius", radiuses_each)
    return cell
