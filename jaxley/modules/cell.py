# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import time
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import vmap
from jax.lax import ScatterDimensionNumbers, scatter_add

from jaxley.modules.base import GroupView, Module, View
from jaxley.modules.branch import Branch, BranchView, Compartment
from jaxley.synapses import Synapse
from jaxley.utils.cell_utils import (
    build_branchpoint_group_inds,
    comp_edges_to_indices,
    compute_children_in_level,
    compute_children_indices,
    compute_coupling_cond,
    compute_coupling_cond_branchpoint,
    compute_impact_on_node,
    compute_levels,
    compute_morphology_indices_in_levels,
    compute_parents_in_level,
    loc_of_index,
    remap_to_consecutive,
)
from jaxley.utils.swc import swc_to_jaxley


class Cell(Module):
    """Cell class.

    This class defines a single cell that can be simulated by itself or
    connected with synapses to build a network. A cell is made up of several branches
    and supports intricate cell morphologies.
    """

    cell_params: Dict = {}
    cell_states: Dict = {}

    def __init__(
        self,
        branches: Optional[Union[Branch, List[Branch]]] = None,
        parents: Optional[List[int]] = None,
        xyzr: Optional[List[np.ndarray]] = None,
    ):
        """Initialize a cell.

        Args:
            branches: A single branch or a list of branches that make up the cell.
                If a single branch is provided, then the branch is repeated `len(parents)`
                times to create the cell.
            parents: The parent branch index for each branch. The first branch has no
                parent and is therefore set to -1.
            xyzr: For every branch, the x, y, and z coordinates and the radius at the
                traced coordinates. Note that this is the full tracing (from SWC), not
                the stick representation coordinates.
        """
        super().__init__()
        assert (
            isinstance(branches, (Branch, List)) or branches is None
        ), "Only Branch or List[Branch] is allowed."
        if branches is not None:
            assert (
                parents is not None
            ), "If `branches` is not a list then you have to set `parents`."
        if isinstance(branches, List):
            assert len(parents) == len(
                branches
            ), "Ensure equally many parents, i.e. len(branches) == len(parents)."

        branches = Branch() if branches is None else branches
        parents = [-1] if parents is None else parents

        if isinstance(branches, Branch):
            self.branch_list = [branches for _ in range(len(parents))]
        else:
            self.branch_list = branches

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

        self.nseg_per_branch = jnp.asarray([branch.nseg for branch in self.branch_list])
        self.nseg = int(jnp.max(self.nseg_per_branch))
        self.cumsum_nseg = jnp.concatenate(
            [jnp.asarray([0]), jnp.cumsum(self.nseg_per_branch)]
        )
        self.total_nbranches = len(self.branch_list)
        self.nbranches_per_cell = [len(self.branch_list)]
        self.comb_parents = jnp.asarray(parents)
        self.comb_children = compute_children_indices(self.comb_parents)
        self.cumsum_nbranches = jnp.asarray([0, len(self.branch_list)])

        # Indexing.
        self.nodes = pd.concat([c.nodes for c in self.branch_list], ignore_index=True)
        self._append_params_and_states(self.cell_params, self.cell_states)
        self.nodes["comp_index"] = np.arange(self.cumsum_nseg[-1])
        self.nodes["branch_index"] = np.repeat(
            np.arange(self.total_nbranches), self.nseg_per_branch
        ).tolist()
        self.nodes["cell_index"] = np.repeat(0, self.cumsum_nseg[-1]).tolist()

        # Channels.
        self._gather_channels_from_constituents(self.branch_list)

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

        # For morphology indexing.
        par_inds = self.comb_parents[1:]
        self.child_inds = np.arange(1, self.total_nbranches)
        self.child_belongs_to_branchpoint = remap_to_consecutive(par_inds)
        self.par_inds = np.unique(par_inds)
        self.total_nbranchpoints = len(self.par_inds)
        self.root_inds = jnp.asarray([0])

        self.initialize()
        self.init_syns()

    def __getattr__(self, key: str):
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
            return GroupView(self, view, BranchView, ["branch"])
        else:
            raise KeyError(f"Key {key} not recognized.")

    def init_morph_custom_spsolve(self):
        """Initialize morphology for the custom sparse solver.

        Running this function is only required for custom Jaxley solvers, i.e., for
        `voltage_solver={'jaxley.stone', 'jaxley.thomas'}`.
        """
        children_and_parents = compute_morphology_indices_in_levels(
            len(self.par_inds),
            self.child_belongs_to_branchpoint,
            self.par_inds,
            self.child_inds,
        )
        self.branchpoint_group_inds = build_branchpoint_group_inds(
            len(self.par_inds),
            self.child_belongs_to_branchpoint,
            self.cumsum_nseg[-1],
        )
        parents = self.comb_parents
        children_inds = children_and_parents["children"]
        parents_inds = children_and_parents["parents"]

        levels = compute_levels(parents)
        self.children_in_level = compute_children_in_level(levels, children_inds)
        self.parents_in_level = compute_parents_in_level(
            levels, self.par_inds, parents_inds
        )

    def init_morph_jax_spsolve(self):
        """For morphology indexing with the `jax.sparse` voltage volver.

        Explanation of `type`:
        `type == 0`: compartment-to-compartment (within branch)
        `type == 1`: compartment-to-branchpoint
        `type == 2`: branchpoint-to-compartment

        Running this function is only required for generic sparse solvers, i.e., for
        `voltage_solver='jax.sparse'`.
        """
        # Edges between compartments within the branches.
        # `[offset, offset, 0]` because we want to offset `source` and `sink`, but
        # not `type`.
        self.comp_edges = pd.concat(
            [
                [offset, offset, 0] + branch.comp_edges
                for offset, branch in zip(self.cumsum_nseg, self.branch_list)
            ]
        )
        # `branch_list` is not needed anymore because all information it contained is
        # now also in `self.comp_edges`.
        del self.branch_list

        # Edges from compartments to branchpoints.
        child_to_branchpoint_edges = pd.DataFrame().from_dict(
            {
                "source": self.cumsum_nseg[self.child_inds],
                "sink": self.child_belongs_to_branchpoint + self.cumsum_nseg[-1],
                "type": 1,
            }
        )
        parent_to_branchpoint_edges = pd.DataFrame().from_dict(
            {
                "source": self.cumsum_nseg[self.par_inds + 1] - 1,
                "sink": np.arange(len(self.par_inds)) + self.cumsum_nseg[-1],
                "type": 1,
            }
        )
        self.comp_edges = pd.concat(
            [
                self.comp_edges,
                parent_to_branchpoint_edges,
                child_to_branchpoint_edges,
            ],
            ignore_index=True,
        )

        # Edges from branchpoints to compartments.
        branchpoint_to_child_edges = pd.DataFrame().from_dict(
            {
                "source": self.child_belongs_to_branchpoint + self.cumsum_nseg[-1],
                "sink": self.cumsum_nseg[self.child_inds],
                "type": 2,
            }
        )
        branchpoint_to_parent_edges = pd.DataFrame().from_dict(
            {
                "source": np.arange(len(self.par_inds)) + self.cumsum_nseg[-1],
                "sink": self.cumsum_nseg[self.par_inds + 1] - 1,
                "type": 2,
            }
        )
        self.comp_edges = pd.concat(
            [
                self.comp_edges,
                branchpoint_to_parent_edges,
                branchpoint_to_child_edges,
            ],
            ignore_index=True,
        )

        n_nodes, data_inds, indices, indptr = comp_edges_to_indices(self.comp_edges)
        self.n_nodes = n_nodes
        self.data_inds = data_inds
        self.indices = indices
        self.indptr = indptr

        # Helper methods for dealing with the masking.
        # This is actually used for the custom solver, but it requires info only
        # obtained here.
        nsegs = jnp.concatenate(
            [
                jnp.asarray([0]),
                jnp.cumsum(self.nseg_per_branch),
            ]
        )
        def remap_index_to_masked(index):
            """Convert actual index of the compartment to the index in the masked system.
            
            E.g. if `nsegs = [2, 4]`, then the index `3` would be mapped to `5` because the
            masked `nsegs` are `[4, 4]`.
            """
            branch_inds = self.nodes.loc[index, "branch_index"].to_numpy()
            remainders = index - nsegs[branch_inds]
            return branch_inds * self.nseg + remainders
        
        self.internal_node_inds = np.arange(self.cumsum_nseg[-1])
        self.remapped_indices = remap_index_to_masked(self.internal_node_inds)
        # branchpoint_inds = np.arange(self.n_nodes - np.max(self.internal_node_inds))
        # self.remapped_indices = np.concatenate([
        #     remapped_node_indices, branchpoint_inds + 1 + remapped_node_indices[-1]
        # ])
        print("self.remapped_indices", self.remapped_indices)

    def init_conds_custom_spsolve(self, params: Dict) -> Dict[str, jnp.ndarray]:
        """Given an axial resisitivity, set the coupling conductances."""

        nbranches = self.total_nbranches
        nseg = self.nseg

        axial_resistivity = jnp.reshape(params["axial_resistivity"], (nbranches, nseg))
        radiuses = jnp.reshape(params["radius"], (nbranches, nseg))
        lengths = jnp.reshape(params["length"], (nbranches, nseg))

        conds = vmap(Branch.init_branch_conds_custom_spsolve, in_axes=(0, 0, 0, None))(
            axial_resistivity, radiuses, lengths, self.nseg
        )
        coupling_conds_fwd = conds[0]
        coupling_conds_bwd = conds[1]
        summed_coupling_conds = conds[2]

        # The conductance from the children to the branch point.
        branchpoint_conds_children = vmap(
            compute_coupling_cond_branchpoint, in_axes=(0, 0, 0)
        )(
            radiuses[self.child_inds, 0],
            axial_resistivity[self.child_inds, 0],
            lengths[self.child_inds, 0],
        )

        branchpoint_conds_parents = vmap(
            compute_coupling_cond_branchpoint, in_axes=(0, 0, 0)
        )(
            radiuses[self.par_inds, -1],
            axial_resistivity[self.par_inds, -1],
            lengths[self.par_inds, -1],
        )

        # Weights with which the compartments influence their nearby node.
        # The impact of the children on the branch point.
        branchpoint_weights_children = vmap(compute_impact_on_node, in_axes=(0, 0, 0))(
            radiuses[self.child_inds, 0],
            axial_resistivity[self.child_inds, 0],
            lengths[self.child_inds, 0],
        )

        # The impact of parents on the branch point.
        branchpoint_weights_parents = vmap(compute_impact_on_node, in_axes=(0, 0, 0))(
            radiuses[self.par_inds, -1],
            axial_resistivity[self.par_inds, -1],
            lengths[self.par_inds, -1],
        )

        summed_coupling_conds = self.update_summed_coupling_conds_custom_spsolve(
            summed_coupling_conds,
            self.child_inds,
            self.par_inds,
            branchpoint_conds_children,
            branchpoint_conds_parents,
        )
        cond_params = {
            "branch_uppers": coupling_conds_bwd,
            "branch_lowers": coupling_conds_fwd,
            "branch_diags": summed_coupling_conds,
            "branchpoint_conds_children": branchpoint_conds_children,
            "branchpoint_conds_parents": branchpoint_conds_parents,
            "branchpoint_weights_children": branchpoint_weights_children,
            "branchpoint_conds_children": branchpoint_conds_children,
            "branchpoint_conds_parents": branchpoint_conds_parents,
            "branchpoint_weights_children": branchpoint_weights_children,
            "branchpoint_weights_parents": branchpoint_weights_parents,
        }

        ###################################################################### 
        ###################################################################### 
        ###################################################################### 
        ###################################################################### 
        ############################### NEW ################################### 
        ###################################################################### 
        ###################################################################### 
        ###################################################################### 
        ###################################################################### 
        # axial_resistivity = params["axial_resistivity"]
        # radiuses = params["radius"]
        # lengths = params["length"]

        # nsegs = jnp.concatenate(
        #     [
        #         jnp.asarray([0]),
        #         self.nseg_per_branch,
        #     ]
        # )

        # child_comp_ind = nsegs[self.child_inds]
        # par_comp_ind = nsegs[self.par_inds]

        # # The conductance from the children to the branch point.
        # branchpoint_conds_children = vmap(
        #     compute_coupling_cond_branchpoint, in_axes=(0, 0, 0)
        # )(
        #     radiuses[child_comp_ind],
        #     axial_resistivity[child_comp_ind],
        #     lengths[child_comp_ind],
        # )
        # # The conductance from the parents to the branch point.
        # branchpoint_conds_parents = vmap(
        #     compute_coupling_cond_branchpoint, in_axes=(0, 0, 0)
        # )(
        #     radiuses[par_comp_ind],
        #     axial_resistivity[par_comp_ind],
        #     lengths[par_comp_ind],
        # )

        # # Weights with which the compartments influence their nearby node.
        # # The impact of the children on the branch point.
        # branchpoint_weights_children = vmap(compute_impact_on_node, in_axes=(0, 0, 0))(
        #     radiuses[child_comp_ind],
        #     axial_resistivity[child_comp_ind],
        #     lengths[child_comp_ind],
        # )
        # # The impact of parents on the branch point.
        # branchpoint_weights_parents = vmap(compute_impact_on_node, in_axes=(0, 0, 0))(
        #     radiuses[par_comp_ind],
        #     axial_resistivity[par_comp_ind],
        #     lengths[par_comp_ind],
        # )

        # cond_params = {
        #     "branchpoint_conds_children": branchpoint_conds_children,
        #     "branchpoint_conds_parents": branchpoint_conds_parents,
        #     "branchpoint_weights_children": branchpoint_weights_children,
        #     "branchpoint_weights_parents": branchpoint_weights_parents,
        # }
        return cond_params

    def init_conds_jax_spsolve(self, params: Dict) -> Dict[str, jnp.ndarray]:
        """Given length, radius, and r_a, set the coupling conductances."""
        # `Compartment-to-compartment` conductances.
        condition = self.comp_edges["type"].to_numpy() == 0
        source_comp_inds = np.asarray(self.comp_edges[condition]["source"].to_list())
        sink_comp_inds = np.asarray(self.comp_edges[condition]["sink"].to_list())

        conds0 = vmap(compute_coupling_cond, in_axes=(0, 0, 0, 0, 0, 0))(
            params["radius"][source_comp_inds],
            params["radius"][sink_comp_inds],
            params["axial_resistivity"][source_comp_inds],
            params["axial_resistivity"][sink_comp_inds],
            params["length"][source_comp_inds],
            params["length"][sink_comp_inds],
        )

        # `branchpoint-to-compartment` conductances.
        condition = self.comp_edges["type"].to_numpy() == 1
        sink_comp_inds = np.asarray(self.comp_edges[condition]["sink"].to_list())

        conds1 = vmap(compute_coupling_cond_branchpoint, in_axes=(0, 0, 0))(
            params["radius"][sink_comp_inds],
            params["axial_resistivity"][sink_comp_inds],
            params["length"][sink_comp_inds],
        )

        # `compartment-to-branchpoint` conductances.
        condition = self.comp_edges["type"].to_numpy() == 2
        source_comp_inds = np.asarray(self.comp_edges[condition]["source"].to_list())

        conds2 = vmap(compute_coupling_cond_branchpoint, in_axes=(0, 0, 0))(
            params["radius"][source_comp_inds],
            params["axial_resistivity"][source_comp_inds],
            params["length"][source_comp_inds],
        )

        # All conductances.
        conds = jnp.concatenate([conds0, conds1, conds2])
        return {"axial_conductances": conds}

    @staticmethod
    def update_summed_coupling_conds_custom_spsolve(
        summed_conds,
        child_inds,
        par_inds,
        branchpoint_conds_children,
        branchpoint_conds_parents,
    ):
        """Perform updates on the diagonal based on conductances of the branchpoints.

        Args:
            summed_conds: shape [num_branches, nseg]
            child_inds: shape [num_branches - 1]
            conds_fwd: shape [num_branches - 1]
            conds_bwd: shape [num_branches - 1]
            parents: shape [num_branches]

        Returns:
            Updated `summed_coupling_conds`.
        """
        summed_conds = summed_conds.at[child_inds, 0].add(branchpoint_conds_children)
        summed_conds = summed_conds.at[par_inds, -1].add(branchpoint_conds_parents)
        return summed_conds


class CellView(View):
    """CellView."""

    def __init__(self, pointer: Module, view: pd.DataFrame):
        view = view.assign(controlled_by_param=view.global_cell_index)
        super().__init__(pointer, view)

    def __call__(self, index: float):
        local_idcs = self._get_local_indices()
        self.view[local_idcs.columns] = (
            local_idcs  # set indexes locally. enables net[0:2,0:2]
        )
        if index == "all":
            self.allow_make_trainable = False
        new_view = super().adjust_view("cell_index", index)
        return new_view

    def __getattr__(self, key: str):
        assert key == "branch"
        return BranchView(self.pointer, self.view)

    def rotate(self, degrees: float, rotation_axis: str = "xy"):
        """Rotate jaxley modules clockwise. Used only for visualization.

        Args:
            degrees: How many degrees to rotate the module by.
            rotation_axis: Either of {`xy` | `xz` | `yz`}.
        """
        nodes = self.set_global_index_and_index(self.view)
        self.pointer._rotate(degrees=degrees, rotation_axis=rotation_axis, view=nodes)


def read_swc(
    fname: str,
    nseg: int,
    max_branch_len: float = 300.0,
    min_radius: Optional[float] = None,
    assign_groups: bool = False,
) -> Cell:
    """Reads SWC file into a `jx.Cell`.

    Jaxley assumes cylindrical compartments and therefore defines length and radius
    for every compartment. The surface area is then 2*pi*r*length. For branches
    consisting of a single traced point we assume for them to have area 4*pi*r*r.
    Therefore, in these cases, we set lenght=2*r.

    Args:
        fname: Path to the swc file.
        nseg: The number of compartments per branch.
        max_branch_len: If a branch is longer than this value it is split into two
            branches.
        min_radius: If the radius of a reconstruction is below this value it is clipped.
        assign_groups: If True, then the identity of reconstructed points in the SWC
            file will be used to generate groups `undefined`, `soma`, `axon`, `basal`,
            `apical`, `custom`. See here:
            http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html

    Returns:
        A `jx.Cell` object.
    """
    parents, pathlengths, radius_fns, types, coords_of_branches = swc_to_jaxley(
        fname, max_branch_len=max_branch_len, sort=True, num_lines=None
    )
    nbranches = len(parents)

    non_split = 1 / nseg
    range_ = np.linspace(non_split / 2, 1 - non_split / 2, nseg)

    comp = Compartment()
    branch = Branch([comp for _ in range(nseg)])
    cell = Cell(
        [branch for _ in range(nbranches)], parents=parents, xyzr=coords_of_branches
    )

    radiuses = np.asarray([radius_fns[b](range_) for b in range(len(parents))])
    radiuses_each = radiuses.ravel(order="C")
    if min_radius is None:
        assert np.all(
            radiuses_each > 0.0
        ), "Radius 0.0 in SWC file. Set `read_swc(..., min_radius=...)`."
    else:
        radiuses_each[radiuses_each < min_radius] = min_radius

    lengths_each = np.repeat(pathlengths, nseg) / nseg

    cell.set("length", lengths_each)
    cell.set("radius", radiuses_each)

    # Description of SWC file format:
    # http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
    ind_name_lookup = {
        0: "undefined",
        1: "soma",
        2: "axon",
        3: "basal",
        4: "apical",
        5: "custom",
    }
    types = np.asarray(types).astype(int)
    if assign_groups:
        for type_ind in np.unique(types):
            if type_ind < 5.5:
                name = ind_name_lookup[type_ind]
            else:
                name = f"custom{type_ind}"
            indices = np.where(types == type_ind)[0].tolist()
            if len(indices) > 0:
                cell.branch(indices).add_to_group(name)
    return cell
