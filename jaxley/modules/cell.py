# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd

from jaxley.modules.base import Module
from jaxley.modules.branch import Branch, Compartment
from jaxley.synapses import Synapse
from jaxley.utils.cell_utils import (
    build_branchpoint_group_inds,
    compute_children_and_parents,
    compute_children_in_level,
    compute_children_indices,
    compute_levels,
    compute_morphology_indices_in_levels,
    compute_parents_in_level,
)
from jaxley.utils.misc_utils import cumsum_leading_zero
from jaxley.utils.solver_utils import (
    JaxleySolveIndexer,
    comp_edges_to_indices,
    remap_index_to_masked,
)
from jaxley.utils.swc import build_radiuses_from_xyzr, swc_to_jaxley


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

        self.total_nbranches = len(branch_list)
        self.nbranches_per_cell = [len(branch_list)]
        self.comb_parents = jnp.asarray(parents)
        self.comb_children = compute_children_indices(self.comb_parents)
        self.cumsum_nbranches = jnp.asarray([0, len(branch_list)])

        # Compartment structure. These arguments have to be rebuilt when `.set_ncomp()`
        # is run.
        self.nseg_per_branch = np.asarray([branch.nseg for branch in branch_list])
        self.nseg = int(np.max(self.nseg_per_branch))
        self.cumsum_nseg = cumsum_leading_zero(self.nseg_per_branch)
        self._internal_node_inds = np.arange(self.cumsum_nseg[-1])

        # Build nodes. Has to be changed when `.set_ncomp()` is run.
        self.nodes = pd.concat([c.nodes for c in branch_list], ignore_index=True)
        self.nodes["global_comp_index"] = np.arange(self.cumsum_nseg[-1])
        self.nodes["global_branch_index"] = np.repeat(
            np.arange(self.total_nbranches), self.nseg_per_branch
        ).tolist()
        self.nodes["global_cell_index"] = np.repeat(0, self.cumsum_nseg[-1]).tolist()
        self._in_view = self.nodes.index.to_numpy()
        self.nodes["controlled_by_param"] = 0
        self._update_local_indices()

        # Appending general parameters (radius, length, r_a, cm) and channel parameters,
        # as well as the states (v, and channel states).
        self._append_params_and_states(self.cell_params, self.cell_states)

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

        # For morphology indexing.
        self.par_inds, self.child_inds, self.child_belongs_to_branchpoint = (
            compute_children_and_parents(self.branch_edges)
        )

        self.initialize()
        self.init_syns()

    def _init_morph_jaxley_spsolve(self):
        """Initialize morphology for the custom sparse solver.

        Running this function is only required for custom Jaxley solvers, i.e., for
        `voltage_solver={'jaxley.stone', 'jaxley.thomas'}`. However, because at
        `.__init__()` (when the function is run), we do not yet know which solver the
        user will use. Therefore, we always run this function at `.__init__()`.
        """
        children_and_parents = compute_morphology_indices_in_levels(
            len(self.par_inds),
            self.child_belongs_to_branchpoint,
            self.par_inds,
            self.child_inds,
        )
        branchpoint_group_inds = build_branchpoint_group_inds(
            len(self.par_inds),
            self.child_belongs_to_branchpoint,
            self.cumsum_nseg[-1],
        )
        parents = self.comb_parents
        children_inds = children_and_parents["children"]
        parents_inds = children_and_parents["parents"]

        levels = compute_levels(parents)
        children_in_level = compute_children_in_level(levels, children_inds)
        parents_in_level = compute_parents_in_level(levels, self.par_inds, parents_inds)
        levels_and_nseg = pd.DataFrame().from_dict(
            {
                "levels": levels,
                "nsegs": self.nseg_per_branch,
            }
        )
        levels_and_nseg["max_nseg_in_level"] = levels_and_nseg.groupby("levels")[
            "nsegs"
        ].transform("max")
        padded_cumsum_nseg = cumsum_leading_zero(
            levels_and_nseg["max_nseg_in_level"].to_numpy()
        )

        # Generate mapping to deal with the masking which allows using the custom
        # sparse solver to deal with different nseg per branch.
        remapped_node_indices = remap_index_to_masked(
            self._internal_node_inds,
            self.nodes,
            padded_cumsum_nseg,
            self.nseg_per_branch,
        )
        self.solve_indexer = JaxleySolveIndexer(
            cumsum_nseg=padded_cumsum_nseg,
            branchpoint_group_inds=branchpoint_group_inds,
            children_in_level=children_in_level,
            parents_in_level=parents_in_level,
            root_inds=np.asarray([0]),
            remapped_node_indices=remapped_node_indices,
        )

    def _init_morph_jax_spsolve(self):
        """For morphology indexing with the `jax.sparse` voltage volver.

        Explanation of `self._comp_eges['type']`:
        `type == 0`: compartment <--> compartment (within branch)
        `type == 1`: branchpoint --> parent-compartment
        `type == 2`: branchpoint --> child-compartment
        `type == 3`: parent-compartment --> branchpoint
        `type == 4`: child-compartment --> branchpoint

        Running this function is only required for generic sparse solvers, i.e., for
        `voltage_solver='jax.sparse'`.
        """

        # Edges between compartments within the branches.
        self._comp_edges = pd.concat(
            [
                pd.DataFrame()
                .from_dict(
                    {
                        "source": list(range(cumsum_nseg, nseg - 1 + cumsum_nseg))
                        + list(range(1 + cumsum_nseg, nseg + cumsum_nseg)),
                        "sink": list(range(1 + cumsum_nseg, nseg + cumsum_nseg))
                        + list(range(cumsum_nseg, nseg - 1 + cumsum_nseg)),
                    }
                )
                .astype(int)
                for nseg, cumsum_nseg in zip(self.nseg_per_branch, self.cumsum_nseg)
            ]
        )
        self._comp_edges["type"] = 0

        # Edges from branchpoints to compartments.
        branchpoint_to_parent_edges = pd.DataFrame().from_dict(
            {
                "source": np.arange(len(self.par_inds)) + self.cumsum_nseg[-1],
                "sink": self.cumsum_nseg[self.par_inds + 1] - 1,
                "type": 1,
            }
        )
        branchpoint_to_child_edges = pd.DataFrame().from_dict(
            {
                "source": self.child_belongs_to_branchpoint + self.cumsum_nseg[-1],
                "sink": self.cumsum_nseg[self.child_inds],
                "type": 2,
            }
        )
        self._comp_edges = pd.concat(
            [
                self._comp_edges,
                branchpoint_to_parent_edges,
                branchpoint_to_child_edges,
            ],
            ignore_index=True,
        )

        # Edges from compartments to branchpoints.
        parent_to_branchpoint_edges = branchpoint_to_parent_edges.rename(
            columns={"sink": "source", "source": "sink"}
        )
        parent_to_branchpoint_edges["type"] = 3
        child_to_branchpoint_edges = branchpoint_to_child_edges.rename(
            columns={"sink": "source", "source": "sink"}
        )
        child_to_branchpoint_edges["type"] = 4

        self._comp_edges = pd.concat(
            [
                self._comp_edges,
                parent_to_branchpoint_edges,
                child_to_branchpoint_edges,
            ],
            ignore_index=True,
        )

        n_nodes, data_inds, indices, indptr = comp_edges_to_indices(self._comp_edges)
        self._n_nodes = n_nodes
        self._data_inds = data_inds
        self._indices_jax_spsolve = indices
        self._indptr_jax_spsolve = indptr

    @staticmethod
    def update_summed_coupling_conds_jaxley_spsolve(
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

    def set_ncomp(self, ncomp: int, min_radius: Optional[float] = None):
        """Raise an explict error if `set_ncomp` is set for an entire cell."""
        raise NotImplementedError(
            "`cell.set_ncomp()` is not supported. Loop over all branches with "
            "`for b in range(cell.total_nbranches): cell.branch(b).set_ncomp(n)`."
        )

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

    comp = Compartment()
    branch = Branch([comp for _ in range(nseg)])
    cell = Cell(
        [branch for _ in range(nbranches)], parents=parents, xyzr=coords_of_branches
    )
    # Also save the radius generating functions in case users post-hoc modify the number
    # of compartments with `.set_ncomp()`.
    cell._radius_generating_fns = radius_fns

    lengths_each = np.repeat(pathlengths, nseg) / nseg
    cell.set("length", lengths_each)

    radiuses_each = build_radiuses_from_xyzr(
        radius_fns,
        range(len(parents)),
        min_radius,
        nseg,
    )
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
