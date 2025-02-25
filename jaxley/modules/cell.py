# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import jax.numpy as jnp
import numpy as np
import pandas as pd

from jaxley.modules.base import Module
from jaxley.modules.branch import Branch
from jaxley.utils.cell_utils import (
    build_branchpoint_group_inds,
    compute_children_and_parents,
    compute_children_in_level,
    compute_children_indices,
    compute_levels,
    compute_morphology_indices_in_levels,
    compute_parents_in_level,
)
from jaxley.utils.misc_utils import cumsum_leading_zero, deprecated_kwargs
from jaxley.utils.solver_utils import (
    JaxleySolveIndexer,
    comp_edges_to_indices,
    remap_index_to_masked,
)


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
        self._cumsum_nbranches = np.asarray([0, len(branch_list)])

        # Compartment structure. These arguments have to be rebuilt when `.set_ncomp()`
        # is run.
        self.ncomp_per_branch = np.asarray([branch.ncomp for branch in branch_list])
        self.ncomp = int(np.max(self.ncomp_per_branch))
        self.cumsum_ncomp = cumsum_leading_zero(self.ncomp_per_branch)
        self._internal_node_inds = np.arange(self.cumsum_ncomp[-1])

        # Build nodes. Has to be changed when `.set_ncomp()` is run.
        self.nodes = pd.concat([c.nodes for c in branch_list], ignore_index=True)
        self.nodes["global_comp_index"] = np.arange(self.cumsum_ncomp[-1])
        self.nodes["global_branch_index"] = np.repeat(
            np.arange(self.total_nbranches), self.ncomp_per_branch
        ).tolist()
        self.nodes["global_cell_index"] = np.repeat(0, self.cumsum_ncomp[-1]).tolist()
        self._update_local_indices()
        self._init_view()

        # Appending general parameters (radius, length, r_a, cm) and channel parameters,
        # as well as the states (v, and channel states).
        self._append_params_and_states(self.cell_params, self.cell_states)

        # Channels.
        self._gather_channels_from_constituents(branch_list)

        self.branch_edges = pd.DataFrame(
            dict(
                parent_branch_index=self.comb_parents[1:],
                child_branch_index=np.arange(1, self.total_nbranches),
            )
        )

        # For morphology indexing.
        self._par_inds, self._child_inds, self._child_belongs_to_branchpoint = (
            compute_children_and_parents(self.branch_edges)
        )

        self._initialize()

    def _init_morph_jaxley_spsolve(self):
        """Initialize morphology for the custom sparse solver.

        Running this function is only required for custom Jaxley solvers, i.e., for
        `voltage_solver={'jaxley.stone', 'jaxley.thomas'}`. However, because at
        `.__init__()` (when the function is run), we do not yet know which solver the
        user will use. Therefore, we always run this function at `.__init__()`.
        """
        children_and_parents = compute_morphology_indices_in_levels(
            len(self._par_inds),
            self._child_belongs_to_branchpoint,
            self._par_inds,
            self._child_inds,
        )
        branchpoint_group_inds = build_branchpoint_group_inds(
            len(self._par_inds),
            self._child_belongs_to_branchpoint,
            self.cumsum_ncomp[-1],
        )
        parents = self.comb_parents
        children_inds = children_and_parents["children"]
        parents_inds = children_and_parents["parents"]

        levels = compute_levels(parents)
        children_in_level = compute_children_in_level(levels, children_inds)
        parents_in_level = compute_parents_in_level(
            levels, self._par_inds, parents_inds
        )
        levels_and_ncomp = pd.DataFrame().from_dict(
            {
                "levels": levels,
                "ncomps": self.ncomp_per_branch,
            }
        )
        levels_and_ncomp["max_ncomp_in_level"] = levels_and_ncomp.groupby("levels")[
            "ncomps"
        ].transform("max")
        padded_cumsum_ncomp = cumsum_leading_zero(
            levels_and_ncomp["max_ncomp_in_level"].to_numpy()
        )

        # Generate mapping to deal with the masking which allows using the custom
        # sparse solver to deal with different ncomp per branch.
        remapped_node_indices = remap_index_to_masked(
            self._internal_node_inds,
            self.nodes,
            padded_cumsum_ncomp,
            self.ncomp_per_branch,
        )
        self._solve_indexer = JaxleySolveIndexer(
            cumsum_ncomp=padded_cumsum_ncomp,
            ncomp_per_branch=self.ncomp_per_branch,
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
                        "source": list(range(cumsum_ncomp, ncomp - 1 + cumsum_ncomp))
                        + list(range(1 + cumsum_ncomp, ncomp + cumsum_ncomp)),
                        "sink": list(range(1 + cumsum_ncomp, ncomp + cumsum_ncomp))
                        + list(range(cumsum_ncomp, ncomp - 1 + cumsum_ncomp)),
                    }
                )
                .astype(int)
                for ncomp, cumsum_ncomp in zip(self.ncomp_per_branch, self.cumsum_ncomp)
            ]
        )
        self._comp_edges["type"] = 0

        # Edges from branchpoints to compartments.
        branchpoint_to_parent_edges = pd.DataFrame().from_dict(
            {
                "source": np.arange(len(self._par_inds)) + self.cumsum_ncomp[-1],
                "sink": self.cumsum_ncomp[self._par_inds + 1] - 1,
                "type": 1,
            }
        )
        branchpoint_to_child_edges = pd.DataFrame().from_dict(
            {
                "source": self._child_belongs_to_branchpoint + self.cumsum_ncomp[-1],
                "sink": self.cumsum_ncomp[self._child_inds],
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
