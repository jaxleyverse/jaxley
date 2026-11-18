# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd

from jaxley.modules.base import Module
from jaxley.modules.branch import Branch
from jaxley.utils.cell_utils import (
    compute_children_and_parents,
    compute_children_indices,
)
from jaxley.utils.misc_utils import cumsum_leading_zero, deprecated_kwargs


class Cell(Module):
    """A cell made up of one or multiple branches (with branchpoints).

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
        self.comb_parents = np.asarray(parents)
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
        )
        self.nodes["global_cell_index"] = np.zeros(self.cumsum_ncomp[-1], dtype=int)

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

        # Compartment edges.
        self.initialize()

    def _init_comp_graph(self):
        """Initialize attributes concerning the compartment graph.

        In particular, it initializes:
        - `_comp_edges`
        - `_branchpoints`
        - `_n_nodes`
        - `_off_diagonal_inds`

        Explanation of `self._comp_eges['type']`:
        `type == 0`: compartment <--> compartment (within branch)
        `type == 1`: branchpoint --> parent-compartment
        `type == 2`: branchpoint --> child-compartment
        `type == 3`: parent-compartment --> branchpoint
        `type == 4`: child-compartment --> branchpoint
        """

        def edges(source, sink, ordered, type):
            n = len(source)
            return np.stack([source, sink, np.full(n, ordered), np.full(n, type)], 1)

        # Edges between compartments within the branches. For every branch, first the
        # edges towards its end (`ordered=1`), then the edges back (`ordered=0`).
        nedges = self.ncomp_per_branch - 1
        branch = np.repeat(np.arange(len(nedges)), nedges)
        position = np.arange(len(branch)) - np.repeat(
            np.cumsum(nedges) - nedges, nedges
        )
        comp = self.cumsum_ncomp[branch] + position  # All but the last of each branch.
        within_branch = np.concatenate(
            [edges(comp, comp + 1, 1, 0), edges(comp + 1, comp, 0, 0)]
        )
        order = np.argsort(np.concatenate([2 * branch, 2 * branch + 1]), kind="stable")

        # Edges between branchpoints and the last compartment of the parent branch or
        # the first compartment of a child branch.
        parent_bp = np.arange(len(self._par_inds)) + self.cumsum_ncomp[-1]
        parent = self.cumsum_ncomp[self._par_inds + 1] - 1
        child_bp = self._child_belongs_to_branchpoint + self.cumsum_ncomp[-1]
        child = self.cumsum_ncomp[self._child_inds]

        all_edges = [
            within_branch[order],
            edges(parent_bp, parent, 0, 1),
            edges(child_bp, child, 1, 2),
            edges(parent, parent_bp, 1, 3),
            edges(child, child_bp, 0, 4),
        ]
        self._comp_edges = pd.DataFrame(
            np.concatenate(all_edges).astype(int),
            columns=["source", "sink", "ordered", "type"],
        )

        # Branchpoints are at the last traced point of their parent branch. Without
        # branchpoints, `_branchpoints` stays the empty DataFrame from `base.py`.
        if len(self._par_inds) > 0:
            self._branchpoints = pd.DataFrame(
                np.asarray([self.xyzr[i][-1, :3] for i in self._par_inds]),
                columns=["x", "y", "z"],
                index=parent_bp,
            )

        sources = self._comp_edges["source"].to_numpy()
        sinks = self._comp_edges["sink"].to_numpy()
        self._n_nodes = np.max(sinks) + 1 if len(sinks) > 0 else 1
        self._off_diagonal_inds = np.stack([sources, sinks])
