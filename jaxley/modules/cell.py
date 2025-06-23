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
        # Edges between compartments within the branches.
        comp_edges = pd.concat(
            [
                pd.DataFrame()
                .from_dict(
                    {
                        "source": list(range(cumsum_ncomp, ncomp - 1 + cumsum_ncomp))
                        + list(range(1 + cumsum_ncomp, ncomp + cumsum_ncomp)),
                        "sink": list(range(1 + cumsum_ncomp, ncomp + cumsum_ncomp))
                        + list(range(cumsum_ncomp, ncomp - 1 + cumsum_ncomp)),
                        "ordered": [1] * (ncomp - 1) + [0] * (ncomp - 1),
                    }
                )
                .astype(int)
                for ncomp, cumsum_ncomp in zip(self.ncomp_per_branch, self.cumsum_ncomp)
            ]
        )
        comp_edges["type"] = 0

        # Edges from branchpoints to compartments.
        branchpoint_to_parent_edges = pd.DataFrame().from_dict(
            {
                "source": np.arange(len(self._par_inds)) + self.cumsum_ncomp[-1],
                "sink": self.cumsum_ncomp[self._par_inds + 1] - 1,
                "type": 1,
                "ordered": 0,
            }
        )
        branchpoint_to_child_edges = pd.DataFrame().from_dict(
            {
                "source": self._child_belongs_to_branchpoint + self.cumsum_ncomp[-1],
                "sink": self.cumsum_ncomp[self._child_inds],
                "type": 2,
                "ordered": 1,
            }
        )
        comp_edges = pd.concat(
            [
                comp_edges,
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
        parent_to_branchpoint_edges["ordered"] = 1
        child_to_branchpoint_edges = branchpoint_to_child_edges.rename(
            columns={"sink": "source", "source": "sink"}
        )
        child_to_branchpoint_edges["type"] = 4
        child_to_branchpoint_edges["ordered"] = 0

        self._comp_edges = pd.concat(
            [
                comp_edges,
                parent_to_branchpoint_edges,
                child_to_branchpoint_edges,
            ],
            ignore_index=True,
        )

        # Branchpoints.
        #
        # Get last xyz of parent.
        branchpoint_xyz = []
        for i in self._par_inds:
            branchpoint_xyz.append(self.xyzr[i][-1, :3])
        if len(branchpoint_xyz) > 0:
            # It is initialized as empty pd.DataFrame in `base.py`.
            self._branchpoints = pd.DataFrame(
                np.asarray(branchpoint_xyz), columns=["x", "y", "z"]
            )
            # Create offset for the branchpoints.
            self._branchpoints.index = (
                np.arange(len(self._par_inds)) + self.cumsum_ncomp[-1]
            )

        sources = np.asarray(self._comp_edges["source"].to_list())
        sinks = np.asarray(self._comp_edges["sink"].to_list())
        self._n_nodes = np.max(sinks) + 1 if len(sinks) > 0 else 1
        self._off_diagonal_inds = jnp.stack([sources, sinks]).astype(int)
