# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from copy import deepcopy
from itertools import chain
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd

from jaxley.modules.base import Module
from jaxley.modules.compartment import Compartment
from jaxley.utils.cell_utils import compute_children_and_parents
from jaxley.utils.misc_utils import cumsum_leading_zero
from jaxley.utils.solver_utils import JaxleySolveIndexer, comp_edges_to_indices


class Branch(Module):
    """Branch class.

    This class defines a single branch that can be simulated by itself or
    connected to build a cell. A branch is linear segment of several compartments
    and can be connected to no, one or more other branches at each end to build more
    intricate cell morphologies.
    """

    branch_params: Dict = {}
    branch_states: Dict = {}

    def __init__(
        self,
        compartments: Optional[Union[Compartment, List[Compartment]]] = None,
        nseg: Optional[int] = None,
    ):
        """
        Args:
            compartments: A single compartment or a list of compartments that make up the
                branch.
            nseg: Number of segments to divide the branch into. If `compartments` is an
                a single compartment, than the compartment is repeated `nseg` times to
                create the branch.
        """
        super().__init__()
        assert (
            isinstance(compartments, (Compartment, List)) or compartments is None
        ), "Only Compartment or List[Compartment] is allowed."
        if isinstance(compartments, Compartment):
            assert (
                nseg is not None
            ), "If `compartments` is not a list then you have to set `nseg`."
        compartments = Compartment() if compartments is None else compartments
        nseg = 1 if nseg is None else nseg

        if isinstance(compartments, Compartment):
            compartment_list = [compartments] * nseg
        else:
            compartment_list = compartments

        self.nseg = len(compartment_list)
        self.nseg_per_branch = np.asarray([self.nseg])
        self.total_nbranches = 1
        self.nbranches_per_cell = [1]
        self.cumsum_nbranches = jnp.asarray([0, 1])
        self.cumsum_nseg = cumsum_leading_zero(self.nseg_per_branch)

        # Indexing.
        self.nodes = pd.concat([c.nodes for c in compartment_list], ignore_index=True)
        self._append_params_and_states(self.branch_params, self.branch_states)
        self.nodes["global_comp_index"] = np.arange(self.nseg).tolist()
        self.nodes["global_branch_index"] = [0] * self.nseg
        self.nodes["global_cell_index"] = [0] * self.nseg
        self._update_local_indices()
        self._init_view()

        # Channels.
        self._gather_channels_from_constituents(compartment_list)

        # Synapse indexing.
        self.syn_edges = pd.DataFrame(
            dict(global_pre_comp_index=[], global_post_comp_index=[], type="")
        )
        self.branch_edges = pd.DataFrame(
            dict(parent_branch_index=[], child_branch_index=[])
        )

        # For morphology indexing.
        self.par_inds, self.child_inds, self.child_belongs_to_branchpoint = (
            compute_children_and_parents(self.branch_edges)
        )
        self._internal_node_inds = jnp.arange(self.nseg)

        self.initialize()
        self.init_syns()

        # Coordinates.
        self.xyzr = [float("NaN") * np.zeros((2, 4))]

    def _init_morph_jaxley_spsolve(self):
        self.solve_indexer = JaxleySolveIndexer(
            cumsum_nseg=self.cumsum_nseg,
            branchpoint_group_inds=np.asarray([]).astype(int),
            remapped_node_indices=self._internal_node_inds,
            children_in_level=[],
            parents_in_level=[],
            root_inds=np.asarray([0]),
        )

    def _init_morph_jax_spsolve(self):
        """Initialize morphology for the jax sparse voltage solver.

        Explanation of `self._comp_eges['type']`:
        `type == 0`: compartment <--> compartment (within branch)
        `type == 1`: branchpoint --> parent-compartment
        `type == 2`: branchpoint --> child-compartment
        `type == 3`: parent-compartment --> branchpoint
        `type == 4`: child-compartment --> branchpoint
        """
        self._comp_edges = pd.DataFrame().from_dict(
            {
                "source": list(range(self.nseg - 1)) + list(range(1, self.nseg)),
                "sink": list(range(1, self.nseg)) + list(range(self.nseg - 1)),
            }
        )
        self._comp_edges["type"] = 0
        n_nodes, data_inds, indices, indptr = comp_edges_to_indices(self._comp_edges)
        self._n_nodes = n_nodes
        self._data_inds = data_inds
        self._indices_jax_spsolve = indices
        self._indptr_jax_spsolve = indptr

    def __len__(self) -> int:
        return self.nseg
