# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from jaxley.modules.base import Module, View
from jaxley.utils.cell_utils import (
    compute_children_and_parents,
    interpolate_xyz,
    loc_of_index,
    local_index_of_loc,
)
from jaxley.utils.misc_utils import cumsum_leading_zero
from jaxley.utils.solver_utils import JaxleySolveIndexer, comp_edges_to_indices


class Compartment(Module):
    """Compartment class.

    This class defines a single compartment that can be simulated by itself or
    connected up into branches. It is the basic building block of a neuron model.
    """

    compartment_params: Dict = {
        "length": 10.0,  # um
        "radius": 1.0,  # um
        "axial_resistivity": 5_000.0,  # ohm cm
        "capacitance": 1.0,  # uF/cm^2
    }
    compartment_states: Dict = {"v": -70.0}

    def __init__(self):
        super().__init__()

        self.nseg = 1
        self.nseg_per_branch = np.asarray([1])
        self.total_nbranches = 1
        self.nbranches_per_cell = [1]
        self.cumsum_nbranches = jnp.asarray([0, 1])
        self.cumsum_nseg = cumsum_leading_zero(self.nseg_per_branch)

        # Setting up the `nodes` for indexing.
        self.nodes = pd.DataFrame(
            dict(comp_index=[0], branch_index=[0], cell_index=[0])
        )
        self._append_params_and_states(self.compartment_params, self.compartment_states)
        self._in_view = self.nodes.index.to_numpy()
        self.nodes["controlled_by_param"] = 0
        self._update_local_indices()
        
        # Synapses.
        self.branch_edges = pd.DataFrame(
            dict(parent_branch_index=[], child_branch_index=[])
        )

        # For morphology indexing.
        self.par_inds, self.child_inds, self.child_belongs_to_branchpoint = (
            compute_children_and_parents(self.branch_edges)
        )
        self._internal_node_inds = jnp.asarray([0])

        # Initialize the module.
        self.initialize()
        self.init_syns()

        # Coordinates.
        self.xyzr = [float("NaN") * np.zeros((2, 4))]

    def _init_morph_jaxley_spsolve(self):
        self.solve_indexer = JaxleySolveIndexer(
            cumsum_nseg=self.cumsum_nseg,
            branchpoint_group_inds=np.asarray([]).astype(int),
            children_in_level=[],
            parents_in_level=[],
            root_inds=np.asarray([0]),
            remapped_node_indices=self._internal_node_inds,
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
            {"source": [], "sink": [], "type": []}
        )
        n_nodes, data_inds, indices, indptr = comp_edges_to_indices(self._comp_edges)
        self._n_nodes = n_nodes
        self._data_inds = data_inds
        self._indices_jax_spsolve = indices
        self._indptr_jax_spsolve = indptr

    def init_conds(self, params: Dict[str, jnp.ndarray]):
        """Override `Base.init_axial_conds()`.

        This is because compartments do not have any axial conductances."""
        return {"axial_conductances": jnp.asarray([])}


class CompartmentView:
    # KEEP AROUND FOR NOW TO NOT BREAK EXISTING CODE
    pass