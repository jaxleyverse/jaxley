# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from jaxley.modules.base import Module
from jaxley.utils.cell_utils import compute_children_and_parents
from jaxley.utils.misc_utils import cumsum_leading_zero


class Compartment(Module):
    """A single compartment.

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

        self.ncomp = 1
        self.ncomp_per_branch = np.asarray([1])
        self.total_nbranches = 1
        self.nbranches_per_cell = [1]
        self._cumsum_nbranches = np.asarray([0, 1])
        self.cumsum_ncomp = cumsum_leading_zero(self.ncomp_per_branch)
        self._n_nodes = 1

        # Setting up the `nodes` for indexing.
        self.nodes = pd.DataFrame(
            dict(global_cell_index=[0], global_branch_index=[0], global_comp_index=[0])
        )
        self._append_params_and_states(self.compartment_params, self.compartment_states)
        self._update_local_indices()
        self._init_view()

        # Synapses.
        self.branch_edges = pd.DataFrame(
            dict(parent_branch_index=[], child_branch_index=[])
        )

        # For morphology indexing.
        self._par_inds, self._child_inds, self._child_belongs_to_branchpoint = (
            compute_children_and_parents(self.branch_edges)
        )
        self._internal_node_inds = jnp.asarray([0])

        # Coordinates.
        self.xyzr = [float("NaN") * np.zeros((2, 4))]

        # Initialize the module.
        self._initialize()

    def _init_comp_graph(self):
        """Initialize attributes concerning the compartment graph.

        In particular, it initializes:
        - `_comp_edges`
        - `_branchpoints`
        - `_comp_to_index_mapping`
        - `_comp_edges_in_view`
        - `_branchpoints_in_view`
        - `_n_nodes`
        - `_off_diagonal_inds`

        It also initializes `_comp_edges_in_view` and `_branchpoints_in_view`."""
        # Compartment edges.
        self._comp_edges = pd.DataFrame().from_dict(
            {"source": [], "sink": [], "type": []}
        )

        # To enable updating `self._comp_edges` and `self._branchpoints` during `View`.
        self._comp_edges_in_view = self._comp_edges.index.to_numpy()
        self._branchpoints_in_view = self._branchpoints.index.to_numpy()

        # Mapping from global_comp_index to `nodes.index`.
        comp_to_index_mapping = np.zeros((len(self.nodes)))
        comp_to_index_mapping[self.nodes["global_comp_index"].to_numpy()] = (
            self.nodes.index.to_numpy()
        )
        self._comp_to_index_mapping = comp_to_index_mapping.astype(int)
        self._n_nodes = 1
        self._off_diagonal_inds = np.asarray([])
