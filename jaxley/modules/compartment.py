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
from jaxley.utils.morph_attributes import (
    compute_axial_conductances,
    cylinder_area,
    cylinder_resistive_load,
    cylinder_volume,
    morph_attrs_from_xyzr,
    split_xyzr_into_equal_length_segments,
)


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
        """Initialize a compartment."""
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

        l = self.compartment_params["length"]
        r = self.compartment_params["radius"]
        # l/2 because we want the input load (left half of the cylinder) and
        # the output load (right half of the cylinder).
        resistive_load = cylinder_resistive_load(l / 2, r)
        self.nodes["area"] = cylinder_area(l, r)
        self.nodes["volume"] = cylinder_volume(l, r)
        self.nodes["resistive_load_in"] = resistive_load
        self.nodes["resistive_load_out"] = resistive_load

        # Initialize the module.
        self.initialize()

    def _init_comp_graph(self):
        """Initialize attributes concerning the compartment graph.

        In particular, it initializes:
        - `_comp_edges`
        - `_branchpoints`
        - `_n_nodes`
        - `_off_diagonal_inds`

        It also initializes `_comp_edges_in_view` and `_branchpoints_in_view`."""
        # Compartment edges.
        self._comp_edges = pd.DataFrame().from_dict(
            {"source": [], "sink": [], "ordered": [], "type": []}
        )

        # Branchpoints.
        self._branchpoints = pd.DataFrame().from_dict({"x": [], "y": [], "z": []})

        # Mapping from global_comp_index to `nodes.index`.
        self._n_nodes = 1
        self._off_diagonal_inds = np.asarray([])
