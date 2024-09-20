# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from jaxley.modules.base import Module, View
from jaxley.utils.cell_utils import (
    comp_edges_to_indices,
    compute_children_and_parents,
    index_of_loc,
    interpolate_xyz,
    loc_of_index,
)


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
        self.nseg_per_branch = [1]
        self.total_nbranches = 1
        self.nbranches_per_cell = [1]
        self.cumsum_nbranches = jnp.asarray([0, 1])

        # Setting up the `nodes` for indexing.
        self.nodes = pd.DataFrame(
            dict(comp_index=[0], branch_index=[0], cell_index=[0])
        )
        self._append_params_and_states(self.compartment_params, self.compartment_states)

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

    def _init_morph_custom_spsolve(self):
        self.branchpoint_group_inds = np.asarray([]).astype(int)
        self.root_inds = jnp.asarray([0])
        self._remapped_node_indices = self._internal_node_inds
        self.children_in_level = []
        self.parents_in_level = []

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

    def _init_conds_custom_spsolve(self, params):
        return {
            "branchpoint_conds_children": jnp.asarray([]),
            "branchpoint_conds_parents": jnp.asarray([]),
            "branchpoint_weights_children": jnp.asarray([]),
            "branchpoint_weights_parents": jnp.asarray([]),
            "branch_uppers": jnp.asarray([]),
            "branch_lowers": jnp.asarray([]),
            "branch_diags": jnp.asarray([0.0]),
        }

    def _init_conds_jax_spsolve(self, params):
        return {"axial_conductances": jnp.asarray([])}


class CompartmentView(View):
    """CompartmentView."""

    def __init__(self, pointer: Module, view: pd.DataFrame):
        view = view.assign(controlled_by_param=view.global_comp_index)
        super().__init__(pointer, view)

    def __call__(self, index: int):
        if not hasattr(self, "_has_been_called"):
            view = super().adjust_view("comp_index", index)
            view._has_been_called = True
            return view
        raise AttributeError(
            "'CompartmentView' object has no attribute 'comp' or 'loc'."
        )

    def loc(self, loc: float) -> "CompartmentView":
        if loc != "all":
            assert (
                loc >= 0.0 and loc <= 1.0
            ), "Compartments must be indexed by a continuous value between 0 and 1."
        index = index_of_loc(0, loc, self.pointer.nseg) if loc != "all" else "all"
        view = self(index)
        view._has_been_called = True
        return view

    def distance(self, endpoint: "CompartmentView") -> float:
        """Return the direct distance between two compartments.

        This does not compute the pathwise distance (which is currently not
        implemented).

        Args:
            endpoint: The compartment to which to compute the distance to.
        """
        start_branch = self.view["global_branch_index"].item()
        start_comp = self.view["comp_index"].item()
        start_xyz = interpolate_xyz(
            loc_of_index(start_comp, self.pointer.nseg), self.pointer.xyzr[start_branch]
        )

        end_branch = endpoint.view["global_branch_index"].item()
        end_comp = endpoint.view["comp_index"].item()
        end_xyz = interpolate_xyz(
            loc_of_index(end_comp, self.pointer.nseg), self.pointer.xyzr[end_branch]
        )

        return np.sqrt(np.sum((start_xyz - end_xyz) ** 2))

    def vis(
        self,
        ax: Optional[Axes] = None,
        col: str = "k",
        dims: Tuple[int] = (0, 1),
        morph_plot_kwargs: Dict = {},
    ) -> Axes:
        nodes = self.set_global_index_and_index(self.view)
        return self.pointer._scatter(
            ax=ax,
            col=col,
            dims=dims,
            view=nodes,
            morph_plot_kwargs=morph_plot_kwargs,
        )
