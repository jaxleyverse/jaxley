from typing import Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from jaxley.modules.base import Module, View
from jaxley.utils.cell_utils import index_of_loc, interpolate_xyz, loc_of_index


class Compartment(Module):
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

        # Initialize the module.
        self.initialize()
        self.init_syns(None)
        self.initialized_conds = True

        # Coordinates.
        self.xyzr = [float("NaN") * np.zeros((2, 4))]

    def init_conds(self, params):
        cond_params = {
            "branch_conds_fwd": jnp.asarray([]),
            "branch_conds_bwd": jnp.asarray([]),
            "coupling_conds_fwd": jnp.asarray([[]]),
            "coupling_conds_bwd": jnp.asarray([[]]),
            "summed_coupling_conds": jnp.asarray([[0.0]]),
        }
        return cond_params


class CompartmentView(View):
    """CompartmentView."""

    def __init__(self, pointer, view):
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

    def loc(self, loc: float):
        if loc != "all":
            assert (
                loc >= 0.0 and loc <= 1.0
            ), "Compartments must be indexed by a continuous value between 0 and 1."
        index = index_of_loc(0, loc, self.pointer.nseg) if loc != "all" else "all"
        view = self(index)
        view._has_been_called = True
        return view

    def distance(self, endpoint: "CompartmentView"):
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
    ):
        nodes = self.set_global_index_and_index(self.view)
        return self.pointer._scatter(
            ax=ax,
            col=col,
            dims=dims,
            view=nodes,
            morph_plot_kwargs=morph_plot_kwargs,
        )
