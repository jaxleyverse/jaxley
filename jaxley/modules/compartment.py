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
from jaxley.utils.solver_utils import comp_edges_to_indices


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
        self.nseg_per_branch = jnp.asarray([1])
        self.total_nbranches = 1
        self.nbranches_per_cell = [1]
        self.cumsum_nbranches = jnp.asarray([0, 1])
        self.cumsum_nseg = cumsum_leading_zero(self.nseg_per_branch)

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

    def _init_morph_jaxley_spsolve(self):
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

    def init_conds(self, params: Dict[str, jnp.ndarray]):
        """Override `Base.init_axial_conds()`.

        This is because compartments do not have any axial conductances."""
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

        branch_ind = np.unique(self.view["global_branch_index"].to_numpy())
        if loc != "all" and len(branch_ind) != 1:
            raise NotImplementedError(
                "Using `.loc()` to index a single compartment of multiple branches is "
                "not supported. Use a for loop or use `.comp` to index."
            )
        branch_ind = np.squeeze(branch_ind)  # shape == (1,) --> shape == ()

        # Cast nseg to numpy because in `local_index_of_loc` we instatiate an array
        # of length `nseg`. However, if we use `.data_set()` or `.data_stimulate()`,
        # the `local_index_of_loc()` method must be compatible with `jit`. Therefore,
        # we have to stop this from being traced here and cast to numpy.
        nsegs = np.asarray(self.pointer.nseg_per_branch)
        index = local_index_of_loc(loc, branch_ind, nsegs) if loc != "all" else "all"
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
        start_comp = self.view["global_comp_index"].item()
        start_xyz = interpolate_xyz(
            loc_of_index(
                start_comp,
                start_branch,
                self.pointer.nseg_per_branch,
            ),
            self.pointer.xyzr[start_branch],
        )

        end_branch = endpoint.view["global_branch_index"].item()
        end_comp = endpoint.view["global_comp_index"].item()
        end_xyz = interpolate_xyz(
            loc_of_index(
                end_comp,
                end_branch,
                self.pointer.nseg_per_branch,
            ),
            self.pointer.xyzr[end_branch],
        )

        return np.sqrt(np.sum((start_xyz - end_xyz) ** 2))

    def vis(
        self,
        ax: Optional[Axes] = None,
        col: str = "k",
        type: str = "scatter",
        dims: Tuple[int] = (0, 1),
        morph_plot_kwargs: Dict = {},
    ) -> Axes:
        """Visualize the compartment.

        Args:
            ax: An axis into which to plot.
            col: The color for all branches.
            type: Whether to plot as point ("scatter") or the projected volume ("volume").
            dims: Which dimensions to plot. 1=x, 2=y, 3=z coordinate. Must be a tuple of
                two of them.
            morph_plot_kwargs: Keyword arguments passed to the plotting function.
        """
        nodes = self.set_global_index_and_index(self.view)
        if type == "volume":
            return self.pointer._vis(
                ax=ax,
                col=col,
                dims=dims,
                view=nodes,
                type="volume",
                morph_plot_kwargs=morph_plot_kwargs,
            )

        return self.pointer._scatter(
            ax=ax,
            col=col,
            dims=dims,
            view=nodes,
            morph_plot_kwargs=morph_plot_kwargs,
        )
