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
        view = view.assign(controlled_by_param=view.comp_index)
        super().__init__(pointer, view)

    def __call__(self, loc: float):
        if loc != "all":
            assert (
                loc >= 0.0 and loc <= 1.0
            ), "Compartments must be indexed by a continuous value between 0 and 1."

        index = index_of_loc(0, loc, self.pointer.nseg) if loc != "all" else "all"
        return super().adjust_view("comp_index", index)

    def connect(self, post: "CompartmentView", synapse_type):
        """Connect two compartments with a chemical synapse.

        High-level strategy:

        We need to first check if the network already has a type of this synapse, else
        we need to register it as a new synapse in a bunch of dictionaries which track
        synapse parameters, state and meta information.

        Next, we register the new connection in the synapse dataframe (`.edges`).
        Then, we update synapse parameter and state arrays with the new connection.
        Finally, we update synapse meta information.
        """
        synapse_name = type(synapse_type).__name__
        is_new_type = True if synapse_name not in self.pointer.synapse_names else False

        if is_new_type:
            # New type: index for the synapse type is one more than the currently
            # highest index.
            max_ind = self.pointer.edges["type_ind"].max() + 1
            type_ind = 0 if jnp.isnan(max_ind) else max_ind
        else:
            # Not a new type: search for the index that this type has previously had.
            type_ind = self.pointer.edges.query(f"type == '{synapse_name}'")[
                "type_ind"
            ].to_numpy()[0]

        # The `edges` dataframe expects the compartment as continuous `loc`, not
        # as discrete compartment index (because the continuous `loc` is used for
        # plotting). Below, we cast the compartment index to its (rough) location.
        pre_comp = loc_of_index(
            self.view["global_comp_index"].to_numpy(), self.pointer.nseg
        )
        post_comp = loc_of_index(
            post.view["global_comp_index"].to_numpy(), self.pointer.nseg
        )
        index = len(self.pointer.edges)

        # Update edges.
        self.pointer.edges = pd.concat(
            [
                self.pointer.edges,
                pd.DataFrame(
                    dict(
                        pre_locs=pre_comp,
                        post_locs=post_comp,
                        pre_branch_index=self.view["branch_index"].to_numpy(),
                        post_branch_index=post.view["branch_index"].to_numpy(),
                        pre_cell_index=self.view["cell_index"].to_numpy(),
                        post_cell_index=post.view["cell_index"].to_numpy(),
                        type=synapse_name,
                        type_ind=type_ind,
                        global_pre_comp_index=self.view["global_comp_index"].to_numpy(),
                        global_post_comp_index=post.view[
                            "global_comp_index"
                        ].to_numpy(),
                        global_pre_branch_index=self.view[
                            "global_branch_index"
                        ].to_numpy(),
                        global_post_branch_index=post.view[
                            "global_branch_index"
                        ].to_numpy(),
                    )
                ),
            ],
            ignore_index=True,
        )

        # Add parameters and states to the `.edges` table.
        indices = list(range(index, index + 1))
        for key in synapse_type.synapse_params:
            param_val = synapse_type.synapse_params[key]
            self.pointer.edges.loc[indices, key] = param_val

        # Update synaptic state array.
        for key in synapse_type.synapse_states:
            state_val = synapse_type.synapse_states[key]
            self.pointer.edges.loc[indices, key] = state_val

        # (Potentially) update variables that track meta information about synapses.
        if is_new_type:
            self.pointer.synapse_names.append(type(synapse_type).__name__)
            self.pointer.synapse_param_names += list(synapse_type.synapse_params.keys())
            self.pointer.synapse_state_names += list(synapse_type.synapse_states.keys())
            self.pointer.synapses.append(synapse_type)

    def distance(self, endpoint: Union[Compartment, "CompartmentView"]):
        """Return the direct distance between two compartments.

        This does not compute the pathwise distance (which is currently not
        implemented).

        Args:
            endpoint: The compartment to which to compute the distance to.
        """
        start_branch = self.view["branch_index"].item()
        start_comp = self.view["comp_index"].item()
        start_xyz = interpolate_xyz(
            loc_of_index(start_comp, self.pointer.nseg), self.pointer.xyzr[start_branch]
        )

        end_branch = endpoint.view["branch_index"].item()
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
        return self.pointer._scatter(
            ax=ax,
            col=col,
            dims=dims,
            view=self.view,
            morph_plot_kwargs=morph_plot_kwargs,
        )
