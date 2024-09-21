# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import jax.numpy as jnp
import numpy as np
import pandas as pd

from jaxley.modules.base import GroupView, Module, View
from jaxley.modules.compartment import Compartment, CompartmentView
from jaxley.utils.cell_utils import compute_children_and_parents
from jaxley.utils.solver_utils import comp_edges_to_indices


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
        self.nseg_per_branch = [self.nseg]
        self.total_nbranches = 1
        self.nbranches_per_cell = [1]
        self.cumsum_nbranches = jnp.asarray([0, 1])

        # Indexing.
        self.nodes = pd.concat([c.nodes for c in compartment_list], ignore_index=True)
        self._append_params_and_states(self.branch_params, self.branch_states)
        self.nodes["comp_index"] = np.arange(self.nseg).tolist()
        self.nodes["branch_index"] = [0] * self.nseg
        self.nodes["cell_index"] = [0] * self.nseg

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

    def __getattr__(self, key: str):
        # Ensure that hidden methods such as `__deepcopy__` still work.
        if key.startswith("__"):
            return super().__getattribute__(key)

        if key in ["comp", "loc"]:
            view = deepcopy(self.nodes)
            view["global_comp_index"] = view["comp_index"]
            view["global_branch_index"] = view["branch_index"]
            view["global_cell_index"] = view["cell_index"]
            compview = CompartmentView(self, view)
            return compview if key == "comp" else compview.loc
        elif key in self.group_nodes:
            inds = self.group_nodes[key].index.values
            view = self.nodes.loc[inds]
            view["global_comp_index"] = view["comp_index"]
            view["global_branch_index"] = view["branch_index"]
            view["global_cell_index"] = view["cell_index"]
            return GroupView(self, view, CompartmentView, ["comp", "loc"])
        else:
            raise KeyError(f"Key {key} not recognized.")

    def set_ncomp(self, ncomp: int):
        """Set the number of compartments with which the branch is discretized."""
        radius_generating_functions = lambda x: 0.5
        within_branch_radiuses = self.nodes["radius"]

        if ~np.all_equal(within_branch_radiuses) and radius_generating_functions is None:
            warn(
                f"You previously modified the radius of individual compartments, but now"
                f"you are modifying the number of compartments in this branch. We are"
                f"resetting every radius in this branch to 1um. To avoid this, first"
                f"set the number of compartments in every branch and then modify their radius."
            )
            within_branch_radiuses = 1.0 * np.ones_like(within_branch_radiuses)

        if ~np.all_equal(within_branch_lengths):
            warn(
                f"You previously modified the length of individual compartments, but now"
                f"you are modifying the number of compartments in this branch. We are"
                f"now assuming that the lenght of every compartment in this branch is equal,"
                f"such that the branch has the same length as with the old number of compartments."
                f"To avoid this, first set the number of compartments in every branch and then modify their radius."
            )

        # Compute new compartment lengths.
        comp_lengths = np.sum(compartment_lengths) / ncomp
        
        # Compute new compartment radiuses.
        if radius_generating_functions is not None:
            comp_radiuses = radius_generating_functions(np.linspace(0, 1, ncomps))
        else:
            comp_radiuses = within_branch_radiuses

        # Add new row as the average of all rows.
        df = self.nodes
        average_row = df.mean(skipna=False)
        average_row = average_row.to_frame().T
        df = pd.concat([df, average_row], axis="rows")

        # Set the correct datatype after having performed an average which cast
        # everything to float.
        integer_cols = ["comp_index", "branch_index", "cell_index"]
        df[integer_cols] = df[integer_cols].astype(int)

        # Update the comp_index, branch_index, cell_index.
        # TODO.

        # Special treatment for channels. Channels will only be added to the new nseg
        # if **all** other segments in the branch also had that channel.
        channel_cols = ["HH"]
        df[channel_cols] = np.floor(df[channel_cols]).astype(bool)

        # Special treatment for the lengths.
        df["length"] = comp_lengths

        # Special treatment for the radiuses.
        df["radius"] = comp_radiuses

        
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


class BranchView(View):
    """BranchView."""

    def __init__(self, pointer: Module, view: pd.DataFrame):
        view = view.assign(controlled_by_param=view.global_branch_index)
        super().__init__(pointer, view)

    def __call__(self, index: float):
        local_idcs = self._get_local_indices()
        self.view[local_idcs.columns] = (
            local_idcs  # set indexes locally. enables net[0:2,0:2]
        )
        self.allow_make_trainable = True
        new_view = super().adjust_view("branch_index", index)
        return new_view

    def __getattr__(self, key):
        assert key in ["comp", "loc"]
        compview = CompartmentView(self.pointer, self.view)
        return compview if key == "comp" else compview.loc
