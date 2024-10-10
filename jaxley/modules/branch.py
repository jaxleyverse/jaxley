# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from copy import deepcopy
from itertools import chain
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd

from jaxley.modules.base import GroupView, Module, View
from jaxley.modules.compartment import Compartment, CompartmentView
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

    def set_ncomp(self, ncomp: int, min_radius: Optional[float] = None):
        """Set the number of compartments with which the branch is discretized.

        Args:
            ncomp: The number of compartments that the branch should be discretized
                into.

        Raises:
            - When the Module is a Network.
            - When there are stimuli in any compartment in the Module.
            - When there are recordings in any compartment in the Module.
            - When the channels of the compartments are not the same within the branch
            that is modified.
            - When the lengths of the compartments are not the same within the branch
            that is modified.
            - Unless the morphology was read from an SWC file, when the radiuses of the
            compartments are not the same within the branch that is modified.
        """
        assert len(self.externals) == 0, "No stimuli allowed!"
        assert len(self.recordings) == 0, "No recordings allowed!"
        assert len(self.trainable_params) == 0, "No trainables allowed!"

        # Update all attributes that are affected by compartment structure.
        (
            self.nodes,
            self.nseg_per_branch,
            self.nseg,
            self.cumsum_nseg,
            self._internal_node_inds,
        ) = self._set_ncomp(
            ncomp,
            self.nodes,
            self.nodes,
            self.nodes["comp_index"].to_numpy()[0],
            self.nseg_per_branch,
            [c._name for c in self.channels],
            list(chain(*[c.channel_params for c in self.channels])),
            list(chain(*[c.channel_states for c in self.channels])),
            self._radius_generating_fns,
            min_radius,
        )

        # Update the morphology indexing (e.g., `.comp_edges`).
        self.initialize()


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

    def set_ncomp(self, ncomp: int, min_radius: Optional[float] = None):
        """Set the number of compartments with which the branch is discretized.

        Args:
            ncomp: The number of compartments that the branch should be discretized
                into.
            min_radius: Only used if the morphology was read from an SWC file. If passed
                the radius is capped to be at least this value.

        Raises:
            - When there are stimuli in any compartment in the module.
            - When there are recordings in any compartment in the module.
            - When the channels of the compartments are not the same within the branch
            that is modified.
            - When the lengths of the compartments are not the same within the branch
            that is modified.
            - Unless the morphology was read from an SWC file, when the radiuses of the
            compartments are not the same within the branch that is modified.
        """
        if self.pointer._module_type == "network":
            raise NotImplementedError(
                "`.set_ncomp` is not yet supported for a `Network`. To overcome this, "
                "first build individual cells with the desired `ncomp` and then "
                "assemble them into a network."
            )

        error_msg = lambda name: (
            f"Your module contains a {name}. This is not allowed. First build the "
            "morphology with `set_ncomp()` and then insert stimuli, recordings, and "
            "define trainables."
        )
        assert len(self.pointer.externals) == 0, error_msg("stimulus")
        assert len(self.pointer.recordings) == 0, error_msg("recording")
        assert len(self.pointer.trainable_params) == 0, error_msg("trainable parameter")
        # Update all attributes that are affected by compartment structure.
        (
            self.pointer.nodes,
            self.pointer.nseg_per_branch,
            self.pointer.nseg,
            self.pointer.cumsum_nseg,
            self.pointer._internal_node_inds,
        ) = self.pointer._set_ncomp(
            ncomp,
            self.view,
            self.pointer.nodes,
            self.view["global_comp_index"].to_numpy()[0],
            self.pointer.nseg_per_branch,
            [c._name for c in self.pointer.channels],
            list(chain(*[c.channel_params for c in self.pointer.channels])),
            list(chain(*[c.channel_states for c in self.pointer.channels])),
            self.pointer._radius_generating_fns,
            min_radius,
        )

        # Update the morphology indexing (e.g., `.comp_edges`).
        self.pointer.initialize()
