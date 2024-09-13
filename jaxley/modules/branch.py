# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd

from jaxley.modules.base import GroupView, Module, View
from jaxley.modules.compartment import Compartment, CompartmentView
from jaxley.utils.cell_utils import compute_coupling_cond


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
        # Explanation of `type`:
        # `type == 0`: compartment-to-compartment (within branch)
        # `type == 1`: compartment-to-branchpoint
        # `type == 2`: branchpoint-to-compartment
        self.comp_edges = pd.DataFrame().from_dict({
            "source": list(range(self.nseg - 1)) + list(range(1, self.nseg)),
            "sink": list(range(1, self.nseg)) + list(range(self.nseg - 1)),
            "type": [0] * (self.nseg - 1) * 2,
        })

        # For morphology indexing.
        self.child_inds = np.asarray([]).astype(int)
        self.child_belongs_to_branchpoint = np.asarray([]).astype(int)
        self.par_inds = np.asarray([]).astype(int)
        self.total_nbranchpoints = 0
        self.branchpoint_group_inds = np.asarray([]).astype(int)

        self.children_in_level = []
        self.parents_in_level = []
        self.root_inds = jnp.asarray([0])

        self.initialize()
        self.init_syns()
        self.initialized_conds = False

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

    def init_conds(self, params: Dict) -> Dict[str, jnp.ndarray]:
        conds = self.init_branch_conds(
            params["axial_resistivity"], params["radius"], params["length"], self.nseg
        )
        cond_params = {
            "branchpoint_conds_children": jnp.asarray([]),
            "branchpoint_conds_parents": jnp.asarray([]),
            "branchpoint_weights_children": jnp.asarray([]),
            "branchpoint_weights_parents": jnp.asarray([]),
        }
        cond_params["branch_lowers"] = conds[0]
        cond_params["branch_uppers"] = conds[1]
        cond_params["branch_diags"] = conds[2]

        return cond_params

    @staticmethod
    def init_branch_conds(
        axial_resistivity: jnp.ndarray,
        radiuses: jnp.ndarray,
        lengths: jnp.ndarray,
        nseg: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Given an axial resisitivity, set the coupling conductances.

        Args:
            axial_resistivity: Axial resistivity of each compartment.
            radiuses: Radius of each compartment.
            lengths: Length of each compartment.
            nseg: Number of compartments in the branch.

        Returns:
            Tuple of forward coupling conductances, backward coupling conductances, and summed coupling conductances.
        """

        # Compute coupling conductance for segments within a branch.
        # `radius`: um
        # `r_a`: ohm cm
        # `length_single_compartment`: um
        # `coupling_conds`: S * um / cm / um^2 = S / cm / um
        r1 = radiuses[:-1]
        r2 = radiuses[1:]
        r_a1 = axial_resistivity[:-1]
        r_a2 = axial_resistivity[1:]
        l1 = lengths[:-1]
        l2 = lengths[1:]
        coupling_conds_bwd = compute_coupling_cond(r1, r2, r_a1, r_a2, l1, l2)
        coupling_conds_fwd = compute_coupling_cond(r2, r1, r_a2, r_a1, l2, l1)

        # Compute the summed coupling conductances of each compartment.
        summed_coupling_conds = jnp.zeros((nseg))
        summed_coupling_conds = summed_coupling_conds.at[1:].add(coupling_conds_fwd)
        summed_coupling_conds = summed_coupling_conds.at[:-1].add(coupling_conds_bwd)
        return coupling_conds_fwd, coupling_conds_bwd, summed_coupling_conds

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
