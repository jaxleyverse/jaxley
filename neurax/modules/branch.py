from typing import Dict, List, Optional, Callable

import jax.numpy as jnp
import pandas as pd
import numpy as np

from neurax.modules.base import Module, View
from neurax.modules.compartment import Compartment, CompartmentView


class Branch(Module):
    branch_params: Dict = {}
    branch_states: Dict = {}

    def __init__(self, compartments: List[Compartment]):
        self.compartments = compartments
        self.nseg = len(compartments)

        self.coupling_conds_fwd = None

        # Indexing.
        self.nodes = pd.DataFrame(
            dict(
                comp_index=np.arange(self.nseg).tolist(),
                branch_index=[0] * self.nseg,
                cell_index=[0] * self.nseg,
            )
        )

        # Parameters.
        self.params = {}
        for key in self.branch_params:
            self.params[key] = jnp.asarray(self.branch_params[key])
        for key in compartments[0].params:
            param_vals = jnp.asarray([c.params[key] for c in compartments])
            self.params[key] = param_vals

        # States.
        self.states = {}
        for key in self.branch_states:
            self.states[key] = jnp.asarray(self.branch_states[key])
        for key in compartments[0].states:
            states_vals = jnp.asarray([c.states[key] for c in compartments])
            self.states[key] = states_vals

    def set_params(self, key, val):
        self.params[key][:] = val

    def __getattr__(self, key):
        assert key == "cell"
        return CompartmentView(self, self.nodes)

    def init_branch_conds(self):
        """Given an axial resisitivity, set the coupling conductances."""
        axial_resistivity = self.params["axial_resistivity"]
        radiuses = self.params["radius"]
        lengths = self.params["length"]

        def compute_coupling_cond(rad1, rad2, r_a, l1, l2):
            return rad1 * rad2 ** 2 / r_a / (rad2 ** 2 * l1 + rad1 ** 2 * l2) / l1

        # Compute coupling conductance for segments within a branch.
        # `radius`: um
        # `r_a`: ohm cm
        # `length_single_compartment`: um
        # `coupling_conds`: S * um / cm / um^2 = S / cm / um
        rad1 = radiuses[1:]
        rad2 = radiuses[:-1]
        l1 = lengths[1:]
        l2 = lengths[:-1]
        r_a1 = axial_resistivity[1:]
        r_a2 = axial_resistivity[:-1]
        self.coupling_conds_bwd = compute_coupling_cond(rad2, rad1, r_a1, l2, l1)
        self.coupling_conds_fwd = compute_coupling_cond(rad1, rad2, r_a2, l1, l2)

        # Convert (S / cm / um) -> (mS / cm^2)
        self.coupling_conds_fwd *= 10 ** 7
        self.coupling_conds_bwd *= 10 ** 7

        # Compute the summed coupling conductances of each compartment.
        self.summed_coupling_conds = jnp.zeros((self.nseg))
        self.summed_coupling_conds = self.summed_coupling_conds.at[1:].add(
            self.coupling_conds_fwd
        )
        self.summed_coupling_conds = self.summed_coupling_conds.at[:-1].add(
            self.coupling_conds_bwd
        )

    def step(self, u: Dict[str, jnp.ndarray], dt: float, i_ext=0):
        """Step for a single compartment.

        Args:
            u: The full state vector, including states of all channels and the voltage.
            dt: Time step.

        Returns:
            Next state. Same shape as `u`.
        """
        if self.coupling_conds_fwd is None:
            self.init_branch_conds()

        voltages = u["voltages"]

        # Parameters have to go in here.
        new_channel_states, (v_terms, const_terms) = self.step_channels(
            u, dt, self.compartments[0].channels, self.params
        )

        nbranches = 1
        nseg_per_branch = self.nseg
        new_voltages = self.step_voltages(
            jnp.reshape(voltages, (nbranches, nseg_per_branch)),
            jnp.reshape(v_terms, (nbranches, nseg_per_branch)),
            jnp.reshape(const_terms, (nbranches, nseg_per_branch)) + i_ext,
            coupling_conds_bwd=jnp.reshape(self.coupling_conds_bwd, (1, self.nseg - 1)),
            coupling_conds_fwd=jnp.reshape(self.coupling_conds_fwd, (1, self.nseg - 1)),
            summed_coupling_conds=jnp.reshape(
                self.summed_coupling_conds, (1, self.nseg)
            ),
            branch_cond_fwd=jnp.asarray([[]]),
            branch_cond_bwd=jnp.asarray([[]]),
            num_branches=1,
            parents=jnp.asarray([-1]),
            kid_inds_in_each_level=jnp.asarray([0]),
            max_num_kids=1,
            parents_in_each_level=[jnp.asarray([-1,])],
            branches_in_each_level=[jnp.asarray([0,])],
            tridiag_solver="thomas",
            delta_t=dt,
        )
        final_state = new_channel_states[0]
        final_state["voltages"] = new_voltages[0]
        return final_state


class BranchView(View):
    def __init__(self, pointer, view):
        super().__init__(pointer, view)

    def __call__(self, index):
        return super().adjust_view("branch_index", index)

    def __getattr__(self, key):
        assert key == "comp"
        return CompartmentView(self.pointer, self.view)
