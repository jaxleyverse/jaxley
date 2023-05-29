from typing import Dict, List, Optional, Callable
import jax.numpy as jnp
import pandas as pd

from neurax.modules.base import Module, View
from neurax.channels import Channel  # , ChannelView
from neurax.stimulus import get_external_input


class Compartment(Module):
    compartment_params: Dict = {
        "length": 10.0,
        "radius": 1.0,
        "axial_resistivity": 5_000.0,
    }
    compartment_states: Dict = {"voltages": -70.0}

    def __init__(self, channels: List[Channel]):
        super().__init__()
        self._init_params_and_state(self.compartment_params, self.compartment_states)

        # Insert channel parameters.
        for channel in channels:
            for key in channel.channel_params:
                self.params[key] = jnp.asarray(
                    [channel.channel_params[key]]
                )  # should be atleast1d

        # Insert channel states.
        for channel in channels:
            for key in channel.channel_states:
                self.states[key] = jnp.asarray(
                    [channel.channel_states[key]]
                )  # should be atleast1d

        # Indexing.
        self.nodes = pd.DataFrame(
            dict(comp_index=[0], branch_index=[0], cell_index=[0])
        )
        self.channels = channels
        self.initialized_morph = True
        self.initialized_conds = True

        self.nseg = 1
        self.nbranches = 1

    def set_params(self, key, val):
        self.params[key][:] = val

    def __getattr__(self, key):
        assert key == "cell"
        return ChannelView(self, self.nodes)

    def step(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        i_inds: jnp.ndarray,
        i_current: jnp.ndarray,
    ):
        """Step for a single compartment.

        Args:
            u: The full state vector, including states of all channels and the voltage.
            dt: Time step.

        Returns:
            Next state. Same shape as `u`.
        """
        voltages = u["voltages"]

        # Parameters have to go in here.
        new_channel_states, (v_terms, const_terms) = self.step_channels(
            u, dt, self.channels, self.params
        )

        # External input.
        i_ext = get_external_input(
            voltages, i_inds, i_current, self.params["radius"], self.params["length"]
        )

        nbranches = 1
        nseg_per_branch = 1
        new_voltages = self.step_voltages(
            voltages=jnp.reshape(voltages, (nbranches, nseg_per_branch)),
            voltage_terms=jnp.reshape(v_terms, (nbranches, nseg_per_branch)),
            constant_terms=jnp.reshape(const_terms, (nbranches, nseg_per_branch))
            + i_ext,
            coupling_conds_bwd=jnp.asarray([[]]),
            coupling_conds_fwd=jnp.asarray([[]]),
            summed_coupling_conds=jnp.asarray([[0.0]]),
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


class CompartmentView(View):
    def __init__(self, pointer, view):
        super().__init__(pointer, view)

    def __call__(self, index):
        return super().adjust_view("comp_index", index)

    def __getattr__(self, key):
        assert key == "channel"
        return ChannelView(self.pointer, self.view)


class ChannelView(View):
    def __init__(self, pointer, view):
        super().__init__(pointer, view)

    def __call__(self, index):
        return super().adjust_view("channel_index", index)
