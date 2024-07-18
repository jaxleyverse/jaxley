# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np
from jaxley_mech.channels.l5pc import CaHVA, CaPump

import jaxley as jx
from jaxley.channels import HH, Channel, K, Na


class Dummy1(Channel):
    """A dummy channel which simply accumulates a state (same state as dummy2)."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {}
        self.channel_states = {"Dummy_s": 0.0}
        self.current_name = f"i_Dummy1"

    @staticmethod
    def update_states(u, dt, voltages, params):
        """Update state."""
        dummy_state = u["Dummy_s"]
        print("dummy_state1", dummy_state)
        return {"Dummy_s": dummy_state + 0.01}

    @staticmethod
    def compute_current(u, voltages, params):
        """Return current."""
        return jnp.zeros_like(voltages)


class Dummy2(Channel):
    """A dummy channel which simply accumulates a state (same state as dummy1)."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {}
        self.channel_states = {"Dummy_s": 0.0}
        self.current_name = f"i_Dummy2"

    @staticmethod
    def update_states(u, dt, voltages, params):
        """Update state."""
        dummy_state = u["Dummy_s"]
        print("dummy_state2", dummy_state)
        return {"Dummy_s": dummy_state + 0.01}

    @staticmethod
    def compute_current(u, voltages, params):
        """Return current."""
        return jnp.zeros_like(voltages)


def test_shared_state():
    """Test whether two channels can share a state.

    This has to be copied into a notebook and executed with `jax.disable_jit():`."""
    comp1 = jx.Compartment()
    comp1.insert(Dummy1())

    comp2 = jx.Compartment()
    comp2.insert(Dummy2())

    comp3 = jx.Compartment()
    comp3.insert(Dummy1())
    comp3.insert(Dummy2())

    voltages = []
    for comp in [comp1, comp2, comp3]:
        comp.record()
        current = jx.step_current(0.1, 0.1, 0.1, 0.025, 0.3)
        comp.stimulate(current)

        voltages.append(jx.integrate(comp))


def test_current_as_state_multicompartment():
    """#323 had discovered a bug when currents are only used in a few compartments."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, 2)

    branch.comp(0).insert(CaHVA())  # defines current `i_Ca`
    branch.comp(0).insert(CaPump())  # uses `states["i_Ca"]`

    branch.comp(0).record()
    _ = jx.integrate(branch, t_max=1.0)
