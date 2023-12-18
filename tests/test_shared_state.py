import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np

import jaxley as jx
from jaxley.channels import HH, Na, K


import jax.numpy as jnp
from jaxley.channels import Channel


class Dummy1(Channel):
    """A dummy channel which simply accumulates a state (same state as dummy2)."""

    channel_params = {}
    channel_states = {"Dummy_s": 0.0}

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

    channel_params = {}
    channel_states = {"Dummy_s": 0.0}

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
