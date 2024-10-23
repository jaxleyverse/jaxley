from typing import Dict, Optional

import jax.numpy as jnp

import jaxley as jx
from jaxley.channels import Channel, Leak


class LeakOldConvention(Channel):
    """Leak current"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = False

        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gLeak": 1e-4,
            f"{prefix}_eLeak": -70.0,
        }
        self.channel_states = {}
        self.current_name = f"i_{prefix}"

    def update_states(self, states, dt, v, params):
        """No state to update."""
        return {}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        gLeak = params[f"{prefix}_gLeak"]  # S/cm^2
        return gLeak * (v - params[f"{prefix}_eLeak"]) * 1000.0  # mA/cm^2 -> uA/cm^2.

    def init_state(self, states, v, params, delta_t):
        return {}


def test_same_result_for_both_current_units():
    """Test whether two channels (with old and new unit convention) match."""
    current = jx.step_current(1.0, 2.0, 0.01, 0.025, 5.0)
    comp1 = jx.Compartment()
    comp2 = jx.Compartment()

    comp1.insert(LeakOldConvention())
    comp2.insert(Leak())

    comp1.record()
    comp1.stimulate(current)
    comp2.record()
    comp2.stimulate(current)

    v1 = jx.integrate(comp1)
    v2 = jx.integrate(comp2)

    assert jnp.allclose(v1, v2)
