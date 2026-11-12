# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
from typing import Optional

import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxley_mech.channels.l5pc import CaHVA

import jaxley as jx
from jaxley.channels import Channel
from tests.mechanisms import CaPump


class Dummy1(Channel):
    """A dummy channel which simply accumulates a state (same state as dummy2)."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {}
        self.channel_states = {"Dummy_s": 0.0}
        self.current_name = f"i_Dummy1"

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Update state."""
        dummy_state = states["Dummy_s"]
        return {"Dummy_s": dummy_state + 0.01}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Return current."""
        return jnp.zeros_like(voltage)


class Dummy2(Channel):
    """A dummy channel which simply accumulates a state (same state as dummy1)."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {}
        self.channel_states = {"Dummy_s": 0.0}
        self.current_name = f"i_Dummy2"

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Update state."""
        dummy_state = states["Dummy_s"]
        return {"Dummy_s": dummy_state + 0.01}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Return current."""
        return jnp.zeros_like(voltage)


def test_shared_state():
    """Test whether two channels can share a state.

    Each channel adds 0.01 to `Dummy_s` per step, so with both channels the state
    grows twice as fast."""
    comp1 = jx.Compartment()
    comp1.insert(Dummy1())

    comp2 = jx.Compartment()
    comp2.insert(Dummy2())

    comp3 = jx.Compartment()
    comp3.insert(Dummy1())
    comp3.insert(Dummy2())

    states = []
    for comp in [comp1, comp2, comp3]:
        comp.record("Dummy_s")
        current = jx.step_current(
            i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
        )
        comp.stimulate(current)

        states.append(jx.integrate(comp)[0])

    num_steps = len(states[0]) - 1
    np.testing.assert_allclose(states[0], 0.01 * np.arange(num_steps + 1))
    np.testing.assert_allclose(states[1], states[0])
    np.testing.assert_allclose(states[2], 2 * states[0])


def test_current_as_state_multicompartment():
    """#323 had discovered a bug when currents are only used in a few compartments."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, 2)

    branch.comp(0).insert(CaHVA())  # defines current `i_Ca`
    branch.comp(0).insert(CaPump())  # uses `states["i_Ca"]`

    branch.comp(0).record()
    branch.comp(0).record("CaCon_i")
    v, cacon_i = jx.integrate(branch, t_max=1.0)
    assert np.invert(np.any(np.isnan(v)))
    # Calcium flows in through `CaHVA`, and the pump sees that current.
    assert cacon_i[-1] > cacon_i[0]
