# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional, Tuple

import jax.numpy as jnp

from jaxley.synapses.synapse import Synapse


class TanhRateSynapse(Synapse):
    """
    Compute synaptic current for tanh synapse (no state).
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gS": 1e-4,
            f"{prefix}_x_offset": -70.0,
            f"{prefix}_slope": 1.0,
        }
        self.synapse_states = {}

    def update_states(
        self,
        states: Dict,
        delta_t: float,
        pre_voltage: float,
        post_voltage: float,
        params: Dict,
    ) -> Dict:
        """Return updated synapse state and current."""
        return {}

    def compute_current(
        self, states: Dict, pre_voltage: float, post_voltage: float, params: Dict
    ) -> float:
        """Return updated synapse state and current."""
        prefix = self._name
        current = (
            -1
            * params[f"{prefix}_gS"]
            * jnp.tanh(
                (pre_voltage - params[f"{prefix}_x_offset"]) * params[f"{prefix}_slope"]
            )
        )
        return current
