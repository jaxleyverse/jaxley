# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional, Tuple

import jax.numpy as jnp
from jax import Array

from jaxley.solver_gate import save_exp
from jaxley.synapses.synapse import Synapse


class TestSynapse(Synapse):
    """
    Compute syanptic current and update synapse state for a test synapse.
    """

    __test__ = False  # Not a unit test - pytest ignores

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {f"{prefix}_gC": 1e-4}
        self.synapse_states = {f"{prefix}_c": 0.2}
        self.node_params = {}
        self.node_states = {}

    def update_states(
        self,
        synapse_states: dict[str, Array],
        synapse_params: dict[str, Array],
        pre_voltage: Array,
        post_voltage: Array,
        pre_states: dict[str, Array],
        post_states: dict[str, Array],
        pre_params: dict[str, Array],
        post_params: dict[str, Array],
        delta_t: float,
    ) -> Dict:
        """Return updated synapse state and current."""
        prefix = self._name
        v_th = -35.0
        delta = 10.0
        k_minus = 1.0 / 40.0

        s_bar = 1.0 / (1.0 + save_exp((v_th - pre_voltage) / delta))
        tau_s = (1.0 - s_bar) / k_minus

        s_inf = s_bar
        slope = -1.0 / tau_s
        exp_term = save_exp(slope * delta_t)
        new_s = synapse_states[f"{prefix}_c"] * exp_term + s_inf * (1.0 - exp_term)
        return {f"{prefix}_c": new_s}

    def compute_current(
        self,
        synapse_states: dict[str, Array],
        synapse_params: dict[str, Array],
        pre_voltage: Array,
        post_voltage: Array,
        pre_states: dict[str, Array],
        post_states: dict[str, Array],
        pre_params: dict[str, Array],
        post_params: dict[str, Array],
        delta_t: float,
    ) -> float:
        prefix = self._name
        e_syn = 0.0
        g_syn = synapse_params[f"{prefix}_gC"] * synapse_states[f"{prefix}_c"]
        return g_syn * (post_voltage - e_syn)
