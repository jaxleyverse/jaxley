# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


from jax import Array

from jaxley.synapses.synapse import Synapse


class SpikeSynapse(Synapse):
    r"""Synapse to be used in networks of LIF neurons.

    This synapse is meant to be used in networks of LIF neurons, together with
    the `Fire` channel. After the pre-synaptic neuron has `Fire_spikes=1.0` (a spike),
    the state of the synapse gets increased immediately and then decays exponentially.

    This synapse implements the following equations:

    .. math::

        I = \overline{g}\, \cdot s

    .. math::

        \tau \frac{\text{d}s}{\text{d}t} = -s

    .. math::

        s \leftarrow s + 1 \quad \text{if } \, \text{Fire}_{\text{pre}} = 1

    The synaptic parameters are:
        - ``gS``: the maximal conductance :math:`\overline{g}` (uS).
        - ``decay_tau``: The time constant of the decay :math:`\tau` (ms).

    The synaptic state is:
        - ``s``: the activity level of the synapse.

    Example usage
    ^^^^^^^^^^^^^

    .. code-block:: python

        from jaxley.channels import Leak, Fire
        from jaxley.connect import fully_connect
        from jaxley.synapse import SpikeSynapse

        cell = jx.Cell()
        net = jx.Network([cell for _ in range(5)])

        net.insert(Leak())
        net.insert(Fire())
        fully_connect(net.cell("all"), net.cell("all"), SpikeSynapse())

        net.record("v")
        v = jx.integrate(net, t_max=100.0, delta_t=0.025)
    """

    def __init__(self, name: str | None = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gS": 1e-4,  # uS
            f"{prefix}_s_decay": 10.0,  # unitless
        }
        self.synapse_states = {f"{prefix}_s": 0.0}
        self.node_parmas = {}
        self.node_states = {"Fire_spikes": 0.0}

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
    ) -> dict:
        """Return updated synapse state and current."""
        prefix = self._name
        s = synapse_states[f"{prefix}_s"]
        s_decay = synapse_params[f"{prefix}_s_decay"]

        spike = pre_states["Fire_spikes"]
        new_s = (spike * 1.0) + ((1 - spike) * (s - (s_decay * s * delta_t)))

        return {f"{prefix}_s": new_s}

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
        """Return updated synapse state and current."""
        prefix = self._name
        g_syn = synapse_params[f"{prefix}_gS"] * synapse_states[f"{prefix}_s"]
        return g_syn
