# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


from jax import Array

from jaxley.synapses.synapse import Synapse


class ExpDecaySynapse(Synapse):
    r"""Synapse which decays exponentially after a pre-synaptic binary spike.

    This synapse requires that the pre-synaptic neuron has a binary voltage of either
    zero or one. After a pre-synaptic one (a spike), the state of the synapse gets
    increased immediately and then decays exponentially.

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

        import jaxley as jx
        from jaxley.connect import connect
        from jaxley.synapses import ExpDecaySynapse
        from jaxley.channels import Leak

        dummy = jx.Cell()
        cell = jx.read_swc("morph_ca1_n120.swc", ncomp=1)
        net = jx.Network([dummy, cell])
        net.cell(1).insert(Leak())

        # Connect pre-synaptic dummy to the morphologically detailed cell.
        connect(net.cell(0), net.cell(1).branch(5).comp(0), ExpDecaySynapse())
        net.set("ExpDecaySynapse_gS", 0.1)  # Synaptic strength.
        net.set("ExpDecaySynapse_decay_tau", 5.0)  # decay time in ms

        # Clamp the voltage of the pre-synaptic cell to the spike train.
        net.cell(0).set("v", 0.0)  # Initial state.
        net.cell(0).clamp("v", spike_train)

        net.cell(1).branch(5).comp(0).record()
        v = jx.integrate(net, delta_t=dt)
    """

    def __init__(self, name: str | None = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gS": 1e-4,  # uS
            f"{prefix}_decay_tau": 10.0,  # ms
        }
        self.synapse_states = {f"{prefix}_s": 0.0}

    def update_states(
        self,
        states: dict[str, Array],
        all_states: dict,
        pre_index: Array,
        post_index: Array,
        params: dict[str, Array],
        delta_t: float,
    ) -> dict:
        """Return updated synapse state and current."""
        prefix = self._name
        s = states[f"{prefix}_s"]
        tau = params[f"{prefix}_decay_tau"]

        spike = all_states["v"][pre_index]
        term1 = spike * 1.0
        term2 = (1 - spike) * (s - (s * delta_t / tau))
        new_s = term1 + term2

        return {f"{prefix}_s": new_s}

    def compute_current(
        self, states: dict, pre_voltage: float, post_voltage: float, params: dict
    ) -> float:
        """Return updated synapse state and current."""
        prefix = self._name
        g_syn = -params[f"{prefix}_gS"] * states[f"{prefix}_s"]
        return g_syn
