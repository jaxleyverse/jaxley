# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


from jax import Array

from jaxley.solver_gate import exponential_euler
from jaxley.synapses.synapse import Synapse


class AlphaSynapse(Synapse):
    r"""Alpha synapse which responds to binary pre-synaptic spike trains.

    This synapse is meant to be used together with pre-synaptic neurons whose voltage
    is binary (indicating spike or no spike).

    This synapse is implemented as two cascaded first-order linear ODEs:

    .. math::

        \tau_{\mathrm{rise}}\frac{d r(t)}{d t} = -r(t) + x(t)

    .. math::

        \tau_{\mathrm{decay}}\frac{d s(t)}{d t} = -s(t) + r(t)

    .. math::

        I = \overline{g}\, \cdot s \cdot (E - V_{\text{post}})

    Here, :math:`x(t)` denotes the presynaptic input (typically a binary spike train),
    :math:`r(t)` is an intermediate *rise* state, and :math:`s(t)` is the synaptic
    state that determines the synaptic conductance.

    For an impulse input :math:`x(t) = \delta(t)`, the resulting synaptic
    kernel is

    .. math::

        s(t) \propto e^{-t / \tau_{\mathrm{decay}}} - e^{-t / \tau_{\mathrm{rise}}}, \qquad t \ge 0.

    The synaptic parameters are:
        - ``gS``: the maximal conductance :math:`\overline{g}` (uS).
        - ``tau_decay``: The decay time constant :math:`\tau_{\text{rise}}` (ms).
        - ``tau_rise``: The rise time constant :math:`\tau_{\text{decay}}` (ms).
    
    The inserted cellular parameters are:
        - ``e_syn``: The synaptic reversal potential :math:`E` (mV).

    The synaptic state is:
        - ``r``: Intermediate state representing the rising phase.
        - ``s``: Activity level of the synapse.

    Example usage
    ^^^^^^^^^^^^^

    .. code-block:: python

        import jaxley as jx
        from jaxley.connect import connect
        from jaxley.synapses import AlphaSynapse
        from jaxley.channels import Leak

        dummy = jx.Cell()
        cell = jx.read_swc("morph_ca1_n120.swc", ncomp=1)
        net = jx.Network([dummy, cell])
        net.cell(1).insert(Leak())

        # Connect pre-synaptic dummy to the morphologically detailed cell.
        connect(net.cell(0), net.cell(1).branch(5).comp(0), AlphaSynapse())
        net.set("AlphaSynapse_gS", 0.1)  # Synaptic strength.
        net.set("AlphaSynapse_tau_decay", 5.0)  # Decay time in ms

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
            f"{prefix}_tau_rise": 10.0,  # ms
            f"{prefix}_tau_decay": 10.0,  # ms
        }
        self.synapse_states = {
            f"{prefix}_r": 0.0,
            f"{prefix}_s": 0.0,
        }
        self.node_params = {
            f"{prefix}_e_syn": 0.0,  # mV,
        }
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
    ) -> dict:
        """Return updated synapse state and current."""
        prefix = self._name
        r = synapse_states[f"{prefix}_r"]
        s = synapse_states[f"{prefix}_s"]
        tau_rise = synapse_params[f"{prefix}_tau_rise"]
        tau_decay = synapse_params[f"{prefix}_tau_decay"]

        r_inf = 1 / delta_t * pre_voltage
        r_new = exponential_euler(r, delta_t, r_inf, tau_rise)
        s_new = exponential_euler(s, delta_t, r, tau_decay)

        return {f"{prefix}_s": s_new, f"{prefix}_r": r_new}

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
        return g_syn * (post_voltage - post_params[f"{prefix}_e_syn"])
