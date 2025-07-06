# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Tuple

import jax.numpy as jnp
import numpy as np
from jax.lax import ScatterDimensionNumbers, scatter_add


def gather_synapes(
    number_of_compartments: jnp.ndarray,
    post_syn_comp_inds: np.ndarray,
    current_each_synapse_voltage_term: jnp.ndarray,
    current_each_synapse_constant_term: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute current at the post synapse.

    All this does it that it sums the synaptic currents that come into a particular
    compartment. It returns an array of as many elements as there are compartments.
    """
    incoming_currents_voltages = jnp.zeros((number_of_compartments,))
    incoming_currents_content = jnp.zeros((number_of_compartments,))

    dnums = ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    incoming_currents_voltages = scatter_add(
        incoming_currents_voltages,
        post_syn_comp_inds[:, None],
        current_each_synapse_voltage_term,
        dnums,
    )
    incoming_currents_content = scatter_add(
        incoming_currents_content,
        post_syn_comp_inds[:, None],
        current_each_synapse_constant_term,
        dnums,
    )
    return incoming_currents_voltages, incoming_currents_content
