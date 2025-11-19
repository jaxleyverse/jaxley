# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.lax import ScatterDimensionNumbers, scatter_add
from jax.typing import ArrayLike


def gather_synapes(
    number_of_compartments: ArrayLike,
    post_syn_comp_inds: np.ndarray,
    current_each_synapse_constant_term: ArrayLike,
) -> tuple[Array, Array]:
    """Compute current at the post synapse.

    All this does it that it sums the synaptic currents that come into a particular
    compartment. It returns an array of as many elements as there are compartments.
    """
    incoming_currents_content = jnp.zeros((number_of_compartments,))
    incoming_currents_content = incoming_currents_content.at[post_syn_comp_inds].add(
        current_each_synapse_constant_term
    )
    return incoming_currents_content
