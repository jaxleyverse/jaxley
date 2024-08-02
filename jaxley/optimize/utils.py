# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax.numpy as jnp
from jax import tree_util


def l2_norm(x: "PyTree") -> jnp.array:
    """Return the L2-norm of a pytree. Taken from GH jax/issues/3124."""
    leaves, _ = tree_util.tree_flatten(x)
    return jnp.sqrt(sum([jnp.sum(leaf**2) for leaf in leaves]))
