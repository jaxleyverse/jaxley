import jaxley as jx
import pytest

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Create an empty compartment
comp = jx.Compartment()

# Check if a ValueError is raised when integrating an empty compartment
with pytest.raises(ValueError):
    v = jx.integrate(comp, delta_t=0.025, t_max=10.0)
