# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

# Non-capacitive channels are all channels that do not follow the default
# Hodgkin-Huxley update:
#
# 1/C dV/dt = i
#
# These non-capacitive achieve this by modifying the voltage as part of their
# `update_states()` method.
