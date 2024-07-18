# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp


class TypeOptimizer:
    """`optax` wrapper which allows different argument values for different params."""

    def __init__(
        self,
        optimizer: Callable,
        optimizer_args: Dict[str, Any],
        opt_params: List[Dict[str, jnp.ndarray]],
    ):
        """Create the optimizers.

        This requires access to `opt_params` in order to know how many optimizers
        should be created. It creates `len(opt_params)` optimizers.

        Example usage:
        ```
        lrs = {"HH_gNa": 0.01, "radius": 1.0}
        optimizer = TypeOptimizer(lambda lr: optax.adam(lr), lrs, opt_params)
        opt_state = optimizer.init(opt_params)
        ```

        ```
        optimizer_args = {"HH_gNa": [0.01, 0.4], "radius": [1.0, 0.8]}
        optimizer = TypeOptimizer(
            lambda args: optax.sgd(args[0], momentum=args[1]),
            optimizer_args,
            opt_params
        )
        opt_state = optimizer.init(opt_params)
        ```

        Args:
            optimizer: A Callable that takes the learning rate and returns the
                `optax.optimizer` which should be used.
            optimizer_args: The arguments for different kinds of parameters.
                Each item of the dictionary will be passed to the `Callable` passed to
                `optimizer`.
            opt_params: The parameters to be optimized. The exact values are not used,
                only the number of elements in the list and the key of each dict.
        """
        self.base_optimizer = optimizer

        self.optimizers = []
        for params in opt_params:
            names = list(params.keys())
            assert len(names) == 1, "Multiple parameters were added at once."
            name = names[0]
            optimizer = self.base_optimizer(optimizer_args[name])
            self.optimizers.append({name: optimizer})

    def init(self, opt_params: List[Dict[str, jnp.ndarray]]) -> List:
        """Initialize the optimizers. Equivalent to `optax.optimizers.init()`."""
        opt_states = []
        for params, optimizer in zip(opt_params, self.optimizers):
            name = list(optimizer.keys())[0]
            opt_state = optimizer[name].init(params)
            opt_states.append(opt_state)
        return opt_states

    def update(self, gradient: jnp.ndarray, opt_state: List) -> Tuple[List, List]:
        """Update the optimizers. Equivalent to `optax.optimizers.update()`."""
        all_updates = []
        new_opt_states = []
        for grad, state, opt in zip(gradient, opt_state, self.optimizers):
            name = list(opt.keys())[0]
            updates, new_opt_state = opt[name].update(grad, state)
            all_updates.append(updates)
            new_opt_states.append(new_opt_state)
        return all_updates, new_opt_states
