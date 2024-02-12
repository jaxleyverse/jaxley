from typing import Dict, List, Optional

import jax.numpy as jnp


class TypeOptimizer:
    """Wrapper for `optax` optimizer which allows different lrs for different params."""

    def __init__(
        self,
        optimizer,
        lrs: Dict,
        opt_params: List[Dict[str, jnp.ndarray]],
    ) -> None:
        """Create the optimizers.

        This requires access to `opt_params` in order to know how many optimizers
        should be created. It creates `len(opt_params)` optimizers.

        Args:
            optimizer: The `optax.optimizer` (not instantiated) which should be used.
            lrs: The learning rates to be used for different kinds of parameters.
            opt_params: The parameters to be optimizer. The exact values are not used,
                only the number of elements in the list and the key of each dict.
        """
        self.base_optimizer = optimizer

        self.optimizers = []
        for params in opt_params:
            names = list(params.keys())
            assert len(names) == 1, "Multiple parameters were added at once."
            name = names[0]
            optimizer = self.base_optimizer(learning_rate=lrs[name])
            self.optimizers.append({name: optimizer})

    def init(self, opt_params: List[Dict[str, jnp.ndarray]]):
        """Initialize the optimizers. Equivalent to `optax.optimizers.init()`."""
        opt_states = []
        for params, optimizer in zip(opt_params, self.optimizers):
            name = list(optimizer.keys())[0]
            opt_state = optimizer[name].init(params)
            opt_states.append(opt_state)
        return opt_states

    def update(self, gradient, opt_state):
        """Update the optimizers. Equivalent to `optax.optimizers.update()`."""
        all_updates = []
        new_opt_states = []
        for grad, state, opt in zip(gradient, opt_state, self.optimizers):
            name = list(opt.keys())[0]
            updates, new_opt_state = opt[name].update(grad, state)
            all_updates.append(updates)
            new_opt_states.append(new_opt_state)
        return all_updates, new_opt_states
