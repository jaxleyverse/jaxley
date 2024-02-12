from typing import Dict, List, Optional

import jax.numpy as jnp
import optax


class ParameterTypeOptimizer:
    """Wrapper for `optax` optimizer which allows different lrs for different params."""
    def __init__(self, optimizer: optax.optimizer, lrs: Dict, opt_params: List[Dict[str, jnp.ndarray]]):
        self.base_optimizer = optimizer

        self.optimizers = []
        for params in opt_params:
            names = list(params.keys())
            assert len(names) == 1, "Multiple parameters were added at once."
            name = names[0]
            optimizer = self.base_optimizer(learning_rate=lrs[name])
            self.optimizers.append({name: optimizer})
        
    def init(self, opt_params: List[Dict[str, jnp.ndarray]]):
        opt_states = []
        for params, optimizer in zip(opt_params, self.optimizers):
            opt_state = optimizer.init(params)
            opt_states.append(opt_state)
        return opt_states

    def update(self, gradient, opt_states):
        all_updates = []
        new_opt_states = []
        for grad, state, opt in zip(gradient, opt_states, self.optimizers):
            updates, new_opt_states = opt.update(grad, state)
            all_updates.append(updates)
            new_opt_states.append(new_opt_states)
        return updates, new_opt_states
