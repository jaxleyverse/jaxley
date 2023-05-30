from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import jax.numpy as jnp


class Synapse:
    synapse_params = None
    synapse_states = None

    def __init__(
        self,
        pre_cell_ind,
        pre_branch_ind,
        pre_loc,
        post_cell_ind,
        post_branch_ind,
        post_loc,
    ):
        self.pre_cell_ind = pre_cell_ind
        self.pre_branch_ind = pre_branch_ind
        self.pre_loc = pre_loc
        self.post_cell_ind = post_cell_ind
        self.post_branch_ind = post_branch_ind
        self.post_loc = post_loc

    def set_params(self, key, val):
        self.params[key][:] = val

    @abstractmethod
    def step(
        self, u, dt, voltages, params
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        pass
