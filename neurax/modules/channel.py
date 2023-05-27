from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from neurax.modules.base import Module, View
import jax.numpy as jnp


class Channel(Module):
    channel_params = None

    def __init__(self):
        super().__init__()
        self.params = self.channel_params
        self.states = self.channel_states

    def set_params(self, key, val):
        self.params[key][:] = val

    @abstractmethod
    def step(self, u, dt, *args) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        pass


class ChannelView(View):
    def __init__(self, pointer, view):
        super().__init__(pointer, view)

    def __call__(self, index):
        return super().adjust_view("channel_index", index)
