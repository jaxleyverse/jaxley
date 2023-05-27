from typing import Dict, List, Optional, Callable

from neurax.modules.base import Module, View
from neurax.modules.cell import Cell, CellView


class Network(Module):
    network_params: Dict = {}
    network_states: Dict = {}

    def __init__(self, nodes):
        self.nodes = nodes
        self.params = {
            "g_na": np.zeros((30,)),
            "g_k": np.zeros((30,)),
            "g_leak": np.zeros((30,)),
        }

    def set_params(self, key, val):
        self.params[key][:] = val

    def __getattr__(self, key):
        assert key == "cell"
        return CellView(self, self.nodes)

    def step(self):
        pass
