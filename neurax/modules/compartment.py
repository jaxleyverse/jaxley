from neurax.modules.base import Module, View
from neurax.modules.channel import Channel, ChannelView


class Compartment(Module):
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
        return ChannelView(self, self.nodes)

    def step(self):
        pass


class CompView(View):
    def __init__(self, pointer, view):
        super().__init__(pointer, view)

    def __call__(self, index):
        return super().adjust_view("comp_index", index)

    def __getattr__(self, key):
        assert key == "channel"
        return ChannelView(self.pointer, self.view)
