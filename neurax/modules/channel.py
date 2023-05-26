from neurax.modules.base import Module, View


class Channel(Module):
    def __init__(self, nodes):
        self.nodes = nodes
        self.params = {
            "g_na": np.zeros((30,)),
            "g_k": np.zeros((30,)),
            "g_leak": np.zeros((30,)),
        }

    def set_params(self, key, val):
        self.params[key][:] = val

    def step(self):
        pass


class ChannelView(View):
    def __init__(self, pointer, view):
        super().__init__(pointer, view)

    def __call__(self, index):
        return super().adjust_view("channel_index", index)
