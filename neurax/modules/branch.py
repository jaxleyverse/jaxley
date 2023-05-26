from neurax.modules.base import Module, View
from neurax.modules.compartment import Compartment, CompartmentView


class Branch(Module):
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
        return CompartmentView(self, self.nodes)

    def step(self):
        pass




class BranchView(View):
    def __init__(self, pointer, view):
        super().__init__(pointer, view)

    def __call__(self, index):
        return super().adjust_view("branch_index", index)

    def __getattr__(self, key):
        assert key == "comp"
        return CompartmentView(self.pointer, self.view)
