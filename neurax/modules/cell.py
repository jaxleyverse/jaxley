from neurax.modules.base import Module, View
from neurax.modules.branch import Branch, BranchView


class Cell(Module):
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
        return BranchView(self, self.nodes)

    def step(self):
        pass


class CellView(View):
    def __init__(self, pointer, view):
        super().__init__(pointer, view)

    def __call__(self, index):
        return super().adjust_view("cell_index", index)

    def __getattr__(self, key):
        assert key == "branch"
        return BranchView(self.pointer, self.view)
