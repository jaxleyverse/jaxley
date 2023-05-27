from typing import Dict, List, Optional, Callable

from neurax.modules.base import Module, View
from neurax.modules.branch import Branch, BranchView


class Cell(Module):
    cell_params: Dict = {}
    cell_states: Dict = {}

    def __init__(self, branches: List[Branch]):
        self.branches = branches
        self.branch_conds = None
        self.num_branches = len(branches)

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
