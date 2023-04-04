import numpy as np


class Recording:
    def __init__(self, cell_ind, branch_ind, loc):
        self.cell_ind = cell_ind
        self.branch_ind = branch_ind
        self.loc = loc
