class Connection:
    def __init__(
        self,
        pre_cell_ind,
        pre_branch_ind,
        pre_loc,
        post_cell_ind,
        post_branch_ind,
        post_loc,
        synaptic_cond,
    ):
        self.pre_cell_ind = pre_cell_ind
        self.pre_branch_ind = pre_branch_ind
        self.pre_loc = pre_loc
        self.post_cell_ind = post_cell_ind
        self.post_branch_ind = post_branch_ind
        self.post_loc = post_loc
        self.synaptic_cond = synaptic_cond
