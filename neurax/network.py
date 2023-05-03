import jax.numpy as jnp
from neurax.utils.syn_utils import prepare_presyn, prepare_postsyn
from neurax.cell import merge_cells, _compute_index_of_kid, cum_indizes_of_kids


class Network:
    """A `Network` is a collection of cells and connectivity patterns."""

    def __init__(self, cells, conns):
        """Initialize network."""
        self.nseg_per_branch = cells[0].nseg_per_branch
        for cell in cells:
            assert (
                cell.nseg_per_branch == self.nseg_per_branch
            ), "Different nseg_per_branch between cells."

        assert isinstance(conns, list), "conns must be a list."
        for conn in conns:
            assert isinstance(conn, list), "conns must be a list of lists."
        self.connectivities = [
            Connectivity(conn, self.nseg_per_branch) for conn in conns
        ]
        self.prepare_cells(cells)

    def prepare_cells(self, cells):
        """Organize multiple cells such that they can be processed in parallel."""
        self.num_branches = [cell.num_branches for cell in cells]
        self.cumsum_num_branches = jnp.cumsum(jnp.asarray([0] + self.num_branches))
        self.max_num_kids = cells[0].max_num_kids
        for c in cells:
            assert (
                self.max_num_kids == c.max_num_kids
            ), "Different max_num_kids between cells."

        parents = [cell.parents for cell in cells]
        self.comb_parents = jnp.concatenate(
            [p.at[1:].add(self.cumsum_num_branches[i]) for i, p in enumerate(parents)]
        )
        self.comb_parents_in_each_level = merge_cells(
            self.cumsum_num_branches, [cell.parents_in_each_level for cell in cells]
        )
        self.comb_branches_in_each_level = merge_cells(
            self.cumsum_num_branches,
            [cell.branches_in_each_level for cell in cells],
            exclude_first=False,
        )

        # Prepare indizes for solve
        comb_ind_of_kids = jnp.concatenate(
            [jnp.asarray(_compute_index_of_kid(cell.parents)) for cell in cells]
        )
        comb_ind_of_kids_in_each_level = [
            comb_ind_of_kids[bil] for bil in self.comb_branches_in_each_level
        ]
        self.comb_cum_kid_inds_in_each_level = cum_indizes_of_kids(
            comb_ind_of_kids_in_each_level, self.max_num_kids
        )

        # Flatten because we flatten all vars.
        self.radiuses = jnp.concatenate([c.radiuses.flatten() for c in cells])
        self.lengths = jnp.concatenate([c.lengths.flatten() for c in cells])
        self.coupling_conds_fwd = jnp.concatenate([c.coupling_conds_fwd for c in cells])
        self.coupling_conds_bwd = jnp.concatenate([c.coupling_conds_bwd for c in cells])
        self.branch_conds_fwd = jnp.concatenate([c.branch_conds_fwd for c in cells])
        self.branch_conds_bwd = jnp.concatenate([c.branch_conds_bwd for c in cells])
        self.summed_coupling_conds = jnp.concatenate(
            [c.summed_coupling_conds for c in cells]
        )


class Connectivity:
    """Given a list of all synapses, this prepares everything for simulation.
    There are two main functions of this class:
    (1) For presynaptic locations, we infer the index of the exact location on a
    single neuron.
    (2) For postsyaptic locations, we also cluster locations such that we can allow
    multiple synaptic contances onto the same postsynaptic compartment.
    """

    def __init__(self, conns, nseg_per_branch):
        pre_syn = prepare_presyn(conns, nseg_per_branch)
        self.pre_syn_cell_inds = pre_syn[0]
        self.pre_syn_inds = pre_syn[1]

        post_syn = prepare_postsyn(conns, nseg_per_branch)
        self.grouped_post_syn_inds = post_syn[0]
        self.grouped_post_syns = post_syn[1]
