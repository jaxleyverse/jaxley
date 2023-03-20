import jax.numpy as jnp
from jax import vmap
from tridiax.thomas import thomas_triang, thomas_backsub


def solve_branched(
    parents,
    lowers,
    diags,
    uppers,
    solves,
    branch_cond,
):
    """
    Solve branched.
    """
    diags, uppers, solves = triang_branched(
        parents, lowers, diags, uppers, solves, branch_cond
    )
    solves = backsub_branched(parents, diags, uppers, solves, branch_cond)
    return solves


def triang_branched(
    parents,
    lowers,
    diags,
    uppers,
    solves,
    branch_cond,
):
    """
    Triang.
    """
    branches_in_each_level = [jnp.asarray([1, 2]), jnp.asarray([0])]
    for bil in branches_in_each_level:
        diags, uppers, solves = _triang_level(bil, lowers, diags, uppers, solves)
        diags, solves = _eliminate_parents_upper(
            bil, parents, diags, solves, branch_cond
        )

    return diags, uppers, solves


def backsub_branched(
    parents,
    diags,
    uppers,
    solves,
    branch_cond,
):
    """
    Backsub.
    """
    branches_in_each_level = [jnp.asarray([0]), jnp.asarray([1, 2])]
    for bil in branches_in_each_level:
        solves = _eliminate_children_lower(bil, parents, solves, branch_cond)
        solves = _backsub_level(bil, diags, uppers, solves)
    return solves


def _triang_level(branches_in_level, lowers, diags, uppers, solves):
    bil = branches_in_level
    new_diags, new_uppers, new_solves = vmap(thomas_triang, in_axes=(0, 0, 0, 0))(
        lowers[bil], diags[bil], uppers[bil], solves[bil]
    )
    diags = diags.at[bil].set(new_diags)
    uppers = uppers.at[bil].set(new_uppers)
    solves = solves.at[bil].set(new_solves)

    return diags, uppers, solves


def _backsub_level(branches_in_level, diags, uppers, solves):
    bil = branches_in_level
    solves.at[bil].set(
        vmap(thomas_backsub, in_axes=(0, 0, 0))(solves[bil], uppers[bil], diags[bil])
    )
    return solves


def _eliminate_single_parent_upper(diag_at_branch, solve_at_branch, branch_cond):
    last_of_3 = diag_at_branch
    last_of_3_solve = solve_at_branch

    multiplying_factor = branch_cond / last_of_3

    update_diag = -multiplying_factor * branch_cond
    update_solve = -multiplying_factor * last_of_3_solve
    return update_diag, update_solve


def _eliminate_parents_upper(branches_in_level, parents, diags, solves, branch_cond):
    bil = branches_in_level
    new_diag, new_solve = vmap(_eliminate_single_parent_upper, in_axes=(0, 0, None))(
        diags[bil, -1], solves[bil, -1], branch_cond
    )
    for i, b in enumerate(bil):
        diags = diags.at[parents[b], 0].set(diags[parents[b], 0] + new_diag[i])
        solves = solves.at[parents[b], 0].set(solves[parents[b], 0] + new_solve[i])
    return diags, solves


def _eliminate_children_lower(branches_in_level, parents, solves, branch_cond):
    bil = branches_in_level
    for b in bil:
        solves = solves.at[b, -1].set(
            solves[b, -1] - branch_cond * solves[parents[b], 0]
        )
    return solves
