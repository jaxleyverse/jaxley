import jax.numpy as jnp
from jax import vmap, lax
from tridiax.thomas import thomas_triang, thomas_backsub
from tridiax.stone import stone_triang, stone_backsub


def solve_branched(
    parents_in_each_level,
    branches_in_each_level,
    parents,
    lowers,
    diags,
    uppers,
    solves,
    branch_cond_fwd,
    branch_cond_bwd,
    solver,
):
    """
    Solve branched.
    """
    diags, uppers, solves = triang_branched(
        parents_in_each_level,
        branches_in_each_level,
        lowers,
        diags,
        uppers,
        solves,
        branch_cond_fwd,
        solver,
    )
    solves = backsub_branched(
        branches_in_each_level,
        parents,
        diags,
        uppers,
        solves,
        branch_cond_bwd,
        solver,
    )
    return solves


def triang_branched(
    parents_in_each_level,
    branches_in_each_level,
    lowers,
    diags,
    uppers,
    solves,
    branch_cond,
    solver,
):
    """
    Triang.
    """
    for bil, parents_in_level in zip(
        reversed(branches_in_each_level[1:]), reversed(parents_in_each_level[1:])
    ):
        diags, uppers, solves = _triang_level(
            bil, lowers, diags, uppers, solves, solver
        )
        diags, solves = _eliminate_parents_upper(
            parents_in_level,
            bil,
            diags,
            solves,
            branch_cond,
        )
    # At last level, we do not want to eliminate anymore.
    diags, uppers, solves = _triang_level(
        branches_in_each_level[0], lowers, diags, uppers, solves, solver
    )

    return diags, uppers, solves


def backsub_branched(
    branches_in_each_level,
    parents,
    diags,
    uppers,
    solves,
    branch_cond,
    solver,
):
    """
    Backsub.
    """
    # At first level, we do not want to eliminate.
    solves = _backsub_level(branches_in_each_level[0], diags, uppers, solves, solver)
    for bil in branches_in_each_level[1:]:
        solves = _eliminate_children_lower(bil, parents, solves, branch_cond)
        solves = _backsub_level(bil, diags, uppers, solves, solver)
    return solves


def _triang_level(branches_in_level, lowers, diags, uppers, solves, solver):
    bil = branches_in_level
    if solver == "stone":
        triang_fn = stone_triang
    elif solver == "thomas":
        triang_fn = thomas_triang
    else:
        raise NameError
    new_diags, new_uppers, new_solves = vmap(triang_fn, in_axes=(0, 0, 0, 0))(
        lowers[bil], diags[bil], uppers[bil], solves[bil]
    )
    diags = diags.at[bil].set(new_diags)
    uppers = uppers.at[bil].set(new_uppers)
    solves = solves.at[bil].set(new_solves)

    return diags, uppers, solves


def _backsub_level(branches_in_level, diags, uppers, solves, solver):
    bil = branches_in_level
    if solver == "stone":
        backsub_fn = stone_backsub
    elif solver == "thomas":
        backsub_fn = thomas_backsub
    else:
        raise NameError
    solves = solves.at[bil].set(
        vmap(backsub_fn, in_axes=(0, 0, 0))(solves[bil], uppers[bil], diags[bil])
    )
    return solves


def _eliminate_single_parent_upper(diag_at_branch, solve_at_branch, branch_cond):
    last_of_3 = diag_at_branch
    last_of_3_solve = solve_at_branch

    multiplying_factor = branch_cond / last_of_3

    update_diag = -multiplying_factor * branch_cond
    update_solve = -multiplying_factor * last_of_3_solve
    return update_diag, update_solve


def _eliminate_parents_upper(
    parents_in_level,
    branches_in_level,
    diags,
    solves,
    branch_cond,
):
    bil = branches_in_level
    new_diag, new_solve = vmap(_eliminate_single_parent_upper, in_axes=(0, 0, 0))(
        diags[bil, -1], solves[bil, -1], branch_cond[bil]
    )
    diags = diags.at[parents_in_level, 0].set(
        diags[parents_in_level, 0] + jnp.sum(jnp.reshape(new_diag, (-1, 2)), axis=1)
    )
    solves = solves.at[parents_in_level, 0].set(
        solves[parents_in_level, 0] + jnp.sum(jnp.reshape(new_solve, (-1, 2)), axis=1)
    )
    return diags, solves


def _eliminate_children_lower(
    branches_in_level,
    parents,
    solves,
    branch_cond,
):
    bil = branches_in_level
    solves = solves.at[bil, -1].set(
        solves[bil, -1] - branch_cond[bil] * solves[parents[bil], 0]
    )
    return solves
