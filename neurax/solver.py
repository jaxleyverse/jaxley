from typing import Callable

import numpy as np
import jax.numpy as jnp


def solve_branched(
    levels,
    parents,
    lowers,
    diags,
    uppers,
    solves,
    branch_cond,
    triang_branch_fn: Callable,
    backsub_branch_fn: Callable,
):
    diags, uppers, solves = triang_branched(
        levels, parents, lowers, diags, uppers, solves, branch_cond, triang_branch_fn
    )
    solves = backsub_branched(
        levels, parents, diags, uppers, solves, branch_cond, backsub_branch_fn
    )
    return solves


def triang_branched(
    levels,
    parents,
    lowers,
    diags,
    uppers,
    solves,
    branch_cond,
    triang_branch_fn: Callable,
):
    for level in range(np.max(levels), -1, -1):

        diags, uppers, solves = _triang_level(
            level, levels, lowers, diags, uppers, solves, triang_branch_fn
        )
        diags, solves = _eliminate_parents_upper(
            level, levels, parents, diags, solves, branch_cond
        )

    return diags, uppers, solves


def backsub_branched(
    levels,
    parents,
    diags,
    uppers,
    solves,
    branch_cond,
    backsub_branch_fn: Callable,
):
    for level in range(np.max(levels) + 1):
        solves = _eliminate_children_lower(level, levels, parents, solves, branch_cond)
        solves = _backsub_level(level, levels, diags, uppers, solves, backsub_branch_fn)
    return solves


def _triang_level(level, levels, lowers, diags, uppers, solves, triang_branch_fn):
    num_branches = len(levels)
    for b in range(num_branches):
        if levels[b] == level:
            diags[b], uppers[b], solves[b] = triang_branch_fn(
                lowers[b], diags[b], uppers[b], solves[b]
            )

    return diags, uppers, solves


def _backsub_level(level, levels, diags, uppers, solves, backsub_branch_fn):
    num_branches = len(levels)
    for b in range(num_branches):
        if levels[b] == level:
            solves[b] = backsub_branch_fn(solves[b], uppers[b], diags[b])

    return solves


def _eliminate_parents_upper(level, levels, parents, diags, solves, branch_cond):
    num_branches = len(levels)
    for b in range(num_branches):
        if levels[b] == level and parents[b] > -1:
            last_of_3 = diags[b][-1]
            last_of_3_solve = solves[b][-1]

            multiplying_factor = branch_cond / last_of_3

            diags[parents[b]] = (
                diags[parents[b]]
                .at[0]
                .set(diags[parents[b]][0] - multiplying_factor * branch_cond)
            )
            solves[parents[b]] = (
                solves[parents[b]]
                .at[0]
                .set(solves[parents[b]][0] - multiplying_factor * last_of_3_solve)
            )
    return diags, solves


def _eliminate_children_lower(level, levels, parents, solves, branch_cond):
    num_branches = len(levels)
    for b in range(num_branches):
        if levels[b] == level and parents[b] > -1:
            solves[b] = (
                solves[b]
                .at[-1]
                .set(solves[b][-1] - branch_cond * solves[parents[b]][0])
            )
    return solves
