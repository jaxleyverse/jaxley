# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import List

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.experimental.sparse.linalg import spsolve as jax_spsolve
from tridiax.stone import stone_backsub_lower, stone_triang_upper
from tridiax.thomas import thomas_backsub_lower, thomas_triang_upper

from jaxley.utils.cell_utils import group_and_sum
from jaxley.utils.solver_utils import JaxleySolveIndexer


def step_voltage_explicit(
    voltages: jnp.ndarray,
    voltage_terms: jnp.ndarray,
    constant_terms: jnp.ndarray,
    axial_conductances: jnp.ndarray,
    internal_node_inds: jnp.ndarray,
    sinks: jnp.ndarray,
    sources: jnp.ndarray,
    types: jnp.ndarray,
    ncomp_per_branch: jnp.ndarray,
    par_inds: jnp.ndarray,
    child_inds: jnp.ndarray,
    nbranches: int,
    solver: str,
    idx: JaxleySolveIndexer,
    debug_states,
    delta_t: float,
) -> jnp.ndarray:
    """Solve one timestep of branched nerve equations with explicit (forward) Euler."""
    voltages = jnp.reshape(voltages, (nbranches, -1))
    voltage_terms = jnp.reshape(voltage_terms, (nbranches, -1))
    constant_terms = jnp.reshape(constant_terms, (nbranches, -1))

    update = _voltage_vectorfield(
        voltages,
        voltage_terms,
        constant_terms,
        types,
        sources,
        sinks,
        axial_conductances,
        par_inds,
        child_inds,
        nbranches,
        solver,
        delta_t,
        idx,
        debug_states,
    )
    new_voltates = voltages + delta_t * update
    return new_voltates.ravel(order="C")


def step_voltage_implicit_with_jaxley_spsolve(
    voltages: jnp.ndarray,
    voltage_terms: jnp.ndarray,
    constant_terms: jnp.ndarray,
    axial_conductances: jnp.ndarray,
    internal_node_inds: jnp.ndarray,
    sinks: jnp.ndarray,
    sources: jnp.ndarray,
    types: jnp.ndarray,
    ncomp_per_branch: jnp.ndarray,
    par_inds: jnp.ndarray,
    child_inds: jnp.ndarray,
    nbranches: int,
    solver: str,
    idx: JaxleySolveIndexer,
    debug_states,
    delta_t: float,
):
    """Solve one timestep of branched nerve equations with implicit (backward) Euler."""
    # Build diagonals.
    c2c = np.isin(types, [0, 1, 2])
    total_ncomp = idx.cumsum_ncomp[-1]
    diags = jnp.ones(total_ncomp)

    # if-case needed because `.at` does not allow empty inputs, but the input is
    # empty for compartments.
    if len(sinks[c2c]) > 0:
        diags = diags.at[idx.mask(sinks[c2c])].add(delta_t * axial_conductances[c2c])

    diags = diags.at[idx.mask(internal_node_inds)].add(delta_t * voltage_terms)

    # Build solves.
    solves = jnp.zeros(total_ncomp)
    solves = solves.at[idx.mask(internal_node_inds)].add(
        voltages + delta_t * constant_terms
    )

    # Build upper and lower within the branch.
    c2c = types == 0  # c2c = compartment-to-compartment.

    # Build uppers.
    uppers = jnp.zeros(total_ncomp)
    upper_inds = sources[c2c] > sinks[c2c]
    sinks_upper = sinks[c2c][upper_inds]
    if len(sinks_upper) > 0:
        uppers = uppers.at[idx.mask(sinks_upper)].add(
            -delta_t * axial_conductances[c2c][upper_inds]
        )

    # Build lowers.
    lowers = jnp.zeros(total_ncomp)
    lower_inds = sources[c2c] < sinks[c2c]
    sinks_lower = sinks[c2c][lower_inds]
    if len(sinks_lower) > 0:
        lowers = lowers.at[idx.mask(sinks_lower)].add(
            -delta_t * axial_conductances[c2c][lower_inds]
        )

    # Build branchpoint conductances.
    branchpoint_conds_parents = axial_conductances[types == 1]
    branchpoint_conds_children = axial_conductances[types == 2]
    branchpoint_weights_parents = axial_conductances[types == 3]
    branchpoint_weights_children = axial_conductances[types == 4]
    all_branchpoint_vals = jnp.concatenate(
        [branchpoint_weights_parents, branchpoint_weights_children]
    )
    # Find unique group identifiers
    num_branchpoints = len(branchpoint_conds_parents)
    branchpoint_diags = -group_and_sum(
        all_branchpoint_vals, idx.branchpoint_group_inds, num_branchpoints
    )
    branchpoint_solves = jnp.zeros((num_branchpoints,))

    branchpoint_conds_children = -delta_t * branchpoint_conds_children
    branchpoint_conds_parents = -delta_t * branchpoint_conds_parents

    # Here, I move all child and parent indices towards a branchpoint into a larger
    # vector. This is wasteful, but it makes indexing much easier. JIT compiling
    # makes the speed difference negligible.
    # Children.
    bp_conds_children = jnp.zeros(nbranches)
    bp_weights_children = jnp.zeros(nbranches)
    # Parents.
    bp_conds_parents = jnp.zeros(nbranches)
    bp_weights_parents = jnp.zeros(nbranches)

    # `.at[inds]` requires that `inds` is not empty, so we need an if-case here.
    # `len(inds) == 0` is the case for branches and compartments.
    if num_branchpoints > 0:
        bp_conds_children = bp_conds_children.at[child_inds].set(
            branchpoint_conds_children
        )
        bp_weights_children = bp_weights_children.at[child_inds].set(
            branchpoint_weights_children
        )
        bp_conds_parents = bp_conds_parents.at[par_inds].set(branchpoint_conds_parents)
        bp_weights_parents = bp_weights_parents.at[par_inds].set(
            branchpoint_weights_parents
        )

    # Triangulate the linear system of equations.
    (
        diags,
        lowers,
        solves,
        uppers,
        branchpoint_diags,
        branchpoint_solves,
        bp_weights_children,
        bp_conds_parents,
    ) = _triang_branched(
        lowers,
        diags,
        uppers,
        solves,
        bp_conds_children,
        bp_conds_parents,
        bp_weights_children,
        bp_weights_parents,
        branchpoint_diags,
        branchpoint_solves,
        solver,
        ncomp_per_branch,
        idx,
        debug_states,
    )

    # Backsubstitute the linear system of equations.
    (
        solves,
        lowers,
        diags,
        bp_weights_parents,
        branchpoint_solves,
        bp_conds_children,
    ) = _backsub_branched(
        lowers,
        diags,
        uppers,
        solves,
        bp_conds_children,
        bp_conds_parents,
        bp_weights_children,
        bp_weights_parents,
        branchpoint_diags,
        branchpoint_solves,
        solver,
        ncomp_per_branch,
        idx,
        debug_states,
    )
    return solves.ravel(order="C")[idx.mask(internal_node_inds)]


def step_voltage_implicit_with_jax_spsolve(
    voltages,
    voltage_terms,
    constant_terms,
    axial_conductances,
    internal_node_inds,
    sinks,
    data_inds,
    indices,
    indptr,
    n_nodes,
    delta_t,
):
    axial_conductances = delta_t * axial_conductances

    # Build diagonals.
    diagonal_values = jnp.zeros(n_nodes)

    # if-case needed because `.at` does not allow empty inputs, but the input is
    # empty for compartments.
    if len(sinks) > 0:
        diagonal_values = diagonal_values.at[sinks].add(axial_conductances)

    diagonal_values = diagonal_values.at[internal_node_inds].add(
        1.0 + delta_t * voltage_terms
    )

    # Concatenate diagonals and off-diagonals (which are just `-axial_conductances`).
    all_values = jnp.concatenate([diagonal_values, -axial_conductances])

    # Build solve.
    solves = jnp.zeros(n_nodes)
    solves = solves.at[internal_node_inds].add(voltages + delta_t * constant_terms)

    # Solve the voltage equations.
    solution = jax_spsolve(all_values[data_inds], indices, indptr, solves)[
        internal_node_inds
    ]
    return solution


def _voltage_vectorfield(
    voltages: jnp.ndarray,
    voltage_terms: jnp.ndarray,
    constant_terms: jnp.ndarray,
    types: jnp.ndarray,
    sources: jnp.ndarray,
    sinks: jnp.ndarray,
    axial_conductances: jnp.ndarray,
    par_inds: jnp.ndarray,
    child_inds: jnp.ndarray,
    nbranches: int,
    solver: str,
    delta_t: float,
    idx: JaxleySolveIndexer,
    debug_states,
) -> jnp.ndarray:
    """Evaluate the vectorfield of the nerve equation."""
    if np.sum(np.isin(types, [1, 2, 3, 4])) > 0:
        raise NotImplementedError(
            f"Forward Euler is not implemented for branched morphologies."
        )

    # Membrane current update.
    vecfield = -voltage_terms * voltages + constant_terms

    # Build upper and lower within the branch.
    c2c = types == 0  # c2c = compartment-to-compartment.

    # Build uppers.
    upper_inds = sources[c2c] > sinks[c2c]
    if len(upper_inds) > 0:
        uppers = axial_conductances[c2c][upper_inds]
    else:
        uppers = jnp.asarray([])

    # Build lowers.
    lower_inds = sources[c2c] < sinks[c2c]
    if len(lower_inds) > 0:
        lowers = axial_conductances[c2c][lower_inds]
    else:
        lowers = jnp.asarray([])

    # For networks consisting of branches.
    uppers = jnp.reshape(uppers, (nbranches, -1))
    lowers = jnp.reshape(lowers, (nbranches, -1))

    # Current through segments within the same branch.
    vecfield = vecfield.at[:, :-1].add((voltages[:, 1:] - voltages[:, :-1]) * uppers)
    vecfield = vecfield.at[:, 1:].add((voltages[:, :-1] - voltages[:, 1:]) * lowers)

    return vecfield


def _triang_branched(
    lowers,
    diags,
    uppers,
    solves,
    branchpoint_conds_children,
    branchpoint_conds_parents,
    branchpoint_weights_children,
    branchpoint_weights_parents,
    branchpoint_diags,
    branchpoint_solves,
    tridiag_solver,
    ncomp_per_branch,
    idx,
    debug_states,
):
    """Triangulation."""
    for cil, pil in zip(
        reversed(idx.children_in_level), reversed(idx.parents_in_level)
    ):
        diags, lowers, solves, uppers = _triang_level(
            cil[:, 0],
            lowers,
            diags,
            uppers,
            solves,
            tridiag_solver,
            idx,
        )
        (
            branchpoint_diags,
            branchpoint_solves,
            branchpoint_weights_children,
        ) = _eliminate_children_lower(
            cil,
            diags,
            solves,
            branchpoint_conds_children,
            branchpoint_weights_children,
            branchpoint_diags,
            branchpoint_solves,
            idx,
        )
        diags, solves, branchpoint_conds_parents = _eliminate_parents_upper(
            pil,
            diags,
            solves,
            branchpoint_conds_parents,
            branchpoint_weights_parents,
            branchpoint_diags,
            branchpoint_solves,
            ncomp_per_branch,
            idx,
        )
    # At last level, we do not want to eliminate anymore.
    diags, lowers, solves, uppers = _triang_level(
        idx.root_inds, lowers, diags, uppers, solves, tridiag_solver, idx
    )
    return (
        diags,
        lowers,
        solves,
        uppers,
        branchpoint_diags,
        branchpoint_solves,
        branchpoint_weights_children,
        branchpoint_conds_parents,
    )


def _backsub_branched(
    lowers,
    diags,
    uppers,
    solves,
    branchpoint_conds_children,
    branchpoint_conds_parents,
    branchpoint_weights_children,
    branchpoint_weights_parents,
    branchpoint_diags,
    branchpoint_solves,
    tridiag_solver,
    ncomp_per_branch,
    idx,
    debug_states,
):
    """
    Backsubstitution.
    """
    # At first level, we do not want to eliminate.
    solves, lowers, diags = _backsub_level(
        idx.root_inds,
        diags,
        lowers,
        solves,
        tridiag_solver,
        idx,
    )
    counter = 0
    for cil, pil in zip(idx.children_in_level, idx.parents_in_level):
        branchpoint_weights_parents, branchpoint_solves = _eliminate_parents_lower(
            pil,
            diags,
            solves,
            branchpoint_weights_parents,
            branchpoint_solves,
            ncomp_per_branch,
            idx,
        )
        branchpoint_conds_children, solves = _eliminate_children_upper(
            cil,
            solves,
            branchpoint_conds_children,
            branchpoint_diags,
            branchpoint_solves,
            idx,
        )
        solves, lowers, diags = _backsub_level(
            cil[:, 0], diags, lowers, solves, tridiag_solver, idx
        )
        counter += 1
    return (
        solves,
        lowers,
        diags,
        branchpoint_weights_parents,
        branchpoint_solves,
        branchpoint_conds_children,
    )


def _triang_level(cil, lowers, diags, uppers, solves, tridiag_solver, idx):
    if tridiag_solver == "jaxley.stone":
        triang_fn = stone_triang_upper
    elif tridiag_solver == "jaxley.thomas":
        triang_fn = thomas_triang_upper
    else:
        raise NameError
    new_diags, new_lowers, new_solves = vmap(triang_fn, in_axes=(0, 0, 0, 0))(
        lowers[idx.lower(cil)],
        diags[idx.branch(cil)],
        uppers[idx.upper(cil)],
        solves[idx.branch(cil)],
    )
    diags = diags.at[idx.branch(cil)].set(new_diags)
    lowers = lowers.at[idx.lower(cil)].set(new_lowers)
    solves = solves.at[idx.branch(cil)].set(new_solves)
    uppers = uppers.at[idx.upper(cil)].set(0.0)

    return diags, lowers, solves, uppers


def _backsub_level(
    cil: jnp.ndarray,
    diags: jnp.ndarray,
    lowers: jnp.ndarray,
    solves: jnp.ndarray,
    tridiag_solver: str,
    idx,
) -> jnp.ndarray:
    bil = cil
    if tridiag_solver == "jaxley.stone":
        backsub_fn = stone_backsub_lower
    elif tridiag_solver == "jaxley.thomas":
        backsub_fn = thomas_backsub_lower
    else:
        raise NameError
    solves = solves.at[idx.branch(bil)].set(
        vmap(backsub_fn, in_axes=(0, 0, 0))(
            solves[idx.branch(bil)], lowers[idx.lower(bil)], diags[idx.branch(bil)]
        )
    )
    lowers = lowers.at[idx.lower(bil)].set(0.0)
    diags = diags.at[idx.branch(bil)].set(1.0)
    return solves, lowers, diags


def _eliminate_children_lower(
    cil,
    diags,
    solves,
    branchpoint_conds_children,
    branchpoint_weights_children,
    branchpoint_diags,
    branchpoint_solves,
    idx,
):
    bil = cil[:, 0]
    bpil = cil[:, 1]
    new_diag, new_solve = vmap(_eliminate_single_child_lower, in_axes=(0, 0, 0, 0))(
        diags[idx.first(bil)],
        solves[idx.first(bil)],
        branchpoint_conds_children[bil],
        branchpoint_weights_children[bil],
    )
    branchpoint_diags = branchpoint_diags.at[bpil].add(new_diag)
    branchpoint_solves = branchpoint_solves.at[bpil].add(new_solve)
    branchpoint_weights_children = branchpoint_weights_children.at[bil].set(0.0)
    return branchpoint_diags, branchpoint_solves, branchpoint_weights_children


def _eliminate_single_child_lower(
    diags,
    solves,
    branchpoint_conds_children,
    branchpoint_weights_children,
):
    multiplying_factor = -branchpoint_weights_children / diags

    update_diag = multiplying_factor * branchpoint_conds_children
    update_solve = multiplying_factor * solves
    return update_diag, update_solve


def _eliminate_parents_upper(
    pil,
    diags,
    solves,
    branchpoint_conds_parents,
    branchpoint_weights_parents,
    branchpoint_diags,
    branchpoint_solves,
    ncomp_per_branch: jnp.ndarray,
    idx,
):
    bil = pil[:, 0]
    bpil = pil[:, 1]
    new_diag, new_solve = vmap(_eliminate_single_parent_upper, in_axes=(0, 0, 0, 0))(
        branchpoint_conds_parents[bil],
        branchpoint_weights_parents[bil],
        branchpoint_diags[bpil],
        branchpoint_solves[bpil],
    )

    # Update the diagonal elements and `b` in `Ax=b` (called `solves`).
    diags = diags.at[idx.last(bil)].add(new_diag)
    solves = solves.at[idx.last(bil)].add(new_solve)
    branchpoint_conds_parents = branchpoint_conds_parents.at[bil].set(0.0)

    return diags, solves, branchpoint_conds_parents


def _eliminate_single_parent_upper(
    branchpoint_conds_parents,
    branchpoint_weights_parents,
    branchpoint_diags,
    branchpoint_solves,
):
    multiplying_factor = branchpoint_conds_parents / branchpoint_diags

    update_diag = -multiplying_factor * branchpoint_weights_parents
    update_solve = -multiplying_factor * branchpoint_solves
    return update_diag, update_solve


def _eliminate_parents_lower(
    pil,
    diags,
    solves,
    branchpoint_weights_parents,
    branchpoint_solves,
    ncomp_per_branch: jnp.ndarray,
    idx,
):
    bil = pil[:, 0]
    bpil = pil[:, 1]
    branchpoint_solves = branchpoint_solves.at[bpil].add(
        -solves[idx.last(bil)] * branchpoint_weights_parents[bil] / diags[idx.last(bil)]
    )
    branchpoint_weights_parents = branchpoint_weights_parents.at[bil].set(0.0)
    return branchpoint_weights_parents, branchpoint_solves


def _eliminate_children_upper(
    cil,
    solves,
    branchpoint_conds_children,
    branchpoint_diags,
    branchpoint_solves,
    idx,
):
    bil = cil[:, 0]
    bpil = cil[:, 1]
    solves = solves.at[idx.first(bil)].add(
        -branchpoint_solves[bpil]
        * branchpoint_conds_children[bil]
        / branchpoint_diags[bpil]
    )
    branchpoint_conds_children = branchpoint_conds_children.at[bil].set(0.0)
    return branchpoint_conds_children, solves
