# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from copy import deepcopy
from typing import Any, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.experimental.sparse.linalg import spsolve as jax_spsolve
from jax.lax import fori_loop
from tridiax.stone import stone_backsub_lower, stone_triang_upper
from tridiax.thomas import thomas_backsub_lower, thomas_triang_upper

from jaxley.utils.cell_utils import group_and_sum
from jaxley.utils.solver_utils import JaxleySolveIndexer


def step_voltage_explicit(
    voltages: jnp.ndarray,
    voltage_terms: jnp.ndarray,
    constant_terms: jnp.ndarray,
    axial_conductances: jnp.ndarray,
    sinks,
    sources,
    types,
    delta_t: float,
) -> jnp.ndarray:
    """Solve one timestep of branched nerve equations with explicit (forward) Euler."""
    update = _voltage_vectorfield(
        voltages,
        voltage_terms,
        constant_terms,
        axial_conductances,
        sinks,
        sources,
        types,
    )
    new_voltates = voltages + delta_t * update
    return new_voltates


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
    diags = delta_t * voltage_terms
    diags = diags.at[internal_node_inds].add(1.0)
    #
    # if-case needed because `.at` does not allow empty inputs, but the input is
    # empty for compartments.
    if len(sinks) > 0:
        diags = diags.at[sinks].add(axial_conductances)

    # Build solve.
    solves = jnp.zeros(n_nodes)
    solves = solves.at[internal_node_inds].set(
        voltages[internal_node_inds] + delta_t * constant_terms[internal_node_inds]
    )

    # Concatenate diagonals and off-diagonals (which are just `-axial_conductances`).
    all_values = jnp.concatenate([diags, -axial_conductances])

    # Solve the voltage equations.
    solution = jax_spsolve(all_values[data_inds], indices, indptr, solves)
    return solution


def step_voltage_implicit_with_dhs_solve(
    voltages,
    voltage_terms,
    constant_terms,
    axial_conductances,
    internal_node_inds,
    sinks,
    n_nodes,
    solve_indexer: Dict[str, Any],
    optimize_for_gpu: bool,
    delta_t,
):
    """Return voltage update via compartment-based matrix inverse.

    Combined with an approriate solve order, this results in `dendritic hierarchical
    scheduling` (DHS, Zhang et al., 2023). The solve order is defined in the graph
    backend of `Jaxley`.

    Args:
        ordered_comp_edges: A list which edges between compartments, each being a
            tuple (child_compartment, parent_compartment). The order of the list
            indicates the solve order.
        map_to_solve_order: An array of indices that permutes diagonal elements into
            the order of the solve. E.g.: `voltages = voltages[mapping_array]`.
        inv_map_to_solve_order: An array of indices that permutes diagonal elements back
            into compartment order. E.g.: `voltages = voltages[inv_mapping_array]`.
        map_to_solve_order_lower_and_upper: An array of indices that permutes
            the concatenation of lowers and uppers into the order of the solve:
            `lowers_and_uppers = lowers_and_uppers[map_to_solve_order_lower_and_upper]`.
        optimize_for_gpu: If True, it does two things: (1) it unrolls the for-loop
            for the triangularization stage. (2) It uses recursive doubling (also
            unrolled) for the backsubstitution stage. Setting this to `True` will
            largely speed up runs on GPU, but it will slow down compilation time and
            run time on CPU.
    """
    axial_conductances = delta_t * axial_conductances

    # Build diagonals.
    diags = delta_t * voltage_terms
    diags = diags.at[internal_node_inds].add(1.0)
    #
    # if-case needed because `.at` does not allow empty inputs, but the input is
    # empty for compartments.
    if len(sinks) > 0:
        diags = diags.at[sinks].add(axial_conductances)

    # Build solve.
    solves = jnp.zeros(n_nodes)
    solves = solves.at[internal_node_inds].set(
        voltages[internal_node_inds] + delta_t * constant_terms[internal_node_inds]
    )

    # Why `n_nodes > 1`? For compartments (or point neurons), we save computation and
    # compile time by skipping the entire solve procedure.
    if len(axial_conductances) > 0:
        # Build lower and upper matrix.
        lowers_and_uppers = -axial_conductances

        # Reorder diagonals and solves.
        diags = diags[solve_indexer["map_to_solve_order"]]
        solves = solves[solve_indexer["map_to_solve_order"]]

        # Reorder the lower and upper values.
        lowers = lowers_and_uppers[solve_indexer["map_to_solve_order_lower"]]
        uppers = lowers_and_uppers[solve_indexer["map_to_solve_order_upper"]]
        ordered_comp_edges = solve_indexer["node_order_grouped"]
        flipped_comp_edges = jnp.flip(ordered_comp_edges, axis=0)

        # Add a spurious compartment that is modified by the masking.
        diags = jnp.concatenate([diags, jnp.asarray([1.0])])
        solves = jnp.concatenate([solves, jnp.asarray([0.0])])
        uppers = jnp.concatenate([uppers, jnp.asarray([0.0])])
        lowers = jnp.concatenate([lowers, jnp.asarray([0.0])])

        # Solve the voltage equations.
        #
        steps = len(flipped_comp_edges)
        if not optimize_for_gpu:
            # Triangulate.
            steps = len(flipped_comp_edges)
            init = (diags, solves, lowers, uppers, flipped_comp_edges)
            diags, solves, _, _, _ = fori_loop(0, steps, _comp_based_triang, init)

            # Backsubstitute.
            lowers /= diags
            solves /= diags
            diags = jnp.ones_like(solves)
            init = (solves, lowers, ordered_comp_edges)
            solves, _, _ = fori_loop(0, steps, _comp_based_backsub, init)
        else:
            # Triangulate by unrolling the loop of the levels.
            for i in range(steps):
                diags, solves, _, _, _ = _comp_based_triang(
                    i, (diags, solves, lowers, uppers, flipped_comp_edges)
                )

            # Backsubstitute with recursive doubling.
            diags, solves = _comp_based_backsub_recursive_doubling(
                diags, solves, lowers, steps, n_nodes, solve_indexer["parent_lookup"]
            )

        # Remove the spurious compartment. This compartment got modified by masking of
        # compartments in certain levels.
        diags = diags[:-1]
        solves = solves[:-1]

    # Get inverse of the diagonalized matrix.
    solution = solves / diags
    solution = solution[solve_indexer["inv_map_to_solve_order"]]

    return solution


def _comp_based_triang(index, carry):
    """Triangulate the quasi-tridiagonal system compartment by compartment."""
    diags, solves, lowers, uppers, flipped_comp_edges = carry

    # `flipped_comp_edges` has shape `(num_levels, num_comps_per_level, 2)`. We first
    # get the relevant level with `[index]` and then we get all children and parents
    # in the level.
    comp_edge = flipped_comp_edges[index]
    child = comp_edge[:, 0]
    parent = comp_edge[:, 1]

    lower_val = lowers[child]
    upper_val = uppers[child]
    child_diag = diags[child]
    child_solve = solves[child]

    # Factor that the child row has to be multiplied by.
    multiplier = upper_val / child_diag

    # Updates to diagonal and solve
    diags = diags.at[parent].add(-lower_val * multiplier)
    solves = solves.at[parent].add(-child_solve * multiplier)

    return (diags, solves, lowers, uppers, flipped_comp_edges)


def _comp_based_backsub(index, carry):
    """Backsubstitute the quasi-tridiagonal system compartment by compartment."""
    solves, lowers, comp_edges = carry

    # `comp_edges` has shape `(num_levels, num_comps_per_level, 2)`. We first get the
    # relevant level with `[index]` and then we get all children and parents in the
    # level.
    comp_edge = comp_edges[index]
    child = comp_edge[:, 0]
    parent = comp_edge[:, 1]

    # Updates to diagonal and solve
    solves = solves.at[child].add(-solves[parent] * lowers[child])
    return (solves, lowers, comp_edges)


def _comp_based_backsub_recursive_doubling(
    diags: jnp.ndarray,
    solves: jnp.ndarray,
    lowers: jnp.ndarray,
    steps: int,
    n_nodes: int,
    parent_lookup: np.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Backsubstitute with recursive doubling.

    This function contains a lot of math, so I will describe what is going on here:

    The matrix describes a system like:
    diag[n] * x[n] + lower[n] * x[parent] = solve[n]

    We rephrase this as:
    x[n] = solve[n]/diag[n] - lower[n]/diag[n] * x[parent].

    and we call variables as follows:
    solve/diag => solve_effect
    -lower/diag => lower_effect

    This gives:
    x[n] = solve_effect[n] + lower_effect[n] * x[parent].

    Recursive doubling solves this equation for `x` in log_2(N) steps. How?

    (1) Notice that lower_effect[n]=0, because x[0] has no parent.

    (2) In the first step, recursive doubling substitutes x[parent] into
    every equation. This leads to something like:
    x[n] = solve_effect[n] + lower_effect[n] * (solve_effect[parent] + ...
    ...lower_effect[parent] * x[parent[parent]])

    Abbreviate this as:
    new_solve_effect[n] = solve_effect[n] + lower_effect[n] * solve_effect[parent]
    new_lower_effect[n] = lower_effect[n] + lower_effect[parent]
    x[n] = new_solve_effect[n] + new_lower_effect[n] * x[parent[parent]]
    Importantly, every node n is now a function of its two-step parent.

    (3) In the next step, recursive doubling substitutes x[parent[parent]].
    Since x[parent[parent]] already depends on its own _two-step_ parent,
    every node then depends on its four step parent. This introduces the
    log_2 scaling.

    (4) The algorithm terminates when all `new_lower_effect=0`. This
    naturally happens because `lower_effect[0]=0`, and the recursion
    keeps multiplying new_lower_effect with the `lower_effect[parent]`.
    """
    # Why `lowers = lowers.at[0].set(0.0)`? During triangulation (and the
    # cpu-optimized solver), we never access `lowers[0]`. Its value should
    # be zero (because the zero-eth compartment does not have a `lower`), but
    # it is not for coding convenience in the other solvers. For the recursive
    # doubling solver below, we do use lowers[0], so we set it to the value
    # it should have anyways: 0.
    lowers = lowers.at[0].set(0.0)

    # Rephrase the equations as a recursion.
    # x[n] = solve[n]/diag[n] - lower[n]/diag[n] * x[parent].
    # x[n] = solve_effect[n] + lower_effect[n] * x[parent].
    lower_effect = -lowers / diags
    solve_effect = solves / diags

    step = 1
    while step <= steps:
        # For each node, get its k-step parent, where k=`step`.
        k_step_parent = np.arange(n_nodes + 1)
        for _ in range(step):
            k_step_parent = parent_lookup[k_step_parent]

        # Update.
        solve_effect = lower_effect * solve_effect[k_step_parent] + solve_effect
        lower_effect *= lower_effect[k_step_parent]
        step *= 2

    # We have to return a `diags` becaus the final solution is computed as
    # `solves/diags` (see `step_voltage_implicit_with_dhs_solve`). For recursive
    # doubling, the solution should just be `solve_effect`, so we define diags as
    # 1.0 so the division has no effect.
    diags = jnp.ones_like(solve_effect)
    solves = solve_effect
    return diags, solves


def _voltage_vectorfield(
    voltages: jnp.ndarray,
    voltage_terms: jnp.ndarray,
    constant_terms: jnp.ndarray,
    axial_conductances: jnp.ndarray,
    sinks,
    sources,
    types,
) -> jnp.ndarray:
    """Evaluate the vectorfield of the nerve equation."""
    if np.sum(np.isin(types, [1, 2, 3, 4])) > 0:
        raise NotImplementedError(
            f"Forward Euler is not implemented for branched morphologies."
        )

    # Membrane current update.
    vecfield = -voltage_terms * voltages + constant_terms

    # Current through segments within the same branch.
    if len(sinks) > 0:
        vecfield = vecfield.at[sinks].add(
            (voltages[sources] - voltages[sinks]) * axial_conductances
        )

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
