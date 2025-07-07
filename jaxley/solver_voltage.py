# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse.linalg import spsolve as jax_spsolve
from jax.lax import fori_loop
from tridiax.stone import stone_backsub_lower, stone_triang_upper


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

    Combined with an appropriate solve order, this results in `dendritic hierarchical
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
        flipped_comp_edges = list(reversed(ordered_comp_edges))

        # Add a spurious compartment that is modified by the masking.
        diags = jnp.concatenate([diags, jnp.asarray([1.0])])
        solves = jnp.concatenate([solves, jnp.asarray([0.0])])
        uppers = jnp.concatenate([uppers, jnp.asarray([0.0])])
        lowers = jnp.concatenate([lowers, jnp.asarray([0.0])])

        # Solve the voltage equations.
        #
        steps = len(flipped_comp_edges)
        if not optimize_for_gpu:
            # Cast from a list to a np.array.
            # `ordered_comp_edges` has shape `(num_levels, num_comps_per_level, 2)`,
            # and `num_comps_per_level=1` for CPU.
            ordered_comp_edges = np.asarray(ordered_comp_edges)
            flipped_comp_edges = np.asarray(flipped_comp_edges)

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

    # We have to return a `diags` because the final solution is computed as
    # `solves/diags` (see `step_voltage_implicit_with_dhs_solve`). For recursive
    # doubling, the solution should just be `solve_effect`, so we define diags as
    # 1.0 so the division has no effect.
    diags = jnp.ones_like(solve_effect)
    solves = solve_effect
    return diags, solves


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


def step_voltage_implicit_with_stone(
    voltages: jnp.ndarray,
    voltage_terms: jnp.ndarray,
    constant_terms: jnp.ndarray,
    axial_conductances: jnp.ndarray,
    internal_node_inds: jnp.ndarray,
    n_nodes: int,
    sinks: jnp.ndarray,
    sources: jnp.ndarray,
    types: jnp.ndarray,
    delta_t: float,
):
    """Solve one timestep of branched nerve equations with implicit (backward) Euler."""
    if np.sum(np.isin(types, [1, 2, 3, 4])) > 0:
        raise NotImplementedError(
            f"The stone solver is not implemented for branched morphologies."
        )

    axial_conductances = delta_t * axial_conductances
    print("axial_conductances", axial_conductances)

    # Build diagonals.
    diags = delta_t * voltage_terms
    diags = diags.at[internal_node_inds].add(1.0)
    #
    # if-case needed because `.at` does not allow empty inputs, but the input is
    # empty for compartments.
    if len(sinks) > 0:
        diags = diags.at[sinks].add(axial_conductances)

    lower_inds = sinks > sources
    lowers = -axial_conductances[lower_inds]

    upper_inds = sinks < sources
    uppers = -axial_conductances[upper_inds]

    # Build solve.
    solves = jnp.zeros(n_nodes)
    solves = solves.at[internal_node_inds].set(
        voltages[internal_node_inds] + delta_t * constant_terms[internal_node_inds]
    )

    # Solve the tridiagonal system.
    diags, lowers, solves = stone_triang_upper(lowers, diags, uppers, solves)
    solves = stone_backsub_lower(solves, lowers, diags)

    return solves
