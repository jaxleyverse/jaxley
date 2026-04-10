# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental.sparse.linalg import spsolve as jax_spsolve
from jax.lax import fori_loop
from jax.typing import ArrayLike
from tridiax.stone import stone_backsub_lower, stone_triang_upper

from jaxley.solver_gate import exponential_euler


def _pad_comp_edges(comp_edges) -> np.ndarray:
    """Convert grouped DHS edges to a dense integer array padded with `-1`.

    `node_order_grouped` is often stored as a ragged list of arrays, one per depth
    level. Padding with `-1` is safe because the voltage solve appends a spurious
    compartment at the end of every solve vector, and `-1` indexes that no-op slot.
    """
    if isinstance(comp_edges, np.ndarray) and comp_edges.dtype != object:
        return comp_edges.astype(np.int32, copy=False)

    comp_edges = list(comp_edges)
    if len(comp_edges) == 0:
        return np.empty((0, 0, 2), dtype=np.int32)

    level_arrays = [np.asarray(level, dtype=np.int32) for level in comp_edges]
    max_width = max(level.shape[0] for level in level_arrays)
    padded = np.full((len(level_arrays), max_width, 2), -1, dtype=np.int32)

    for idx, level in enumerate(level_arrays):
        if level.ndim != 2 or level.shape[1] != 2:
            raise ValueError(
                "Expected each DHS level to have shape (num_edges_in_level, 2)."
            )
        padded[idx, : level.shape[0], :] = level

    return padded


def _make_dhs_solve(solve_indexer, optimize_for_gpu, n_nodes):
    """Create a DHS solve function with custom JVP for efficient differentiation.

    The tridiagonal solve A x = b has JVP: dx = A^{-1} (db - dA x). This means the
    tangent is itself a solve with the same matrix A but a different RHS. By using a
    custom_jvp, we avoid JAX having to differentiate through the O(n)-step fori_loop,
    which causes O(n^2) memory traffic in the backward pass.

    JAX automatically derives the transpose (VJP) from the custom JVP rule, so this
    works for both forward-mode and reverse-mode differentiation.
    """
    ordered_comp_edges = solve_indexer["node_order_grouped"]
    flipped_comp_edges = list(reversed(ordered_comp_edges))
    all_children = np.asarray(solve_indexer["all_children"], dtype=np.int32)
    all_parents = np.asarray(solve_indexer["all_parents"], dtype=np.int32)

    steps = len(flipped_comp_edges)

    ordered_comp_edges_np = _pad_comp_edges(ordered_comp_edges)
    flipped_comp_edges_np = _pad_comp_edges(flipped_comp_edges)

    def _raw_solve(diags, lowers, uppers, solves):
        """Solve the tree-structured linear system (no custom JVP)."""
        if not optimize_for_gpu:
            init = (diags, solves, lowers, uppers, flipped_comp_edges_np)
            diags_out, solves_out, _, _, _ = fori_loop(
                0, steps, _comp_based_triang, init
            )

            lowers_norm = lowers / diags_out
            solves_norm = solves_out / diags_out
            diags_out = jnp.ones_like(solves_norm)
            init = (solves_norm, lowers_norm, ordered_comp_edges_np)
            solves_out, _, _ = fori_loop(0, steps, _comp_based_backsub, init)

            return solves_out / diags_out
        else:
            d, s = diags, solves
            for i in range(steps):
                d, s, _, _, _ = _comp_based_triang(
                    i, (d, s, lowers, uppers, flipped_comp_edges_np)
                )

            d, s = _comp_based_backsub_recursive_doubling(
                d, s, lowers, steps, solve_indexer["parent_lookup"]
            )
            return s / d

    @jax.custom_jvp
    def _solve(diags, lowers, uppers, solves):
        return _raw_solve(diags, lowers, uppers, solves)

    @_solve.defjvp
    def _solve_jvp(primals, tangents):
        diags, lowers, uppers, solves = primals
        d_diags, d_lowers, d_uppers, d_solves = tangents

        # Primal output: x = A^{-1} b
        x = _raw_solve(diags, lowers, uppers, solves)

        # Compute dA @ x. For each edge (child, parent), the matrix entries are:
        #   A[parent, child] = uppers[child]
        #   A[child, parent] = lowers[child]
        # (this is consistent with the triangulation multiplier using `uppers`).
        #
        # Therefore:
        #   (dA @ x)[parent] += d_uppers[child] * x[child]
        #   (dA @ x)[child]  += d_lowers[child] * x[parent]
        dA_x = d_diags * x
        dA_x = dA_x.at[all_parents].add(d_uppers[all_children] * x[all_children])
        dA_x = dA_x.at[all_children].add(d_lowers[all_children] * x[all_parents])

        # JVP: dx = A^{-1} (db - dA @ x)
        new_rhs = d_solves - dA_x
        dx = _raw_solve(diags, lowers, uppers, new_rhs)

        return x, dx

    return _solve


# Cache key for storing the compiled solve function directly on the solve_indexer
# dict. Using a tuple key avoids collisions with the string keys it already uses.
# The cached function is automatically discarded when the owning Module (and its
# _dhs_solve_indexer dict) is garbage-collected, and naturally invalidated when
# _init_solver_jaxley_dhs_solve() creates a fresh dict.
_SOLVE_FN_KEY = "_cached_solve_fn"


def _get_dhs_solve(solve_indexer: dict, optimize_for_gpu: bool, n_nodes: int):
    cache_key = (_SOLVE_FN_KEY, optimize_for_gpu, n_nodes)
    solve_fn = solve_indexer.get(cache_key)
    if solve_fn is None:
        solve_fn = _make_dhs_solve(solve_indexer, optimize_for_gpu, n_nodes)
        solve_indexer[cache_key] = solve_fn
    return solve_fn


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

        # Add a spurious compartment that is modified by the masking.
        diags = jnp.concatenate([diags, jnp.asarray([1.0])])
        solves = jnp.concatenate([solves, jnp.asarray([0.0])])
        uppers = jnp.concatenate([uppers, jnp.asarray([0.0])])
        lowers = jnp.concatenate([lowers, jnp.asarray([0.0])])

        # Get or create the solve function with custom JVP.
        dhs_solve = _get_dhs_solve(solve_indexer, optimize_for_gpu, int(n_nodes))

        # Solve the voltage equations with efficient custom JVP.
        solution = dhs_solve(diags, lowers, uppers, solves)

        # Remove the spurious compartment.
        solution = solution[:-1]
    else:
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
    diags: ArrayLike,
    solves: ArrayLike,
    lowers: ArrayLike,
    steps: int,
    parent_lookup: np.ndarray,
) -> tuple[Array, Array]:
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

    num_recursive_steps = int(np.ceil(np.log2(steps + 1))) if steps > 0 else 0
    parent_jump = jnp.asarray(parent_lookup, dtype=jnp.int32)

    # Only O(log2(steps)) iterations; unrolling these often recovers GPU runtime
    # without significantly increasing compile time.
    for _ in range(num_recursive_steps):
        solve_effect = lower_effect * solve_effect[parent_jump] + solve_effect
        lower_effect = lower_effect * lower_effect[parent_jump]
        parent_jump = parent_jump[parent_jump]

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
    voltages: ArrayLike,
    voltage_terms: ArrayLike,
    constant_terms: ArrayLike,
    axial_conductances: ArrayLike,
    sinks,
    sources,
    types,
    n_nodes,
    delta_t: float,
) -> Array:
    """Solve one timestep of branched nerve equations with explicit (forward) Euler."""
    update = _voltage_vectorfield(
        voltages,
        voltage_terms,
        constant_terms,
        axial_conductances,
        sinks,
        sources,
        types,
        n_nodes,
    )
    new_voltates = voltages + delta_t * update
    return new_voltates


def _voltage_vectorfield(
    voltages: ArrayLike,
    voltage_terms: ArrayLike,
    constant_terms: ArrayLike,
    axial_conductances: ArrayLike,
    sinks,
    sources,
    types,
    n_nodes,
) -> Array:
    """Evaluate the vectorfield of the nerve equation."""
    # Membrane current update.
    vecfield = -voltage_terms * voltages + constant_terms

    # Compute branchpoint voltages. The voltages at branchpoints are a weighted
    # average of all neighboring compartments: vBP = (g1 v1 + g2 v2) / (g1 + g2)
    condition = np.isin(types, [3, 4])
    if np.sum(condition) > 0:
        sink_comp_inds = sinks[condition]
        source_comp_inds = sources[condition]
        # Set set the voltages at branchpoints to zero.
        voltages = voltages.at[sink_comp_inds].set(0.0)
        # Compute all terms that influence the branchpoints: g1 v1 + g2 v2
        voltages = voltages.at[sink_comp_inds].add(
            axial_conductances[condition] * voltages[source_comp_inds]
        )
        # Compute the normalizer for the branchpoint: (g1 + g2)
        # Initialize at one such that we can also just divide the compartment voltages
        # by it.
        summed_axials = jnp.ones(n_nodes)
        summed_axials = summed_axials.at[sink_comp_inds].set(0.0)
        summed_axials = summed_axials.at[sink_comp_inds].add(
            axial_conductances[condition]
        )
        voltages = voltages / summed_axials

    # Update compartment voltages with the impact from other compartments and from
    # branchpoints.
    condition = np.isin(types, [0, 1, 2])
    if np.sum(condition) > 0:
        sink_comp_inds = sinks[condition]
        source_comp_inds = sources[condition]
        vecfield = vecfield.at[sinks].add(
            (voltages[sources] - voltages[sinks]) * axial_conductances
        )
    return vecfield


def step_voltage_exponential(
    voltages: ArrayLike,
    voltage_terms: ArrayLike,
    constant_terms: ArrayLike,
    axial_conductances: ArrayLike,
    matrix_exponential,
    internal_node_inds,
    delta_t: float,
) -> Array:
    """Solve one timestep of branched nerve equations with exponential Euler.

    This performs Lie-Trotter splitting for the voltages:

    dv/dt = G * v + g_channel * (E - V)

    v_{t+1} = exp(dt * G) * exp_euler(channel_terms, v_{t})

    I.e., with Lie-Trotter splitting, we first perform an update of the term related to
    the channels, and then update the 'diffusive' term G * v.

    Args:
        voltages: The voltages (including branchpoint voltages).
        voltage_terms: The linear part of the channels terms, i.e., g_channel.
        constant_terms: The constant part of the channel terms, i.e., g_channel * E
        axial_conductances: Unused (because we use the `matrix_exponential`).
        matrix_exponential: The matrix exp(G * dt).
        internal_node_inds: Indexes into all compartments.
        delta_t: Time step.

    Returns:
        The updated voltages (including branchpoint voltages).
    """
    # Update of the channels. This requires a `jnp.where` because: some compartments
    # might not have _any_ channels. In that case, their `voltate_terms = 0`, which
    # causes tau = inf. This if-case solves this.
    tau_is_inf = voltage_terms == 0
    v_tau_not_inf = exponential_euler(
        voltages, delta_t, constant_terms / voltage_terms, 1.0 / voltage_terms
    )
    v_tau_is_inf = voltages + constant_terms * delta_t
    voltages = jnp.where(tau_is_inf, v_tau_is_inf, v_tau_not_inf)

    # Update of the diffusion matrix.
    return voltages.at[internal_node_inds].set(
        matrix_exponential @ voltages[internal_node_inds]
    )


def step_voltage_implicit_with_stone(
    voltages: ArrayLike,
    voltage_terms: ArrayLike,
    constant_terms: ArrayLike,
    axial_conductances: ArrayLike,
    internal_node_inds: ArrayLike,
    n_nodes: int,
    sinks: ArrayLike,
    sources: ArrayLike,
    types: ArrayLike,
    delta_t: float,
):
    """Solve one timestep of branched nerve equations with implicit (backward) Euler."""
    if np.sum(np.isin(types, [1, 2, 3, 4])) > 0:
        raise NotImplementedError(
            f"The stone solver is not implemented for branched morphologies."
        )
    axial_conductances = delta_t * axial_conductances

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
