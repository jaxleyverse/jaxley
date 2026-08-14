# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
from typing import Tuple
from warnings import warn

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array
from jax.typing import ArrayLike

from jaxley.utils.misc_utils import cumsum_leading_zero


def _cumulative_cone_props(
    ls: np.ndarray, rs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Cumulative frustum integrals along a path, evaluated at each of the points.

    The path is a chain of truncated cones. Each returned array starts at 0 and gives
    the integral from `ls[0]` to `ls[i]` of:

    - `radius`: :math:`\int r \, dl`, the length-weighted radius.
    - `area`: the surface area of the frustums.
    - `volume`: :math:`\int \pi r^2 \, dl`.
    - `load`: :math:`\frac{1}{\pi} \int r^{-2} \, dl`, the resistive load.

    Being cumulative, the value over any sub-path is the difference of its two ends.

    Args:
        ls: Cumulative path length of each point, shape `(N,)`, non-decreasing.
        rs: Radius of each point, shape `(N,)`.

    Returns:
        4 arrays of shape `(N,)`: the cumulative radius, area, volume and resistive
        load.
    """
    dl = np.diff(ls)
    r1, r2 = rs[:-1], rs[1:]
    dr = r2 - r1

    radius = (r1 + r2) / 2 * dl
    area = np.pi * (r1 + r2) * np.sqrt(dl**2 + dr**2)
    volume = np.pi * dl / 3 * (r1**2 + r1 * r2 + r2**2)

    load = np.empty_like(dl)
    is_constant = np.isclose(dr, 0)
    load[is_constant] = dl[is_constant] / r1[is_constant] ** 2  # cylinder
    load[~is_constant] = (  # truncated cone
        dl[~is_constant]
        / dr[~is_constant]
        * (1 / r1[~is_constant] - 1 / r2[~is_constant])
    )
    load = load / np.pi

    return tuple(cumsum_leading_zero(x) for x in (radius, area, volume, load))


def compute_cone_props(
    ls: np.ndarray, rs: np.ndarray, bounds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Frustum properties of the compartments delimited by `bounds` along a path.

    The path is radii `rs` at path lengths `ls`, treated as a chain of truncated cones.
    Every compartment boundary and midpoint is added to the path, so that each returned
    quantity is a difference of the cumulative integrals of `_cumulative_cone_props()`.

    Args:
        ls: Cumulative path length of each point, shape `(N,)`.
        rs: Radius of each point, shape `(N,)`.
        bounds: The `ncomp + 1` compartment boundaries along the path.

    Returns:
        Five arrays of shape `(ncomp,)`: the length-weighted average radius, the surface
        area, the volume, and the resistive loads of the first and second half of each
        compartment (`resistive_load_in` and `resistive_load_out`).
    """
    bounds = np.asarray(bounds, dtype=float)
    assert np.all(np.diff(bounds) > 0), "Compartment bounds must be increasing."

    # Knots: every compartment boundary and midpoint.
    mids = (bounds[:-1] + bounds[1:]) / 2
    knots = np.empty(len(bounds) + len(mids))
    knots[0::2], knots[1::2] = bounds, mids

    # Add knots to the path, unless a point already sits on it. If the radius
    # steps between them, the second segment would count the step's area twice.
    at = np.searchsorted(ls, knots)
    exists = np.zeros(len(knots), dtype=bool)
    inside = at < len(ls)
    atol = 1e-12 * max(ls[-1], 1.0)  # absolute, so it scales with the path
    exists[inside] = np.abs(ls[at[inside]] - knots[inside]) <= atol

    insert = ~exists
    ls_at_knots = np.insert(ls, at[insert], knots[insert])
    rs_at_knots = np.insert(rs, at[insert], np.interp(knots[insert], ls, rs))
    # Where each knot ended up: its insertion point, shifted by the knots added before it.
    pos = at + np.cumsum(insert) - insert

    cum_radius, cum_area, cum_volume, cum_load = _cumulative_cone_props(
        ls_at_knots, rs_at_knots
    )
    start, mid, end = pos[0:-2:2], pos[1::2], pos[2::2]

    return (
        (cum_radius[end] - cum_radius[start]) / (ls_at_knots[end] - ls_at_knots[start]),
        cum_area[end] - cum_area[start],
        cum_volume[end] - cum_volume[start],
        cum_load[mid] - cum_load[start],
        cum_load[end] - cum_load[mid],
    )


COMP_ATTRS = [
    "length",
    "x",
    "y",
    "z",
    "radius",
    "area",
    "volume",
    "resistive_load_in",
    "resistive_load_out",
]


def compartmentalize_branch(
    branch_nodes: pd.DataFrame,
    ncomp: int,
) -> pd.DataFrame:
    """Interpolate or integrate node attributes along branch.

    Takes a dataframe with nodes (index) and node attributes (columns) and returns a
    dataframe of compartments and compartment attributes. Compartments are spaced at
    equidistant points along the branch. Node attributes, like radius are linearly
    interpolated along its length.

    Example: 4 compartments | edges = - | nodes = o | comp_nodes = x
    o-----------o----------o---o---o---o--------o
    o-------x---o----x-----o--xo---o---ox-------o

    Args:
        branch_nodes: DataFrame of node attributes for nodes in a branch.
            needs to include morph attributes `x`, `y`, `z`, `radius`.
        ncomp: Number of compartments per branch.

    Returns:
        DataFrame of compartments and compartment attributes, the columns being
        `COMP_ATTRS`.
    """
    for attr in set(["x", "y", "z", "radius"]):
        assert attr in branch_nodes.columns, f"Branch nodes must contain '{attr}'."

    xyzr = branch_nodes[["x", "y", "z", "radius"]].to_numpy(dtype=float)
    return pd.DataFrame(compartmentalize(xyzr, ncomp), columns=COMP_ATTRS)


def compartmentalize(xyzr: np.ndarray, ncomp: int) -> dict:
    """Split a branch into `ncomp` compartments. The array core of the function above.

    Kept apart from `compartmentalize_branch()` so that callers which run it once per branch
    do not pay for a DataFrame going in and another coming out each time.

    Args:
        xyzr: The branch's traced points, shape (N, 4), ordered along the branch.
        ncomp: Number of compartments to split the branch into.

    Returns:
        One array of length `ncomp + 2` per entry of `COMP_ATTRS`. The first and last entry
        of each are the branch's two ends, which have no attributes other than a position.
    """
    # NEURON's `Import3d` builds a cylinder of length 2*r along +x for a single-point soma.
    if len(xyzr) == 1:
        xyzr = np.repeat(xyzr, 2, axis=0)
        xyzr[:, 0] += [-xyzr[0, 3], xyzr[0, 3]]

    branch_xyz, rs = xyzr[:, :3], xyzr[:, 3]
    edge_lens = np.linalg.norm(np.diff(branch_xyz, axis=0), axis=1)
    ls = cumsum_leading_zero(edge_lens)  # path length
    branch_len = ls[-1]

    if branch_len < 1e-8:
        warn(
            "Found a branch with length 0. To avoid NaN while integrating the "
            "ODE, we capped this length to 0.1 um. The underlying cause for the "
            "branch with length 0 is likely a strange SWC file. The "
            "most common reason for this is that the SWC contains a soma "
            "traced by a single point, and a dendrite that connects to the soma "
            "has no further child nodes."
        )
        branch_len = 0.1  # cap, as promised by the warning above
    comp_len = branch_len / ncomp

    # Create node indices and attributes for branch-tips/branchpoints and comps
    # is_comp, comp_len, comp_id, x, y, z, r, area, volume, res_in, res_out
    cone_prop_cols = COMP_ATTRS[4:]
    n_rows = ncomp + 2

    is_comp = np.zeros(n_rows, dtype=bool)
    is_comp[1:-1] = True

    data = {col: np.full(n_rows, np.nan) for col in COMP_ATTRS}
    data["length"][is_comp] = comp_len

    # Interpolate along the branch. The two branch ends bracket the compartment centers,
    # so the tip/branchpoint rows get the branch's end coordinates.
    comp_centers = np.linspace(comp_len / 2, branch_len - comp_len / 2, ncomp)
    comp_centers = np.array([0, *comp_centers, branch_len])

    for i, col in enumerate(["x", "y", "z"]):
        data[col] = np.interp(comp_centers, ls, branch_xyz[:, i])

    # radius, area, volume, resistive_load_in, resistive_load_out of every compartment
    comp_ends = np.linspace(0, branch_len, ncomp + 1)
    for col, values in zip(cone_prop_cols, compute_cone_props(ls, rs, comp_ends)):
        data[col][is_comp] = values

    return data


def cylinder_area(length: ArrayLike, radius: ArrayLike) -> Array:
    r"""Return the surface area of a cylindric compartment, given its length and radius.

    Args:
        lengths: The lengths of M cylindric compartments, shape (M,).
        radii: The radii of M cylindric compartments, shape (M,).

    Returns:
        The membrane surface area of each M cylindric compartments, shape (M,)."""
    return 2.0 * jnp.pi * radius * length


def cylinder_volume(length: ArrayLike, radius: ArrayLike) -> Array:
    r"""Return the volume of a cylindric compartment, given its length and radius.

    The radius is constant along a cylinder, so the volume is the cross section times the
    length, :math:`\pi r^2 l`.

    Args:
        lengths: The lengths of M cylindric compartments, shape (M,).
        radii: The radii of M cylindric compartments, shape (M,).

    Returns:
        The volume of each M cylindric compartments, shape (M,)."""
    return length * radius**2 * jnp.pi


def cylinder_resistive_load(length: ArrayLike, radius: ArrayLike) -> Array:
    r"""Return the resistive load of a cylindric compartment, given length and radius.

    The resistive load is defined as the integral over :math:`1/(\pi r^2)`, i.e.,

    .. math::

        r_l = \frac{1}{\pi} \int \frac{1}{r^2} \, dl

    For a cylinder, the radius is constant, so we obtain :math:`l / r^2 / \pi`.
    This corresponds exactly to the length divided by the cross section.

    Args:
        lengths: The lengths of M cylindric compartments, shape (M,).
        radii: The radii of M cylindric compartments, shape (M,).

    Returns:
        The resistive load of each M cylindric compartments, shape (M,)."""
    return length / radius**2 / jnp.pi


def compute_axial_conductances(
    comp_edges: pd.DataFrame,
    params: dict[str, Array],
    diffusion_states: list[str],
) -> dict[str, Array]:
    r"""Given `comp_edges`, radius, length, r_a, cm, compute the axial conductances.

    Note that the resulting axial conductances will already by divided by the
    capacitance `cm`.
    """
    ordered_conds = jnp.zeros((1 + len(diffusion_states), len(comp_edges)))

    axial_conds = jnp.stack(
        [1 / params["axial_resistivity"]]
        + [params[f"axial_diffusion_{d}"] for d in diffusion_states]
    )
    # These are still _compartment_ properties.
    comp_source_r_a = params["resistive_load_out"] / axial_conds
    comp_sink_r_a = params["resistive_load_in"] / axial_conds

    # comp_r_a has shape (N, 2, num_comps). Here, N is the number of states that are
    # diffused (including voltage).
    comp_r_a = jnp.stack([comp_sink_r_a, comp_source_r_a], axis=1)

    # `Compartment-to-compartment` (c2c) axial coupling conductances.
    condition = comp_edges["type"].to_numpy() == 0
    source_comp_inds = np.asarray(comp_edges[condition]["source"].to_list()).astype(int)
    sink_comp_inds = np.asarray(comp_edges[condition]["sink"].to_list()).astype(int)
    ordered_edge = np.asarray(comp_edges[condition]["ordered"].to_list())

    # Now we compute c2c _comp_edges_ properties.
    if len(sink_comp_inds) > 0:
        r_a_of_sources = comp_r_a[:, ordered_edge, source_comp_inds]
        r_a_of_sinks = comp_r_a[:, 1 - ordered_edge, sink_comp_inds]
        r_a = r_a_of_sources + r_a_of_sinks

        # Voltage diffusion.
        conds_c2c = 1 / r_a[:1] / params["area"][sink_comp_inds]

        # We only divide the axial _voltage_ conductances by the
        # capacitance, _not_ the axial conductances of the diffusing ions.
        conds_c2c /= params["capacitance"][sink_comp_inds]
        # Multiply by 10**7 to convert (S / cm / um) -> (mS / cm^2).
        conds_c2c *= 10**7

        # For ion diffusion, we have to divide by the volume, not the surface area.
        conds_diffusion = 1 / r_a[1:] / params["volume"][sink_comp_inds]
        conds_c2c = jnp.concatenate([conds_c2c, conds_diffusion], axis=0)

        inds = jnp.asarray(comp_edges[condition].index)
        ordered_conds = ordered_conds.at[:, inds].set(conds_c2c)

    # `branchpoint-to-compartment` (bp2c) axial coupling conductances.
    condition = comp_edges["type"].isin([1, 2])
    sink_comp_inds = np.asarray(comp_edges[condition]["sink"].to_list()).astype(int)
    ordered_edge = np.asarray(comp_edges[condition]["ordered"].to_list())

    if len(sink_comp_inds) > 0:
        r_a = comp_r_a[:, 1 - ordered_edge, sink_comp_inds]

        # Voltage diffusion.
        conds_bp2c = 1 / r_a[:1] / params["area"][sink_comp_inds]
        conds_bp2c /= params["capacitance"][sink_comp_inds]
        # Multiply by 10**7 to convert (S / cm / um) -> (mS / cm^2).
        conds_bp2c *= 10**7

        # For ion diffusion, we have to divide by the volume, not the surface area.
        conds_diffusion = 1 / r_a[1:] / params["volume"][sink_comp_inds]
        conds_bp2c = jnp.concatenate([conds_bp2c, conds_diffusion], axis=0)

        inds = jnp.asarray(comp_edges[condition].index)
        ordered_conds = ordered_conds.at[:, inds].set(conds_bp2c)

    # `compartment-to-branchpoint` (c2bp) axial coupling conductances.
    condition = comp_edges["type"].isin([3, 4])
    source_comp_inds = np.asarray(comp_edges[condition]["source"].to_list()).astype(int)

    comp_source_g_a = 1 / params["resistive_load_out"] * axial_conds
    comp_sink_g_a = 1 / params["resistive_load_in"] * axial_conds
    comp_g_a = jnp.stack([comp_sink_g_a, comp_source_g_a], axis=1)

    if len(source_comp_inds) > 0:
        conds_c2bp = comp_g_a[:, 1 - ordered_edge, source_comp_inds]
        inds = jnp.asarray(comp_edges[condition].index)
        ordered_conds = ordered_conds.at[:, inds].set(conds_c2bp)

    # Reformat the conductances along the key of the quantity being diffused.
    ordered_conds_as_dict = {}
    for i, key in enumerate(["v"] + diffusion_states):
        ordered_conds_as_dict[key] = ordered_conds[i]

    return ordered_conds_as_dict
