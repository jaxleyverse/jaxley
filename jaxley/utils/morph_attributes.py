# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd


def morph_attrs_from_xyzr(
    xyzr: np.ndarray,
    min_radius: Optional[float],
    ncomp: int,
) -> float:
    """Return radius, area, volume, and resistive loads of a comp given its SWC xyzr.

    Args:
        radius_fns: Functions which, given compartment locations return the radius.
        branch_indices: The indices of the branches for which to return the radiuses.
        min_radius: If passed, the radiuses are clipped to be at least as large.
        ncomp: The number of compartments that every branch is discretized into.
    """
    # Extract 3D coordinates and radii
    positions = xyzr[:, :3]  # shape (N, 3): x, y, z
    radii = xyzr[:, 3]  # shape (N,): radius at each point

    if len(xyzr) > 1:
        # Compute Euclidean distances between consecutive points
        position_deltas = np.diff(positions, axis=0)  # shape (N-1, 3)
        segment_lengths = np.linalg.norm(position_deltas, axis=1)  # shape (N-1,)

        avg_radius = swc_radius(segment_lengths, radii)
        total_surface_area = swc_area(segment_lengths, radii)
        total_volume = swc_volume(segment_lengths, radii)

        # Finally, we compute the input and output resistive loads. For this, we first
        # have to split the xyzr into two: the ones on the left half of the
        # compartment, and the ones on the right half.
        xyzr_split = split_xyzr_into_equal_length_segments(xyzr, 2)
        resistive_load = []
        for xyzr_half in xyzr_split:
            # Extract 3D coordinates and radii
            positions_half = xyzr_half[:, :3]  # shape (N, 3): x, y, z
            radii_half = xyzr_half[:, 3]  # shape (N,): radius at each point

            # Compute Euclidean distances between consecutive points
            position_deltas_half = np.diff(positions_half, axis=0)  # shape (N-1, 3)
            segment_lengths_half = np.linalg.norm(
                position_deltas_half, axis=1
            )  # shape (N-1,)
            resistive_load.append(swc_resistive_load(segment_lengths_half, radii_half))
    else:
        avg_radius = radii.mean()
        total_surface_area = 4 * np.pi * radii[0] ** 2 / ncomp  # Surface of a sphere.
        total_volume = 4 / 3 * np.pi * radii[0] ** 3 / ncomp  # Volume of a sphere.

        # Resistive load.
        # For single point, the total length of a branch is: length = radius.
        # Thus, the length of a compartment is `radius/ncomp`.
        length = radii[0] / ncomp
        resistive_load = [length / radii[0] ** 2 / np.pi] * 2

    if min_radius is None:
        assert (
            avg_radius > 0.0
        ), "Radius 0.0 in SWC file. Set `read_swc(..., min_radius=...)`."
    else:
        avg_radius = (
            min_radius
            if (avg_radius < min_radius or np.isnan(avg_radius))
            else avg_radius
        )
    return avg_radius, total_surface_area, total_volume, *resistive_load


def split_xyzr_into_equal_length_segments(
    xyzr: np.ndarray, ncomp: int
) -> List[np.ndarray]:
    """Split xyzr into equal-length segments by inserting interpolated points as needed.

    This function was written by ChatGPT, based on the prompt:
    ```I have an array of shape 100x3. The 3 indicate x, y, z coordinates. I want to
    split this array into 4 segments, each with equal euclidean length. To have
    euclidean length exactly equal, I would like to insert additional points into
    the 100x3 array (to make it length 100 + 4 segments - 1). These points should be
    linear interpolation of neighboring points. In the final split array, the newly
    inserted nodes should be the last point of one segment and the first point of
    another segment.```

    Args:
        points: Array of 3D coordinates representing a path.
        num_segments: Number of segments to split the path into.

    Returns:
        A list of `num_segments` arrays, each containing the 3D coordinates
        of one segment. The segments have (approximately) equal Euclidean
        length, and split points are interpolated between original points.
    """
    if len(xyzr) == 1:
        return [xyzr] * ncomp

    # Compute distances between consecutive points
    xyz = xyzr[:, :3]

    # Compute distances and cumulative distances
    deltas = np.diff(xyz, axis=0)
    dists = np.linalg.norm(deltas, axis=1)
    cum_dists = np.concatenate([[0], np.cumsum(dists)])
    total_length = cum_dists[-1]

    # Target cumulative distances where we want to split
    target_dists = np.linspace(0, total_length, ncomp + 1)

    # Find insertion indices and interpolation factors
    idxs = np.searchsorted(cum_dists, target_dists, side="right") - 1
    idxs = np.clip(idxs, 0, len(xyz) - 2)  # Ensure valid indices
    local_dist = target_dists - cum_dists[idxs]

    # When two traced SWC are right on top of each other, their dists=0. Then, when
    # these points lie exactly at the point where the xyz is supposed to be split,
    # we get `segment_lens=0`, which causes frac to be infinity.
    dists = np.where(dists < 1e-14, 1e-14, dists)
    segment_lens = dists[idxs]
    frac = (local_dist / segment_lens)[:, None]  # shape (n, 1)

    # Interpolate split points
    split_points = xyzr[idxs] + frac * (xyzr[idxs + 1] - xyzr[idxs])

    # Build final list of points with inserted nodes
    all_points = [split_points[0]]
    compartment_xyzrs = []

    for i in range(1, len(split_points)):
        # Collect original points between splits.
        mask = (cum_dists > target_dists[i - 1]) & (cum_dists < target_dists[i])
        between_points = xyzr[mask]
        segment = np.vstack([all_points[-1], *between_points, split_points[i]])
        compartment_xyzrs.append(segment)
        all_points.append(split_points[i])
    return compartment_xyzrs


def swc_radius(lengths: np.ndarray, radii: np.ndarray) -> np.ndarray:
    r"""Return the radius of a branch given its SWC coordinates.

    This function computes the average radius, weighted by the length between
    two SWC points.

    Args:
        lengths: Array of shape `(N-1)`, indicating the spacing between all SWC
            points within the branch.
        radii: Array of shape `(N)`, indicating the radius of each SWC point.

    Returns:
        A radius as a scalar value."""
    radius_weights = np.zeros(len(lengths) + 1)
    radius_weights[1:] += lengths
    radius_weights[:-1] += lengths
    radius_weights /= np.sum(radius_weights)
    return np.sum(radii * radius_weights)


def swc_area(lengths: np.ndarray, radii: np.ndarray) -> np.ndarray:
    r"""Return the surface area of a compartment given its SWC coordinates.

    This function makes a truncated cone approximation between any two SWC points
    and then computes the surface area.

    Args:
        lengths: Array of shape `(N-1)`, indicating the spacing between all SWC
            points within the compartment.
        radii: Array of shape `(N)`, indicating the radius of each SWC point.

    Returns:
        A membrane surface area as a scalar value."""
    radius_start = radii[:-1]
    radius_end = radii[1:]
    delta_radii = radius_end - radius_start
    slant_lengths = np.sqrt(delta_radii**2 + lengths**2)
    frustum_surface_areas = np.pi * (radius_start + radius_end) * slant_lengths
    return np.sum(frustum_surface_areas)


def swc_volume(lengths: np.ndarray, radii: np.ndarray) -> np.ndarray:
    r"""Return the volume of a compartment given its SWC coordinates.

    This function makes a truncated cone approximation between any two SWC points
    and then computes the volume. This function is used only for ion diffusion.

    Args:
        lengths: Array of shape `(N-1)`, indicating the spacing between all SWC
            points within the compartment.
        radii: Array of shape `(N)`, indicating the radius of each SWC point.

    Returns:
        A volume as a scalar value."""
    radius_start = radii[:-1]
    radius_end = radii[1:]
    volume = (
        (np.pi / 3)
        * lengths
        * (radius_start**2 + radius_start * radius_end + radius_end**2)
    )
    return np.sum(volume)


def swc_resistive_load(lengths: np.ndarray, radii: np.ndarray) -> np.ndarray:
    r"""Return the resistive load of a compartment given its SWC coordinates.

    The resistive load is defined as the integral over :math:`1/(\pi r^2)`, i.e.,

    .. math::

        r_l = \frac{1}{\pi} \int \frac{1}{r^2} \, dl

    As an example, if the radius is constant, then we obtain :math:`l / r^2 / \pi`,
    which corresponds exactly to the length divided by the cross section.

    This function makes a truncated cone approximation between any two SWC points
    and then computes the resistive load with the equation above. In the function
    `compute_axial_conductances()`, the resistive load gets multiplied by the
    axial resistivity (r_a, in :math:`\ohm` cm) to obtain the resistance between
    compartments.

    Args:
        lengths: Array of shape `(N-1)`, indicating the spacing between all SWC
            points within the compartment.
        radii: Array of shape `(N)`, indicating the radius of each SWC point.

    Returns:
        A resistive load as a scalar value."""
    lengths = np.asarray(lengths)
    radius_start = np.asarray(radii[:-1])
    radius_end = np.asarray(radii[1:])
    delta_radius = radius_end - radius_start

    segment_integrals = np.empty_like(lengths)

    # Segments with constant radius.
    is_constant_radius = np.isclose(delta_radius, 0)
    segment_integrals[is_constant_radius] = (
        lengths[is_constant_radius] / radius_start[is_constant_radius] ** 2
    )

    # Segments with varying radius (truncated cone).
    is_varying_radius = ~is_constant_radius
    segment_integrals[is_varying_radius] = (
        lengths[is_varying_radius]
        / delta_radius[is_varying_radius]
        * (1 / radius_start[is_varying_radius] - 1 / radius_end[is_varying_radius])
    )

    return np.sum(segment_integrals) / jnp.pi


def cylinder_area(length: jnp.ndarray, radius: jnp.ndarray) -> jnp.ndarray:
    r"""Return the surface area of a cylindric compartment, given its length and radius.

    Args:
        lengths: The lengths of M cylindric compartments, shape (M,).
        radii: The radii of M cylindric compartments, shape (M,).

    Returns:
        The membrane surface area of each M cylindric compartments, shape (M,)."""
    return 2.0 * jnp.pi * radius * length


def cylinder_volume(length: jnp.ndarray, radius: jnp.ndarray) -> jnp.ndarray:
    r"""Return the volume of a cylindric compartment, given its length and radius.

    The resistive load is defined as the integral over :math:`1/(\pi r^2)`, i.e.,

    .. math::

        r_l = \frac{1}{\pi} \int \frac{1}{r^2} \, dl

    For a cylinder, the radius is constant, so we obtain :math:`l / r^2 / \pi`.
    This corresponds exactly to the length divided by the cross section.

    Args:
        lengths: The lengths of M cylindric compartments, shape (M,).
        radii: The radii of M cylindric compartments, shape (M,).

    Returns:
        The volume of each M cylindric compartments, shape (M,)."""
    return length * radius**2 * jnp.pi


def cylinder_resistive_load(length: jnp.ndarray, radius: jnp.ndarray) -> jnp.ndarray:
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
    params: Dict[str, jnp.ndarray],
    diffusion_states: List[str],
) -> Dict[str, jnp.ndarray]:
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
