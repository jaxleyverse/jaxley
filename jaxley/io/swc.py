# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from copy import copy
from typing import Callable, List, Optional, Tuple
from warnings import warn

import jax.numpy as jnp
import numpy as np

from jaxley.modules import Branch, Cell, Compartment
from jaxley.utils.cell_utils import (
    _build_parents,
    _compute_pathlengths,
    _radius_generating_fns,
    _split_into_branches_and_sort,
    build_radiuses_from_xyzr,
)


def swc_to_jaxley(
    fname: str,
    max_branch_len: float = 100.0,
    sort: bool = True,
    num_lines: Optional[int] = None,
) -> Tuple[List[int], List[float], List[Callable], List[float], List[np.ndarray]]:
    """Read an SWC file and bring morphology into `jaxley` compatible formats.

    Args:
        fname: Path to swc file.
        max_branch_len: Maximal length of one branch. If a branch exceeds this length,
            it is split into equal parts such that each subbranch is below
            `max_branch_len`.
        num_lines: Number of lines of the SWC file to read.
    """
    content = np.loadtxt(fname)[:num_lines]
    types = content[:, 1]
    is_single_point_soma = types[0] == 1 and types[1] != 1

    if is_single_point_soma:
        # Warn here, but the conversion of the length happens in `_compute_pathlengths`.
        warn(
            "Found a soma which consists of a single traced point. `Jaxley` "
            "interprets this soma as a spherical compartment with radius "
            "specified in the SWC file, i.e. with surface area 4*pi*r*r."
        )
    sorted_branches, types = _split_into_branches_and_sort(
        content,
        max_branch_len=max_branch_len,
        is_single_point_soma=is_single_point_soma,
        sort=sort,
    )

    parents = _build_parents(sorted_branches)
    each_length = _compute_pathlengths(
        sorted_branches, content[:, 1:6], is_single_point_soma=is_single_point_soma
    )
    pathlengths = [np.sum(length_traced) for length_traced in each_length]
    for i, pathlen in enumerate(pathlengths):
        if pathlen == 0.0:
            warn("Found a segment with length 0. Clipping it to 1.0")
            pathlengths[i] = 1.0
    radius_fns = _radius_generating_fns(
        sorted_branches, content[:, 5], each_length, parents, types
    )

    if np.sum(np.asarray(parents) == -1) > 1.0:
        parents = np.asarray([-1] + parents)
        parents[1:] += 1
        parents = parents.tolist()
        pathlengths = [0.1] + pathlengths
        radius_fns = [lambda x: content[0, 5] * np.ones_like(x)] + radius_fns
        sorted_branches = [[0]] + sorted_branches

        # Type of padded section is assumed to be of `custom` type:
        # http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
        types = [5.0] + types

    all_coords_of_branches = []
    for i, branch in enumerate(sorted_branches):
        # Remove 1 because `content` is an array that is indexed from 0.
        branch = np.asarray(branch) - 1

        # Deal with additional branch that might have been added above in the lines
        # `if np.sum(np.asarray(parents) == -1) > 1.0:`
        branch[branch < 0] = 0

        # Get traced coordinates of the branch.
        coords_of_branch = content[branch, 2:6]
        all_coords_of_branches.append(coords_of_branch)

    return parents, pathlengths, radius_fns, types, all_coords_of_branches


def read_swc(
    fname: str,
    nseg: int,
    max_branch_len: float = 300.0,
    min_radius: Optional[float] = None,
    assign_groups: bool = False,
) -> Cell:
    """Reads SWC file into a `Cell`.

    Jaxley assumes cylindrical compartments and therefore defines length and radius
    for every compartment. The surface area is then 2*pi*r*length. For branches
    consisting of a single traced point we assume for them to have area 4*pi*r*r.
    Therefore, in these cases, we set lenght=2*r.

    Args:
        fname: Path to the swc file.
        nseg: The number of compartments per branch.
        max_branch_len: If a branch is longer than this value it is split into two
            branches.
        min_radius: If the radius of a reconstruction is below this value it is clipped.
        assign_groups: If True, then the identity of reconstructed points in the SWC
            file will be used to generate groups `undefined`, `soma`, `axon`, `basal`,
            `apical`, `custom`. See here:
            http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html

    Returns:
        A `Cell` object.
    """
    parents, pathlengths, radius_fns, types, coords_of_branches = swc_to_jaxley(
        fname, max_branch_len=max_branch_len, sort=True, num_lines=None
    )
    nbranches = len(parents)

    comp = Compartment()
    branch = Branch([comp for _ in range(nseg)])
    cell = Cell(
        [branch for _ in range(nbranches)], parents=parents, xyzr=coords_of_branches
    )
    # Also save the radius generating functions in case users post-hoc modify the number
    # of compartments with `.set_ncomp()`.
    cell._radius_generating_fns = radius_fns

    lengths_each = np.repeat(pathlengths, nseg) / nseg
    cell.set("length", lengths_each)

    radiuses_each = build_radiuses_from_xyzr(
        radius_fns,
        range(len(parents)),
        min_radius,
        nseg,
    )
    cell.set("radius", radiuses_each)

    # Description of SWC file format:
    # http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
    ind_name_lookup = {
        0: "undefined",
        1: "soma",
        2: "axon",
        3: "basal",
        4: "apical",
        5: "custom",
    }
    types = np.asarray(types).astype(int)
    if assign_groups:
        for type_ind in np.unique(types):
            if type_ind < 5.5:
                name = ind_name_lookup[type_ind]
            else:
                name = f"custom{type_ind}"
            indices = np.where(types == type_ind)[0].tolist()
            if len(indices) > 0:
                cell.branch(indices).add_to_group(name)
    return cell
