import numpy as np


def read_swc(fname):
    content = np.loadtxt(fname)

    branches = _split_into_branches(content)
    parents = _build_parents(branches)
    pathlengths = _compute_pathlengths(branches, content[:, 2:5])
    endpoint_radiuses = _extract_endpoint_radiuses(branches, content[:, 5])
    start_radius = content[0, 5]
    return parents, pathlengths, endpoint_radiuses, start_radius


def _split_into_branches(content):
    prev_ind = None
    n_branches = 0
    branch_inds = []
    for c in content:
        current_ind = c[0]
        current_parent = c[-1]
        if current_parent != prev_ind:
            branch_inds.append(int(current_parent))
            n_branches += 1
        prev_ind = current_ind

    all_branches = []
    current_branch = []
    for c in content:
        current_ind = c[0]
        current_parent = c[-1]
        if current_parent in branch_inds[1:]:
            all_branches.append(current_branch)
            current_branch = [int(current_parent), int(current_ind)]
        else:
            current_branch.append(int(current_ind))
    all_branches.append(current_branch)

    return all_branches


def _build_parents(all_branches):
    parents = [None] * len(all_branches)
    all_last_inds = [b[-1] for b in all_branches]
    for i, b in enumerate(all_branches):
        parent_ind = b[0]
        ind = np.where(np.asarray(all_last_inds) == parent_ind)[0]
        if len(ind) > 0 and ind != i:
            parents[i] = ind[0]
        else:
            parents[i] = -1

    return parents


def _extract_endpoint_radiuses(all_branches, radiuses):
    endpoint_radiuses = []
    for b in all_branches:
        branch_endpoint = b[-1]
        # Beause SWC starts counting at 1, but numpy counts from 0.
        ind_of_branch_endpoint = branch_endpoint - 1
        endpoint_radiuses.append(radiuses[ind_of_branch_endpoint])
    return endpoint_radiuses


def _compute_pathlengths(all_branches, coords):
    branch_pathlengths = []
    for b in all_branches:
        coords_in_branch = coords[np.asarray(b) - 1]
        point_diffs = np.diff(coords_in_branch, axis=0)
        dists = np.sqrt(
            point_diffs[:, 0] ** 2 + point_diffs[:, 1] ** 2 + point_diffs[:, 2] ** 2
        )
        branch_pathlengths.append(np.sum(dists))
    return branch_pathlengths


# def old_read_swc(fname):
#     content = np.loadtxt(fname)

#     prev_ind = None
#     n_branches = 0
#     branch_inds = []
#     for c in content:
#         current_ind = c[0]
#         current_parent = c[-1]
#         if current_parent != prev_ind:
#             branch_inds.append(int(current_parent))
#             n_branches += 1
#         prev_ind = current_ind

#     all_branches = []
#     current_branch = []
#     for c in content:
#         current_ind = c[0]
#         current_parent = c[-1]
#         if current_parent in branch_inds:
#             all_branches.append(current_branch)
#             current_branch = [int(current_parent), int(current_ind)]
#         else:
#             current_branch.append(int(current_ind))

#     parents = [0]
#     allb = all_branches[2:]

#     for loc_to_jump_back_to, parent_branch in enumerate(allb):
#         current_endpoint_ind = parent_branch[-1]
#         current_parent_ind = loc_to_jump_back_to
#         for _ in range(len(allb)):
#             for child_b_ind in allb:
#                 if child_b_ind[0] == current_endpoint_ind:
#                     parents.append(current_parent_ind + 1)
#                     current_endpoint_ind = child_b_ind[-1]
#                     break
#             current_parent_ind = len(parents) - 1

#         current_endpoint_ind = allb[loc_to_jump_back_to][-1]
#         current_parent_ind = loc_to_jump_back_to
#         for i in range(len(allb)):
#             for child_b_ind in reversed(range(len(allb))):
#                 if allb[child_b_ind][0] == current_endpoint_ind:
#                     parents.append(current_parent_ind + 1)
#                     current_endpoint_ind = allb[child_b_ind][-1]
#                     break
#             current_parent_ind = len(parents) - 1

#     return parents
