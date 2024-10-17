# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import List, Optional, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd


def concat_and_ignore_empty(dfs: List[pd.DataFrame], **kwargs) -> pd.DataFrame:
    """Concatenate dataframes and ignore empty dataframes.

    This is mainly to avoid `pd.concat` throwing a warning when concatenating empty
    and non-empty dataframes."""
    return pd.concat([df for df in dfs if len(df) > 0], **kwargs)


def childview(
    module,
    index: Union[int, str, list, range, slice],
    child_name: Optional[str] = None,
):
    """Return the child view of the current module.

    network.cell(index) at network level.
    cell.branch(index) at cell level.
    branch.comp(index) at branch level."""
    if child_name is None:
        parent_name = module.__class__.__name__.lower()
        views = np.array(["net", "cell", "branch", "comp", "/"])
        child_idx = np.roll([v in parent_name for v in views], 1)
        child_name = views[child_idx][0]
    if child_name != "/":
        return module.__getattr__(child_name)(index)
    raise AttributeError("Compartment does not support indexing")


def cumsum_leading_zero(array: Union[np.ndarray, List]) -> np.ndarray:
    """Return the `cumsum` of a numpy array and pad with a leading zero."""
    arr = np.asarray(array)
    return np.concatenate([np.asarray([0]), np.cumsum(arr)]).astype(arr.dtype)

def index_is_all(idx, force=True):
    """Check if the index is "all"."""
    if isinstance(idx, str):
        if force:
            assert idx == "all", "Only 'all' is allowed"
        return idx == "all"
    return False