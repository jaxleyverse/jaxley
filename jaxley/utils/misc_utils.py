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


def recursive_compare(a, b):
    if isinstance(a, (int, float)):
        if abs(a - b) > 1e-5 and not (np.isnan(a) and np.isnan(b)):
            return False
    elif isinstance(a, str):
        if a != b:
            return False
    elif isinstance(a, (np.ndarray, jnp.ndarray)):
        if a.size > 1:
            for i in range(len(a)):
                if not recursive_compare(a[i], b[i]):
                    return False
        else:
            if not recursive_compare(a.item(), b.item()):
                return False
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if not recursive_compare(a[i], b[i]):
                return False
    elif isinstance(a, dict):
        if len(a) != len(b) and len(a) != 0:
            return False
        if set(a.keys()) != set(b.keys()):
            return False
        for k in a.keys():
            if not recursive_compare(a[k], b[k]):
                return False
    elif isinstance(a, pd.DataFrame):
        if not recursive_compare(a.to_dict(), b.to_dict()):
            return False
    elif a is None or b is None:
        if not (a is None and b is None):
            return False
    else:
        raise ValueError(f"Type {type(a)} not supported")
    return True
