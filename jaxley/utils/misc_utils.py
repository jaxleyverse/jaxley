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


def cumsum_leading_zero(array: Union[np.ndarray, List]) -> np.ndarray:
    """Return the `cumsum` of a numpy array and pad with a leading zero."""
    arr = np.asarray(array)
    return np.concatenate([np.asarray([0]), np.cumsum(arr)]).astype(arr.dtype)


def is_str_all(arg, force: bool = True) -> bool:
    """Check if arg is "all".

    Args:
        arg: The arg to check.
        force: If True, then assert that arg is "all".
    """
    if isinstance(arg, str):
        if force:
            assert arg == "all", "Only 'all' is allowed"
        return arg == "all"
    return False
