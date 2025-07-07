# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
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


class deprecated:
    """Decorator to mark a function as deprecated.

    Can be used to mark functions that will be removed in future versions. This will
    also be tested in the CI pipeline to ensure that deprecated functions are removed.

    Warns with: "func_name is deprecated and will be removed in version version."

    Args:
        version: The version in which the function will be removed, i.e. "0.1.0".
        amend_msg: An optional message to append to the deprecation warning.
    """

    def __init__(self, version: str, amend_msg: str = ""):
        self._version: str = version
        self._amend_msg: str = amend_msg

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            msg = (
                f"{func.__name__} is deprecated and will be removed in version "
                f"{self._version}."
            )
            warnings.warn(msg + self._amend_msg)
            return func(*args, **kwargs)

        return wrapper


class deprecated_kwargs:
    """Decorator to mark a keyword argument of a function as deprecated.

    Can be used to mark kwargs that will be removed in future versions. This will
    also be tested in the CI pipeline to ensure that deprecated kwargs are removed.

    Warns with: "kwarg is deprecated and will be removed in version version."

    Args:
        version: The version in which the keyword argument will be removed, i.e.
            `0.1.0`.
        deprecated_kwargs: A list of keyword arguments that are deprecated.
        amend_msg: An optional message to append to the deprecation warning.
    """

    def __init__(self, version: str, kwargs: List = [], amend_msg: str = ""):
        self._version: str = version
        self._amend_msg: str = amend_msg
        self._deprecated_kwargs: List = kwargs

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            for deprecated_kwarg in self._deprecated_kwargs:
                if deprecated_kwarg in kwargs and kwargs[deprecated_kwarg] is not None:
                    msg = (
                        f"{deprecated_kwarg} is deprecated and will be removed in "
                        f"version {self._version}."
                    )
                    warnings.warn(msg + self._amend_msg)
            return func(*args, **kwargs)

        return wrapper
