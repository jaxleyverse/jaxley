# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike, DTypeLike

Carry = TypeVar("Carry")
Input = TypeVar("Input")
Output = TypeVar("Output")
Func = TypeVar("Func", bound=Callable)


def to_device(
    x: ArrayLike, dtype: Optional[DTypeLike] = None, device: Optional[Any] = None
) -> jax.Array:
    """Move an array to a JAX device without compiling an XLA program.

    `jnp.asarray(x)` compiles a small XLA program for every new shape, even if `x`
    already has the right dtype. `jax.device_put()` of an array whose dtype JAX
    supports does not.

    Args:
        x: A NumPy or JAX array.
        dtype: The dtype of the result. Defaults to the dtype of `x`, as JAX stores it
            (64-bit dtypes become 32-bit ones unless `jax_enable_x64` is set).
        device: A `jax.Device` (or sharding) to put the array on. Defaults to JAX's
            default device.
    """
    x = np.asarray(x)
    dtype = jax.dtypes.canonicalize_dtype(x.dtype if dtype is None else dtype)
    return jax.device_put(x.astype(dtype, copy=False), device)


def nested_checkpoint_scan(
    f: Callable[[Carry, dict[str, ArrayLike]], tuple[Carry, Output]],
    init: Carry,
    xs: dict[str, ArrayLike],
    length: int | None = None,
    *,
    nested_lengths: Sequence[int],
    scan_fn=jax.lax.scan,
    checkpoint_fn: Callable[[Func], Func] = jax.checkpoint,
):
    """A version of lax.scan that supports recursive gradient checkpointing.

    Code taken from: https://github.com/google/jax/issues/2139

    The interface of `nested_checkpoint_scan` exactly matches lax.scan, except for
    the required `nested_lengths` argument.

    The key feature of `nested_checkpoint_scan` is that gradient calculations
    require O(max(nested_lengths)) memory, vs O(prod(nested_lengths)) for unnested
    scans, which it achieves by re-evaluating the forward pass
    `len(nested_lengths) - 1` times.

    `nested_checkpoint_scan` reduces to `lax.scan` when `nested_lengths` has a
    single element.

    Args:
        f: function to scan over.
        init: initial value.
        xs: scanned over values.
        length: leading length of all dimensions
        nested_lengths: required list of lengths to scan over for each level of
            checkpointing. The product of nested_lengths must match length (if
            provided) and the size of the leading axis for all arrays in ``xs``.
        scan_fn: function matching the API of lax.scan
        checkpoint_fn: function matching the API of jax.checkpoint.
    """
    if length is not None and length != math.prod(nested_lengths):
        raise ValueError(f"inconsistent {length=} and {nested_lengths=}")

    def nested_reshape(x):
        x = jnp.asarray(x)
        new_shape = tuple(nested_lengths) + x.shape[1:]
        return x.reshape(new_shape)

    sub_xs = jax.tree_util.tree_map(nested_reshape, xs)
    return _inner_nested_scan(f, init, sub_xs, nested_lengths, scan_fn, checkpoint_fn)


def _inner_nested_scan(f, init, xs, lengths, scan_fn, checkpoint_fn):
    """Recursively applied scan function."""
    if len(lengths) == 1:
        return scan_fn(f, init, xs, lengths[0])

    @checkpoint_fn
    def sub_scans(carry, xs):
        return _inner_nested_scan(f, carry, xs, lengths[1:], scan_fn, checkpoint_fn)

    carry, out = scan_fn(sub_scans, init, xs, lengths[0])
    stacked_out = jax.tree_util.tree_map(jnp.concatenate, out)
    return carry, stacked_out


def infer_device() -> str:
    """Automatically infer the jax device.

    Returns:
        Either of `gpu`, `tpu`, `cpu`, as a string."""
    platform = jax.devices()[0].platform
    if platform == "gpu":
        return "gpu"
    elif platform == "tpu":
        return "tpu"
    else:
        return "cpu"
