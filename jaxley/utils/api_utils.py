# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pandas as pd
from jax import Array


def verbose_recordings(
    recordings: Array, module_recordings: pd.DataFrame
) -> dict[str, Array]:
    """Turn recordings into a dictionary which indicates the recorded state.

    If the recordings record a certain compartment (and the same state) multiple times,
    then it will show up only once in the returned dictionary (i.e., no duplicates).

    Args:
        recordings: The recordings returned by ``jx.integrate()``.
        module: The dataframe containing information about the recordings. Can be
            obtained from ``module.recordings`` (e.g., ``cell.recordings``).

    Returns:
        A dictionary where each key contains one recording. The is made up of three
        parts: '{prefix}{state name}_{index}'. The `index` refers to
        a row in `.nodes` or `.edges`. Additional information about this compartment
        can be obtained with ``cell.nodes.loc[index]``.

    Example usage
    ^^^^^^^^^^^^^

    ::

        import jaxley as jx
        from jaxley.utils import verbose_recordings

        cell = jx.Cell()
        cell.record("v", prefix="excitatory_")
        v = jx.integrate(cell, t_max=10.0)

        recs = verbose_recordings(v, cell.recordings)
        print(recs["excitatory_v_0"])  # Array of shape (T,)

    """
    verbose_recs = {}
    for i, rec in module_recordings.iterrows():
        verbose_recs[f"{rec['prefix']}{rec['state']}_{rec['rec_index']}"] = recordings[
            i
        ]
    return verbose_recs
