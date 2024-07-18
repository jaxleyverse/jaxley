from typing import List, Optional, Union

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