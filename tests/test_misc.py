# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import os
import re
from pathlib import Path
from typing import List

import numpy as np


def test_rm_all_deprecated_functions():
    from jaxley.__version__ import __version__ as package_version

    package_version = np.array([int(s) for s in package_version.split(".")])

    # Pattern to match both @deprecated and @deprecated_kwargs
    decorator_pattern = r"@deprecated(?:_kwargs)?"
    version_pattern = r"[v]?(\d+\.\d+\.\d+)"

    package_dir = Path(__file__).parent.parent / "jaxley"
    project_root = Path(__file__).parent.parent

    violations = []
    for py_file in package_dir.rglob("*.py"):
        with open(py_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                if re.search(decorator_pattern, line):
                    version_match = re.search(version_pattern, line)
                    if version_match:
                        depr_version_str = version_match.group(1)
                        depr_version = np.array(
                            [int(s) for s in depr_version_str.split(".")]
                        )
                        if np.all(depr_version <= package_version):
                            relative_path = py_file.relative_to(project_root)
                            violations.append(f"{relative_path}:L{line_num}")

    assert not violations, "\n".join(
        ["Found deprecated items that should have been removed:", *violations]
    )
