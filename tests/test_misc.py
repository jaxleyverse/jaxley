# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import os
import re
from pathlib import Path
from typing import List

import numpy as np
import pytest


def list_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                yield os.path.join(root, file)


license_txt = """# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>"""


@pytest.mark.parametrize("dir", ["../jaxley", "."])
def test_license(dir):
    for i, file in enumerate(list_files(dir)):
        with open(file, "r") as f:
            header = f.read(len(license_txt))
        assert (
            header == license_txt
        ), f"File {file} does not have the correct license header"


def test_rm_all_deprecated_functions():
    from jaxley.__version__ import __version__ as package_version

    package_version = np.array([int(s) for s in package_version.split(".")])

    decorator_pattern = r"@deprecated(?:_signature)?"
    version_pattern = r"[v]?(\d+\.\d+\.\d+)"

    package_dir = Path(__file__).parent.parent / "jaxley"

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
                        if not np.all(package_version <= depr_version):
                            violations.append(f"{py_file}:L{line_num}")

    assert not violations, "\n".join(
        ["Found deprecated items that should have been removed:", *violations]
    )
