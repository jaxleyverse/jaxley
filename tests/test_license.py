# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import os

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
