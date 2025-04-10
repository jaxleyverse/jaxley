# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from morph_tool import convert


def run_conversion():
    convert(
        "morph_tmp.swc", "morph_ca1_n120.swc", sanitize=True, single_point_soma=True
    )


if __name__ == "__main__":
    run_conversion()
