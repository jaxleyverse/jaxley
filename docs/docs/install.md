## Install the most recent stable version
`Jaxley` is available on [`pypi`](https://pypi.org/project/jaxley/):
```sh
pip install jaxley
```
This will install `Jaxley` with CPU support. If you want GPU support, follow the instructions on the [`JAX` github repository](https://github.com/google/jax) to install `JAX` with GPU support (in addition to installing `Jaxley`). For example, for NVIDIA GPUs, run
```sh
pip install -U "jax[cuda12]"
```

## Install from source
You can also install `Jaxley` from source:
```sh
git clone https://github.com/jaxleyverse/jaxley.git
cd jaxley
pip install -e .
```

Note that `pip>=21.3` is required to install the editable version with `pyproject.toml` see [pip docs](https://pip.pypa.io/en/latest/reference/build-system/pyproject-toml/#editable-installation). 