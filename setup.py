from setuptools import find_packages, setup

REQUIRED = [
    "jax[cpu]",
    "numpy",
    "pandas>=2.2.0",
    "matplotlib",
    "networkx",
]


EXTRAS = {
    "dev": [
        "black",
        "isort",
        "jupyter",
        "mkdocs",
        "mkdocs-material",
        "markdown-include",
        "mkdocs-redirects",
        "mkdocstrings[python]>=0.18",
        "neuron",
        "pytest",
        "pyright",
        "optax",
    ],
}

setup(
    name="jaxley",
    python_requires=">=3.8.0",
    packages=find_packages(),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
)
