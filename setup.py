from setuptools import find_packages, setup

REQUIRED = [
    "jax[cpu]",
    "numpy",
    "pandas",
    "matplotlib",
]


EXTRAS = {
    "dev": [
        "black",
        "isort",
        "jupyter",
        "markdown-include",
        "mkdocs",
        "mkdocs-material",
        "neuron",
        "pytest",
        "pyright",
    ],
}

setup(
    name="neurax",
    python_requires=">=3.8.0",
    packages=find_packages(),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
)
