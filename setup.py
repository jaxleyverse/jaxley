from setuptools import find_packages, setup

REQUIRED = [
    "jax[cpu]",
    "numpy",
]


EXTRAS = {
    "dev": [
        "black",
        "isort",
        "jupyter",
        "mkdocs",
        "mkdocs-material",
        "pytest",
        "pyright",
    ],
}

setup(
    name="neurax",
    python_requires=">=3.6.0",
    packages=find_packages(),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
)
