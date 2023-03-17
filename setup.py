from setuptools import find_packages, setup

REQUIRED = [
    "jax[cpu]",
    "numpy",
    "pytest",
    "black",
]

setup(
    name="neurax",
    python_requires=">=3.6.0",
    packages=find_packages(),
    install_requires=REQUIRED,
)
