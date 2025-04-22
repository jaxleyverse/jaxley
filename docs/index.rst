.. Jaxley documentation master file, created by
   sphinx-quickstart on Tue Oct 18 10:21:12 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to Jaxley!
===================

``Jaxley`` is a differentiable simulator for biophysical neuron models in `JAX <https://github.com/google/jax>`_. Its key features are:

- automatic differentiation, allowing gradient-based optimization of thousands of parameters  
- support for CPU, GPU, or TPU without any changes to the code  
- ``jit``-compilation, making it as fast as other packages while being fully written in python  
- backward-Euler solver for stable numerical solution of multicompartment neurons  
- elegant mechanisms for parameter sharing


Getting started
---------------

``Jaxley`` allows to simulate biophysical neuron models on CPU, GPU, or TPU:

.. code-block:: python

   import matplotlib.pyplot as plt
   from jax import config

   import jaxley as jx
   from jaxley.channels import HH

   config.update("jax_platform_name", "cpu")  # Or "gpu" / "tpu".

   cell = jx.Cell()  # Define cell.
   cell.insert(HH())  # Insert channels.

   current = jx.step_current(i_delay=1.0, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=10.0)
   cell.stimulate(current)  # Stimulate with step current.
   cell.record("v")  # Record voltage.

   v = jx.integrate(cell)  # Run simulation.
   plt.plot(v.T)  # Plot voltage trace.


If you want to learn more, check out our material:

.. grid:: 4

   .. grid-item-card:: 🧠 Tutorials
      :link: tutorials
      :link-type: doc

      Step-by-step introductions.

   .. grid-item-card:: ⚙️ Advanced tutorials
      :link: advanced_tutorials
      :link-type: doc

      In-depth guides for power-users.

   .. grid-item-card:: 🧩 How-to guides
      :link: how_to_guide
      :link-type: doc

      Practical recipes for common tasks.

   .. grid-item-card:: 📚 API Reference
      :link: jaxley
      :link-type: doc

      Full documentation of modules and functions.


Installation
------------

``Jaxley`` is available on `PyPI <https://pypi.org/project/jaxley/>`_:

.. code-block:: console

   pip install jaxley

This will install ``Jaxley`` with CPU support. If you want GPU support, follow the instructions on the `JAX github repository <https://github.com/google/jax>`_ to install ``JAX`` with GPU support (in addition to installing ``Jaxley``). For example, for NVIDIA GPUs, run

.. code-block:: console

   pip install -U "jax[cuda12]"


Feedback and Contributions
--------------------------

We welcome any feedback on how ``Jaxley`` is working for your neuron models and are happy to receive bug reports, pull requests and other feedback (see `contribute <https://github.com/jaxleyverse/jaxley/blob/main/CONTRIBUTING.md>`_). We wish to maintain a positive community, please read our `Code of Conduct <https://github.com/jaxleyverse/jaxley/blob/main/CODE_OF_CONDUCT.md>`_.


Citation
--------

If you use `Jaxley`, consider citing the `corresponding paper <https://www.biorxiv.org/content/10.1101/2024.08.21.608979>`_:

.. code-block:: console
   
   @article{deistler2024differentiable,
      doi = {10.1101/2024.08.21.608979},
      year = {2024},
      publisher = {Cold Spring Harbor Laboratory},
      author = {Deistler, Michael and Kadhim, Kyra L. and Pals, Matthijs and Beck, Jonas and Huang, Ziwei and Gloeckler, Manuel and Lappalainen, Janne K. and Schr{\"o}der, Cornelius and Berens, Philipp and Gon{\c c}alves, Pedro J. and Macke, Jakob H.},
      title = {Differentiable simulation enables large-scale training of detailed biophysical models of neural dynamics},
      journal = {bioRxiv}
   }

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting started

   installation

.. toctree::
   :hidden:
   :maxdepth: 1

   tutorials
   faq

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: More guides

   advanced_tutorials
   how_to_guide
   jaxley

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: About the Project

   contributor_guide
   changelog
   credits
