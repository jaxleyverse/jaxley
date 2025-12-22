API reference
=============

Modules
-------

.. autosummary::
   :toctree: reference/_autosummary
   :nosignatures:

   jaxley.Compartment
   jaxley.Branch
   jaxley.Cell
   jaxley.Network


Simulation
----------

.. autosummary::
   :toctree: reference/_autosummary
   :nosignatures:

   jaxley.integrate
   jaxley.integrate.build_init_and_step_fn


Morphologies
------------

.. autosummary::
   :toctree: reference/_autosummary
   :nosignatures:

   jaxley.read_swc
   jaxley.morphology.distance_direct
   jaxley.morphology.distance_pathwise
   jaxley.morphology.morph_delete
   jaxley.morphology.morph_connect


Channels
--------

.. autosummary::
   :toctree: reference/_autosummary
   :nosignatures:

   jaxley.channels.Channel
   jaxley.channels.HH
   jaxley.channels.Izhikevich
   jaxley.channels.Rate
   jaxley.channels.Fire
   jaxley.channels.Na
   jaxley.channels.K
   jaxley.channels.Leak
   jaxley.channels.Km
   jaxley.channels.CaL
   jaxley.channels.CaT


Pumps
-----

.. autosummary::
   :toctree: reference/_autosummary
   :nosignatures:

   jaxley.pumps.Pump
   jaxley.pumps.CaFaradayConcentrationChange
   jaxley.pumps.CaNernstReversal
   jaxley.pumps.CaPump


Synapses
--------

.. autosummary::
   :toctree: reference/_autosummary
   :nosignatures:

   jaxley.synapses.Synapse
   jaxley.synapses.CurrentSynapse
   jaxley.synapses.ConductanceSynapse
   jaxley.synapses.DynamicSynapse
   jaxley.synapses.IonotropicSynapse
   jaxley.synapses.SpikeSynapse
   jaxley.synapses.AlphaSynapse


Connectivity
------------

.. autosummary::
   :toctree: reference/_autosummary
   :nosignatures:

   jaxley.connect.connect
   jaxley.connect.connectivity_matrix_connect
   jaxley.connect.fully_connect
   jaxley.connect.sparse_connect


Optimization
------------

.. autosummary::
   :toctree: reference/_autosummary
   :nosignatures:

   jaxley.optimize.transforms.SigmoidTransform
   jaxley.optimize.transforms.SoftplusTransform
   jaxley.optimize.transforms.NegSoftplusTransform
   jaxley.optimize.transforms.AffineTransform
   jaxley.optimize.transforms.ChainTransform
   jaxley.optimize.transforms.MaskedTransform
   jaxley.optimize.transforms.CustomTransform
   jaxley.optimize.optimizer.TypeOptimizer


Graph backend
-------------

.. autosummary::
   :toctree: reference/_autosummary
   :nosignatures:

   jaxley.io.graph.to_swc_graph
   jaxley.io.graph.build_compartment_graph
   jaxley.io.graph.vis_compartment_graph
   jaxley.io.graph.from_graph
   jaxley.modules.base.to_graph


Utilities
---------

.. autosummary::
   :toctree: reference/_autosummary
   :nosignatures:

   jaxley.solver_gate.save_exp
   jaxley.solver_gate.solve_gate_implicit
   jaxley.solver_gate.solve_gate_exponential
   jaxley.solver_gate.exponential_euler
   jaxley.solver_gate.solve_inf_gate_exponential
   jaxley.solver_gate.heaviside
