==================
Firefly Simulation
==================

Model
-----

The model consists of a set of nodes (representing the fireflies) 
who `fire` at intervals dictated by a phase :math:`\phi`. The phase
increases linearly over time until it hits a threshold :math:`\phi_{t}`
where node 'fires' and sets the phase back to zero. The events are observed
by other nodes who then adjust then update their own phase.

Currently this phase is updated when an event is observed using

.. math::
    \phi+\Delta\phi = \text{min}(\alpha.\phi+\beta, 1)

where

.. math::
    \alpha=\text{exp}(b.\epsilon)

and

.. math::

    \beta=\frac{\text{exp}(b.\epsilon)-1}{\text{exp}(b)-1}

This implementation supports finite transmission times between the agent
(i.e. the updates are not received simultaneously between nodes).

Time is discrete in model, and as such transmission times must also be
represented by integers.

Usage
-----

A conda environment with the requirements can be created using

.. code-block::

    conda env create -f environment.yml

examples of initializing and running using the model can be found in
the :code:`usage.ipynb` notebook.
the :code:`usage.ipynb` notebook.
