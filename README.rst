Installation
============

.. code:: bash

          pip install -e .


Run
===

.. code:: python

          OMP_NUM_THREADS=4 verlet | tee verlet.out


Config example
==============

`input.yml`

.. code:: yaml

          Methods:
            - &plain
              type: plain

            - &kolafa
              type: kolafa
              order: 3

            - &niklasson
              type: niklasson_wf
              order: 3

          parameters:
            method: *plain
            # solver either [ot, scf, mvp2]
            solver: ot
            # maximal number of scf iterations
            maxiter: 30
            potential_tol: 1.0e-6
            energy_tol: 1.0e-6
            # time step in fs
            dt: 1
            # number of time steps
            N: 100
