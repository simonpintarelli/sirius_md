Installation
============

.. code:: bash

          pip install -e .


Run
===

.. code:: python

          OMP_NUM_THREADS=4 verlet | tee verlet.out


.. code:: python

          OMP_NUM_THREADS=4 cpmd | tee cpmd.out


Input files
============

BOMB
----

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
            density_tol: 1.0e-6
            energy_tol: 1.0e-6
            # time step in fs
            dt: 1
            # number of time steps
            N: 100

CPMD
----

`input_cpmd.yml`

.. code:: yaml
          parameters:
          # time step in atomic units
            dt: 10
            # number of time steps
            N: 1000
            # electronic fictitious mass
            me: 300
            # temperature in Kelvin
            T: 0
