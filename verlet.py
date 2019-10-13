import json

import matplotlib.pyplot as plt
import numpy as np
from atom_mass import atom_masses

from logger import Logger
from sirius import (DFT_ground_state_find, atom_positions,
                    initialize_subspace,
                    set_atom_positions)


class NumpyEncoder(json.JSONEncoder):
    """Numpy helper for json."""
    # pylint: disable=method-hidden,arguments-differ
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def initialize():
    """Initialize DFT_ground_state object."""
    res = DFT_ground_state_find(num_dft_iter=100)

    return res["kpointset"], res["density"], res["potential"], res["dft_gs"]


class ScfConvergenceError(Exception):
    """Custom exception."""


class Force:
    """Compute forces. """

    def __init__(self, dft):
        self.dft = dft
        self.potential = dft.potential()
        self.density = dft.density()
        self.unit_cell = dft.k_point_set().ctx().unit_cell()
        self.L = unit_cell.lattice_vectors()
        self.Lh = np.linalg.inv(self.L)

    def __call__(self, pos):
        """Computes ground state for updated positions and returns total energy and forces.
        Hint: no extrapolation is done
        Arguments:
        - pos Atom poitions in reduced coordinates

        Returns:
        - forces (in reduced coordinates)
        - Etot
        """
        # apply periodic boundary conditions
        pos = np.mod(pos, 1)
        set_atom_positions(self.unit_cell, pos)
        self.dft.update()
        # initialize_subspace(self.dft, self.dft.k_point_set().ctx())
        # self.dft.initial_state()
        res = dft_gs.find(
            potential_tol=1e-8,
            energy_tol=1e-8,
            initial_tol=1e-2,
            num_dft_iter=100,
            write_state=False,
        )

        if not res['converged']:
            raise ScfConvergenceError('failed to converge')
        Logger().insert({'nscf': res['num_scf_iterations'],
                         'band_gap': res['band_gap']})
        print('band_gap: ', res['band_gap'])
        forces = np.array(self.dft.forces().calc_forces_total()).T
        print('nscf: ', res['num_scf_iterations'])
        # convert forces to reduced coordinates
        return forces@self.Lh.T, res['energy']['total']


def velocity_verlet(x, v, F, dt, Fh, m):
    """Velocity Verlet.
    Arguments:
    x  -- atom positions in reduced coordinates
    v  -- velocity
    F  -- forces
    dt -- time step
    Fh -- handle to compute force, e.g Fh(x)
    m  -- atomic masses

    Returns:
    x -- atom positions
    v -- velocities
    F -- forces
    E -- Kohn-Sham energy
    """
    m = m[:, np.newaxis]  # enable broadcasting in numpy
    # update positions
    xn = x + v * dt + 0.5 * F / m * dt ** 2
    # apply periodic bc
    xn = np.mod(xn, 1)

    # update forces, KS energy
    Fn, EKS = Fh(xn)
    vn = v + 0.5 / m * (F + Fn) * dt

    return xn, vn, Fn, EKS


kset, _, _, dft_gs = initialize()

unit_cell = kset.ctx().unit_cell()
lattice_vectors = np.array(unit_cell.lattice_vectors())

# TODO: make a wrapper class for dft_gs which does the extrapolation in dft_gs.update()
# see dft_ground_state.py
Fh = Force(dft_gs)

x0 = atom_positions(unit_cell)
F, EKS = Fh(x0)
v0 = np.zeros_like(x0)
dt = 0.5 # time in fs
N = 500  # number of time steps
na = len(x0)  # number of atoms
atom_types = [unit_cell.atom(i).label for i in range(na)]
# masses in A_r
m = np.array([atom_masses[label] for label in atom_types])

L = lattice_vectors.T

with Logger():
    # Velocity Verlet time-stepping
    for i in range(N):
        print('iteration: ', i, '\n')

        xn, vn, Fn, EKS = velocity_verlet(x0, v0, F, dt, Fh, m)

        # TODO: take periodic bc into account
        print("displacement: %.2e" % np.linalg.norm(xn - x0))

        vc = lattice_vectors.T @ vn.T  # velocity in cartesian coordinates
        ekin = 0.5 * np.sum(vc**2 * m[np.newaxis, :])
        print("Etot: %10.4f, Ekin: %10.4f, Eks: %10.4f" % (EKS + ekin, ekin, EKS))
        Logger().insert(
            {"i": i, "vc": vc, "x": xn, "E": EKS + ekin, "EKS": EKS, "ekin": ekin, 't': i*dt}
        )
        x0 = xn
        v0 = vn
        F = Fn

# dump results to json
log = Logger().log
with open("results_small_dt.json", "w") as fh:
    json.dump(log, fh, cls=NumpyEncoder)

# plot energies over time
ts = np.array([x['t'] for x in log])
plt.plot(ts, [x['E'] for x in log], label='Etot')
plt.plot(ts, [x['EKS'] for x in log], label='KS energy')
plt.xlabel('t [fs]')
plt.ylabel('E [Ha]')
plt.grid(True)
plt.legend(loc='best')
plt.show()
