import json
import yaml

import matplotlib.pyplot as plt
import numpy as np
from atom_mass import atom_masses
import argparse
from dft_ground_state import make_dft

from logger import Logger
from sirius import (DFT_ground_state_find, atom_positions,
                    initialize_subspace,
                    set_atom_positions)


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
        self.unit_cell = dft.dft_obj.k_point_set().ctx().unit_cell()
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
        res = self.dft.update_and_find(pos)
        if not res['converged']:
            raise ScfConvergenceError('failed to converge')
        Logger().insert({'nscf': res['num_scf_iterations'],
                         'band_gap': res['band_gap']})
        print('band_gap: ', res['band_gap'])
        forces = np.array(self.dft.dft_obj.forces().calc_forces_total()).T
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
    print('xn', xn)
    # update forces, KS energy
    Fn, EKS = Fh(xn)
    vn = v + 0.5 / m * (F + Fn) * dt

    return xn, vn, Fn, EKS


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-t', '--dt', type=float, default=1.0, help='time-step in fs')
    parser.add_argument('-N', type=int, default=50, help='number of timesteps')

    return parser.parse_args()


input_vars = yaml.load(open('input.yaml', 'r'))
potential_tol = input_vars['parameters']['potential_tol']
energy_tol = input_vars['parameters']['potential_tol']
energy_tol = input_vars['parameters']['potential_tol']
N = input_vars['parameters']['N']
dt = input_vars['parameters']['dt']

kset, _, _, dft_ = initialize()

# dft = DftWfExtrapolate(dft_, order=2, potential_tol=1e-4, energy_tol=1e-4, num_dft_iter=100)
# dft = DftGroundState(dft_, potential_tol=1e-4, energy_tol=1e-4, num_dft_iter=100)
dft = make_dft(dft_, input_vars)

unit_cell = kset.ctx().unit_cell()
lattice_vectors = np.array(unit_cell.lattice_vectors())

Fh = Force(dft)

x0 = atom_positions(unit_cell)
F, EKS = Fh(x0)
v0 = np.zeros_like(x0)
na = len(x0)  # number of atoms
atom_types = [unit_cell.atom(i).label for i in range(na)]
# masses in A_r
m = np.array([atom_masses[label] for label in atom_types])

L = lattice_vectors.T

with Logger('logger.out'):
    # Velocity Verlet time-stepping
    for i in range(N):
        print('iteration: ', i, '\n')

        xn, vn, Fn, EKS = velocity_verlet(x0, v0, F, dt, Fh, m)
        print("displacement: %.2e" % np.linalg.norm(xn - x0))
        print('pos: ', xn)

        vc = lattice_vectors.T @ vn.T  # velocity in cartesian coordinates
        ekin = 0.5 * np.sum(vc**2 * m[np.newaxis, :])
        print("Etot: %10.4f, Ekin: %10.4f, Eks: %10.4f" % (EKS + ekin, ekin, EKS))
        Logger().insert(
            {"i": i, "vc": vc, "x": xn, "E": EKS + ekin, "EKS": EKS, "ekin": ekin, 't': i*dt}
        )
        x0 = xn
        v0 = vn
        F = Fn

# plot energies over time
log = Logger().log
ts = np.array([x['t'] for x in log])
plt.plot(ts, [x['E'] for x in log], label='Etot')
plt.plot(ts, [x['EKS'] for x in log], label='KS energy')
plt.xlabel('t [fs]')
plt.ylabel('E [Ha]')
plt.grid(True)
plt.legend(loc='best')
plt.show()
