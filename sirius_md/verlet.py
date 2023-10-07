import json
import yaml
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sirius
from sirius import CoefficientArray, atom_positions
from sirius import Logger as pprinter
from .constants import dalton_to_me
from .atom_mass import atom_masses
from .dft_ground_state import create_ground_state_solver
from .logger import Logger
from .utils import to_cart, from_cart, initialize, ScfConvergenceError
import time
import h5py
from h5py import File
from mpi4py import MPI

# pprint = pprinter()


class Force:
    """Compute forces.

    The Force class is constructed using a dft object (groundstate solver).
    Method `__call__` takes ion positions as input, solves the groundstate, and
    returns the forces acting on the ions and the KS energy.
    """

    def __init__(self, dft):
        self.dft = dft
        self.unit_cell = dft.dft_obj.k_point_set().ctx().unit_cell()
        self.L = self.unit_cell.lattice_vectors()
        self.Lh = np.linalg.inv(self.L)

    def __call__(self, pos):
        """Computes ground state for updated positions and returns total energy and forces.
        Hint: no extrapolation is done
        Arguments:
        - pos Atom poitions in reduced coordinates

        Returns:
        forces -- Forces in reduced coordinates
        Etot   -- Free energy
        """
        # apply periodic boundary conditions
        res = self.dft.update_and_find(pos)
        if not res["converged"]:
            raise ScfConvergenceError("failed to converge")
        Logger().insert(
            {
                "nscf": res["num_scf_iterations"],
                "band_gap": res["band_gap"],
                "scf_dict": res,
            }
        )
        # pprint("band_gap: ", res["band_gap"])
        forces = np.array(
            self.dft.dft_obj.forces().calc_forces_total(add_scf_corr=False)
        ).T
        # pprint("nscf: ", res["num_scf_iterations"])

        Logger().insert(
            {
                "forces": {
                    "ewald": np.array(self.dft.dft_obj.forces().ewald),
                    "vloc": np.array(self.dft.dft_obj.forces().vloc),
                    "nonloc": np.array(self.dft.dft_obj.forces().nonloc),
                    "core": np.array(self.dft.dft_obj.forces().core),
                    "scf_corr": np.array(self.dft.dft_obj.forces().scf_corr),
                }
            }
        )
        # convert forces to reduced coordinates
        return forces @ self.Lh.T, res["energy"]["total"]


def velocity_verlet_step(x, v, F, dt, Fh, m):
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
    xn = x + v * dt + 0.5 * F / m * dt**2
    t1 = time.time()
    Fn, EKS = Fh(xn)
    gs_json = Fh.dft.dft_obj.serialize()

    t2 = time.time()
    vn = v + 0.5 / m * (F + Fn) * dt
    # remove momentum
    p = np.sum(vn * m, axis=0)
    # pprint("momentum:", p)
    # vn -= p / np.sum(m)
    # pprint('momentum (fixed):', p)
    Logger().insert(
        {
            "t_evalforce": t2 - t1,
            "momentum": p,
            "energy_components": gs_json["energy"],
            "F": Fn,
        }
    )

    return xn, vn, Fn, EKS


def velocity_verlet_raw(input_vars, kset, x0=None, v0=None):
    """
    Arguments:
    input_vars -- input_vars
    kset       -- sirius k_point_set
    x0         -- initial positions (reduced coordinates)
    v0         -- initial velocities (reduced coordinates)
    """

    N = None if "N" not in input_vars["paratmers"]  else input_vars["Paratmers"]["N"]
    dt = input_vars["parameters"]["dt"]
    # create ground state solver: this is a DFT solver with the given extrapolation method
    gs_solver = create_ground_state_solver(sirius.DFT_ground_state(kset), input_vars)
    unit_cell = kset.ctx().unit_cell()
    lattice_vectors = np.array(unit_cell.lattice_vectors())
    # create a force object
    Fh = Force(gs_solver)

    if x0 is None:
        x0 = atom_positions(unit_cell)
    if v0 is None:
        v0 = np.zeros_like(x0)
    F, EKS = Fh(x0)


    na = len(x0)  # number of atoms
    atom_types = [unit_cell.atom(i).label for i in range(na)]
    # masses in A_r
    m = (
        np.array([atom_masses[label] for label in atom_types], dtype=np.float64)
        * dalton_to_me
    )

    i = 0
    while N is None or i < N:
        # pprint("iteration: ", i, "\n")

        xn, vn, Fn, EKS = velocity_verlet_step(x0, v0, F, dt, Fh, m)
        # pprint("displacement: %.2e" % np.linalg.norm(xn - x0))
        vc = to_cart(vn, lattice_vectors)
        ekin = 0.5 * np.sum(vc**2 * m[:, np.newaxis])
        # pprint("Etot: %10.8f, Ekin: %10.8f, Eks: %10.8f" % (EKS + ekin, ekin, EKS))

        data = {
            "i": i,
            "v": to_cart(vn, lattice_vectors),
            "x": to_cart(xn, lattice_vectors),
            "E": EKS + ekin,
            "EKS": EKS,
            "ekin": ekin,
            "t": i * dt,
        }
        yield data

        Logger().insert(data)
        x0 = xn
        v0 = vn
        F = Fn
        i += 1


def velocity_verlet(input_vars, restart):
    N = input_vars["parameters"]["N"]
    dt = input_vars["parameters"]["dt"]

    initial_positions = None
    initial_velocities = None
    if restart:
        restart_data = json.load(restart)
        initial_positions = restart_data[-1]["x"]
        initial_velocities = restart_data[-1]["v"]

    # initialize SIRIUS, run DFT_ground_state class
    kset, _, _, dft_ = initialize(
        tol=input_vars["parameters"]["energy_tol"], atom_positions=initial_positions
    )
    # create ground state solver: this is a DFT solver with the given extrapolation method
    gs_solver = create_ground_state_solver(dft_, input_vars)
    unit_cell = kset.ctx().unit_cell()
    lattice_vectors = np.array(unit_cell.lattice_vectors())
    # create a force object
    Fh = Force(gs_solver)

    x0 = atom_positions(unit_cell)
    F, EKS = Fh(x0)

    if initial_velocities:
        v0 = from_cart(initial_velocities, lattice_vectors)
    else:
        v0 = np.zeros_like(x0)

    with Logger():
        for _ in velocity_verlet_raw(input_vars, kset, x0, v0):
            pass

def run():
    parser = argparse.ArgumentParser()

    parser.add_argument("--restart", nargs="?", type=argparse.FileType("r"))
    args = parser.parse_args()

    input_vars = yaml.safe_load(open("input.yml", "r"))
    velocity_verlet(input_vars, args.restart)

if __name__ == "__main__":
    run()
