import argparse
import json
from sirius_md.dft_ground_state import loewdin
from .cpmd import CPMDForce, shake, rattle, update_sirius, g_shake, g_rattle
import logging as log
import yaml
import numpy as np
from sirius import atom_positions, PwCoeffs, DFT_ground_state
from sirius.coefficient_array import zeros_like, identity_like
from .atom_mass import atom_masses
from .constants import dalton_to_me
import time
from .utils import to_cart, from_cart, initialize, sirius_load_state


def cpmd_velocity_verlet(x, v, u, F, Hx, Fh, dt, m, me, kset, solvers):
    """Velocity verlet for Car-Parinello"""
    m = m[:, np.newaxis]  # enable broadcasting in numpy
    C = kset.C
    fn = kset.fn

    xn = x + v * dt + 0.5 * F / m * dt**2
    Cn = C + u * dt - 0.5 * kset.fn * Hx / me * dt**2
    Cn, XC = solvers["shake"](Cn, C)  # XC is used to update the electronic velocities
    # log.debug(f"plane wave norms: {np.linalg.norm(Cn[0,0], axis=0)}")
    # log.debug(f"plane wave norms in gamme point: {np.diag(g_dot(Cn,Cn)[0,0])}")
    log.debug(f"occupation numbers: {fn}")

    kset.C = Cn  # update wfc
    Fn, Eksn, Hxn = Fh(Cn, fn, xn)
    gs_json = Fh.sirius_dft_gs.serialize()
    log.debug(f"energy components: \n {gs_json['energy']}")

    vn = v + 0.5 / m * (F + Fn) * dt
    un = u - 0.5 / me * kset.fn * (Hx + Hxn) * dt
    un = solvers["rattle"](un, Cn, XC, dt)

    return xn, vn, Cn, un, Fn, Hxn, Eksn


def boltzmann_velocities(m, T):
    num_atoms = len(m)
    m = m[:, np.newaxis]
    kT = T * 3.16532e-6
    log.info(f"Initial ionic temperature: {T}")
    factors = np.sqrt(kT / m)
    return np.random.normal(
        loc=0, scale=factors, size=(num_atoms, 3)
    )  # Assuming 3D simulations


def cpmd_verlet_raw(input_vars, kset, x0=None, v0=None, u0=None):
    """
    TODO

    Arguments:
    input_vars -- cpmd yaml
    kset       -- sirius kpointset
    x0         -- ion positions (reduced coordinates)
    v0         -- initial velocities (reduced coordinates)
    u0         -- TODO
    """

    me = input_vars["parameters"]["me"]  # Electronic fictitious mass
    dt = input_vars["parameters"]["dt"]
    N = None if "N" not in input_vars["parameters"]  else input_vars["parameters"]["N"]
    dft_ = DFT_ground_state(kset)
    Fh = CPMDForce(dft_)

    unit_cell = kset.ctx().unit_cell()
    lattice_vectors = np.array(unit_cell.lattice_vectors())

    log.debug(f"Initial x \n {x0}")
    log.debug(f"Initial v \n {v0}")
    # log.debug(f"Initial C \n {kset.C[0,0].shape} \n{kset.C[0,0]}")
    # log.debug(f"Initial u \n {u0[0,0]}")
    atom_types = [unit_cell.atom(i).label for i in range(unit_cell.num_atoms)]
    m = (
        np.array([atom_masses[label] for label in atom_types], dtype=np.float64)
        * dalton_to_me
    )

    if x0 is None:
        x0 = atom_positions(unit_cell)
    if v0 is None:
        v0 = np.zeros_like(x0)
    if u0 is None:
        u0 = zeros_like(kset.C)

    vc = to_cart(v0, lattice_vectors)
    log.debug(f"v cartesian \n {vc}")
    ekin = 0.5 * np.sum(vc**2 * m[:, np.newaxis])
    log.debug(f"Initial ion kinetic energy \n {ekin}")
    if kset.ctx().gamma_point:
        log.info("Using gamma approximation")
        solvers = {"shake": g_shake, "rattle": g_rattle}
    else:
        log.info("NOT using gamma approximation")
        solvers = {"shake": shake, "rattle": rattle}
    Fh = CPMDForce(dft_)
    F, Eks, Hx = Fh(kset.C, kset.fn, x0)
    log.debug(f"initial KS energy: {Eks:.12f}")
    log.info("---------Starting main loop-----------")
    t1 = time.time()

    i = 0
    while N is None or i < N:
        log.info(f" iteration {i}")
        xn, vn, Cn, un, Fn, Hxn, Eksn = cpmd_velocity_verlet(
            x0, v0, u0, F, Hx, Fh, dt, m, me, kset, solvers
        )
        log.info(f"KSEnergy : {Eksn}")
        vc = to_cart(vn, lattice_vectors)
        ekin_x = 0.5 * np.sum(vc**2 * m[:, np.newaxis])
        ekin_c = me * np.sum(np.real(np.diag((un.H @ un)[0, 0])))  # TODO: generalize
        log.info(f"T_ions : {ekin_x:.12f}")
        log.info(f"T_coeff: {ekin_c:.12f}")
        log.info(f"Total: {Eksn + ekin_x + ekin_c:.12f}")
        log.info(f"Positions:\n {xn} \n")
        x0 = xn
        v0 = vn
        u0 = un
        yield {
            "i": i,
            "t": i*dt,
            "x": to_cart(x0, lattice_vectors),
            "v": vc,
            "E": Eksn + ekin_x + ekin_c,
            "EKS": Eksn,
            "ekin": ekin_x,
            "ekin_u": ekin_c,
        }
        F = Fn
        Hx = Hxn
        i += 1
    t2 = time.time()
    log.info(f"Simulation ended successfully. Total time: {t2-t1}")


def cpmd_verlet(input_vars, restart):
    """TODO"""
    log.info("Starting CPMD simulation")
    t1 = time.time()
    input_vars = yaml.safe_load(open("input_cpmd.yml", "r"))
    me = input_vars["parameters"]["me"]  # Electronic fictitious mass
    dt = input_vars["parameters"]["dt"]
    N = input_vars["parameters"]["N"]
    T = input_vars["parameters"]["T"]
    initial_positions = None
    initial_velocities = None
    initial_coeff = None

    if restart:
        log.info("Restarting from MD results")
        restart_data = json.load(restart)
        initial_positions = restart_data[-1]["x"]
        initial_velocities = restart_data[-1]["v"]
        kset, _, _, dft_ = initialize(atom_positions=initial_positions, num_dft_iter=1)
        log.debug(f"kset.C.shape: {kset.C[0,0].shape}")
    else:
        log.info("Initializing Sirius DFT object")
        kset, _, _, dft_ = initialize(
            num_dft_iter=10000
        )  # Ask Simon about res["density"/"potential"]

    unit_cell = kset.ctx().unit_cell()
    lattice_vectors = np.array(unit_cell.lattice_vectors())
    log.debug(f"lattice vectors: {lattice_vectors}")
    atom_types = [unit_cell.atom(i).label for i in range(unit_cell.num_atoms)]
    m = (
        np.array([atom_masses[label] for label in atom_types], dtype=np.float64)
        * dalton_to_me
    )

    log.info("Setting initial conditions")
    if restart:
        x0 = atom_positions(unit_cell)
        update_sirius(dft_)
        v0 = from_cart(initial_velocities, lattice_vectors)
        na = len(x0)  # TODO: Avoid code repetition by setting na from input file
        u0 = zeros_like(kset.C)
        C_from_disk = sirius_load_state(
            "init", "C", dst=PwCoeffs(ctype=np.matrix, dtype=np.complex128)
        )
        Cprev_from_disk = sirius_load_state(
            "second", "C", dst=PwCoeffs(ctype=np.matrix, dtype=np.complex128)
        )
        u0 = (C_from_disk - Cprev_from_disk) / dt
        error_vels = np.max(np.abs((u0.H @ kset.C + kset.C.H @ u0)[0, 0]))
        log.debug(f"Pre-rattle velocities error: {error_vels}")
        I = identity_like(kset.C.H @ kset.C)
        C_init, XC = shake(C_from_disk, Cprev_from_disk)
        error_ortho = np.max(np.abs((C_init.H @ C_init - I)[0, 0]))
        log.debug(f"error_ortho after shake: {error_ortho}")
        log.debug(f"{C_init[0,0].shape}")
        kset.C = C_init
        u0 = rattle(u0, kset.C, XC, dt)
        error_vels = np.max(np.abs((u0.H @ kset.C + kset.C.H @ u0)[0, 0]))
        log.info(f"After-rattle velocities error: {error_vels}")
        error_ortho = np.max(np.abs((C_from_disk.H @ C_from_disk - I)[0, 0]))
        log.debug(f"error_ortho before shake: {error_ortho}")
    else:
        v0 = from_cart(boltzmann_velocities(m, T), lattice_vectors)  # np.zeros_like(x0)
        u0 = zeros_like(kset.C)
        x0 = atom_positions(unit_cell)

    log.debug(f"Initial x \n {x0}")
    log.debug(f"Initial v \n {v0}")
    # log.debug(f"Initial C \n {kset.C[0,0].shape} \n{kset.C[0,0]}")
    # log.debug(f"Initial u \n {u0[0,0]}")

    for _ in cpmd_verlet_raw(input_vars, kset, x0, v0, u0):
        continue


def run():
    log.basicConfig(format="%(levelname)s:%(message)s", level=log.DEBUG, force=True)
    input_vars = yaml.safe_load(open("input_cpmd.yml", "r"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", nargs="?", type=argparse.FileType("r"))
    args = parser.parse_args()

    cpmd_verlet(input_vars, restart=args.restart)
