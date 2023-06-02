from sirius_md.verlet import initialize, to_cart
from sirius_md.dft_ground_state import loewdin 
from .cpmd import CPMDForce
import logging as log
import yaml
import numpy as np
from sirius import atom_positions
from sirius.coefficient_array import zeros_like
from .atom_mass import atom_masses
log.basicConfig(format='%(levelname)s:%(message)s', level=log.INFO )



def cpmd_velocity_verlet(x, v, C, u, F, Hx, Fh, dt, m, me , fn):
    """TODO"""
    m = m[:, np.newaxis]  # enable broadcasting in numpy

    log.debug("Updating positions and eDoF")
    xn = x + v * dt + 0.5 * F / m * dt ** 2
    Cn = C + u * dt - 0.5 * Hx / me * dt ** 2 
    # Orthonormalization
    magnitudes = np.linalg.norm(Cn[0,0], axis=0)
    Cn[0,0] = Cn[0,0]/magnitudes
    Cn = loewdin(Cn)

    Fn, Eksn, Hxn = Fh(Cn, fn, xn)

    log.debug("Updating both velocities")
    vn =  v + 0.5 / m * (F + Fn) * dt 
    un =  u - 0.5 / me * (Hx + Hxn) * dt 

    return xn, vn, Cn, un, Fn, Eksn  



def run():
    """TODO"""
    log.info("Starting CPMD simulation")  
    input_vars = yaml.safe_load(open('input_cpmd.yml', 'r'))
    me = input_vars['parameters']['me'] #Electronic fictitious mass
    dt = input_vars['parameters']['dt'] 
    N = input_vars['parameters']['N'] 

    log.info("Initializing Sirius DFT object")  
    kset, _, _, dft_ = initialize() #Ask Simon about res["density"/"potential"]
    log.info("Setting initial conditions")  
    unit_cell = kset.ctx().unit_cell()
    lattice_vectors = np.array(unit_cell.lattice_vectors())
    x0 = atom_positions(unit_cell)
    v0 = np.zeros_like(x0)
    u0 = zeros_like(kset.C) 
    na = len(x0)  # number of atoms
    Fh = CPMDForce(dft_)
    F, Eks, Hx = Fh(kset.C, kset.fn, x0)
    atom_types = [unit_cell.atom(i).label for i in range(na)]
    m = np.array([atom_masses[label] for label in atom_types])
    log.info ("---------Starting main loop-----------")
    for i in range(N):
        log.info(f"iteration {i}")
        xn, vn, Cn, un, Fn, Eksn = cpmd_velocity_verlet(x0, v0, kset.C, u0, F, Hx, Fh, dt, m, me, kset.fn)
        log.info(f"KSEnergy = {Eksn}")
        vc = to_cart(vn, lattice_vectors)
        ekin = 0.5 * np.sum(vc**2 * m[:, np.newaxis])
        x0 = xn
        v0 = vn
        kset.C = Cn
        F = Fn
    log.info("Simulation ended successfully.")
    return 0 
