from sirius_md.verlet import initialize, to_cart
from sirius_md.dft_ground_state import loewdin 
from .cpmd import (CPMDForce, shake, rattle)
import logging as log
import yaml
import numpy as np
from sirius import atom_positions
from sirius.coefficient_array import zeros_like
from .atom_mass import atom_masses
import time
log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)



def cpmd_velocity_verlet(x, v, u, F, Hx, Fh, dt, m, me, kset):
    """TODO"""
    m = m[:, np.newaxis]  # enable broadcasting in numpy
    C = kset.C
    fn = kset.fn

    xn = x + v * dt + 0.5 * F / m * dt ** 2
    Cn = C + u * dt - 0.5 * Hx / me * dt ** 2 
    Cn, XC = shake(Cn, C) # XC is used to update the electronic velocities
    log.debug(f"plane wave norms: {np.linalg.norm(Cn[0,0], axis=0)}")

    kset.C = Cn #update wfc 
    Fn, Eksn, Hxn = Fh(Cn, fn, xn)

    vn =  v + 0.5 / m * (F + Fn) * dt 
    un =  u - 0.5 / me * (Hx + Hxn) * dt 
    un = rattle(un, Cn, XC, dt) 

    return xn, vn, Cn, un, Fn, Eksn  

def boltzmann_velocities(m, kT):
    num_atoms = len(m)
    m = m[:, np.newaxis]
    factors = np.sqrt(kT/m)
    return np.random.normal(loc=0, scale = factors, size=(num_atoms, 3)) #Assuming 3D simulations



def run():
    """TODO"""
    log.info("Starting CPMD simulation")  
    t1 = time.time()
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
    na = len(x0)  # number of atoms
    atom_types = [unit_cell.atom(i).label for i in range(na)] 
    m = np.array([atom_masses[label] for label in atom_types])*1822.89
    v0 = boltzmann_velocities(m,0.001) #np.zeros_like(x0)
    u0 = zeros_like(kset.C) 

    Fh = CPMDForce(dft_)
    F, Eks, Hx = Fh(kset.C, kset.fn, x0)
    log.info ("---------Starting main loop-----------")
    for i in range(N):
        log.info(f"iteration {i}")
        xn, vn, Cn, un, Fn, Eksn = cpmd_velocity_verlet(x0, v0, u0, F, Hx, Fh, dt, m, me, kset)
        log.info(f"KSEnergy = {Eksn}")
        vc = to_cart(vn, lattice_vectors)
        ekin_x = 0.5 * np.sum(vc**2 * m[:, np.newaxis])
        ekin_c = 0.5 * me * np.sum(np.real(np.diag((un.H@un)[0,0]))) #TODO: generalize
        log.info(f"T_ions :{ekin_x}")
        log.info(f"T_coeff:{ekin_c}")
        log.info(f"Total:{Eksn + ekin_x + ekin_c}")
        log.info(f"Positions:\n {xn}")
        x0 = xn
        v0 = vn
        u0 = un
        F = Fn
    t2 = time.time()
    log.info(f"Simulation ended successfully. Total time: {t2-t1}")
    return 0 
