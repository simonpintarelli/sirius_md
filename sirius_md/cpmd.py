from sirius import set_atom_positions
from sirius import DFT_ground_state
from sirius.coefficient_array import PwCoeffs, CoefficientArray, identity_like, l2norm
import numpy as np
from sirius.ot import Energy, ApplyHamiltonian
import logging as log
import copy as cp

def shake(Cn, C, etol = 1e-6, max_iter = 100 ):
    """ Add the Lagrange multipliers to the current wave function coefficients (wfc). 
    The notation follows Tuckerman & Parrinello (T&P)
    "Implementing the Car-Parrinello equations I" Section IV.B.Velocity Verlet
    """
    A = Cn.H @ Cn
    B = C.H @ Cn
    I = identity_like(A)
    Xn = 0.5 * (I - A)
    for i in range(max_iter): #TODO: Add the number of iteration in input file
        log.debug(f"shake iteration {i}")
        Xn = 0.5*(I - A + Xn @ (I - B) + (I - B.H) @ Xn - (Xn @ Xn))
        XC = C @ Xn
        Cp = Cn + XC # eq. (4.3) in T&P
        error = np.max(np.abs((Cp.H@Cp-I)[0,0])) #TODO: generalize
        log.debug(f"error shake {error}")
        if error < etol:
            break
    return Cp, XC

def rattle(un, Cn, XC, dt):
    un = un + XC/dt # eq. (4.9) in T&P. Note that XC = (dt^2/2 me)\sum_j\Lambda_{ij}C_j
    D = Cn.H@un # in T&P, C is used instead of D
    Y = -0.5*(D+D.H)
    YC = (Y @ Cn.T).T
    un = un + YC # eq. (4.11) in T&P
    error = np.max(np.abs((un.H @ Cn + Cn.H @ un)[0,0])) 
    log.debug(f"error rattle {error}")
    return un 

class CPMDForce:
    """Helper class to compute force and Hx for CPMD."""

    def __init__(self, sirius_dft_gs: DFT_ground_state):
        self.sirius_dft_gs = sirius_dft_gs

        self.H = ApplyHamiltonian(sirius_dft_gs.potential(), sirius_dft_gs.k_point_set())
        # create object to compute the total energy
        self.E = Energy(sirius_dft_gs.k_point_set(), sirius_dft_gs.potential(), sirius_dft_gs.density(), self.H)

    def __call__(self, C: PwCoeffs, fn: CoefficientArray, pos):
        kset = self.sirius_dft_gs.k_point_set()

        # set new ion positions
        unit_cell = kset.ctx().unit_cell()
        pos = np.mod(pos, 1)
        set_atom_positions(unit_cell, pos)
        self.sirius_dft_gs.update()

        # update density and potential
        use_sym = kset.ctx().use_symmetry()
        self.sirius_dft_gs.density().generate(
            kset,
            use_sym,
            False,  # add core (only for lapw)
            True,  # transform to real space grid
        )
        self.sirius_dft_gs.potential().generate(
            self.sirius_dft_gs.density(), use_sym, True  # transform to real space grid
        )

        # Kohn-Sham energy and H*Psi
        Eks, Hx = self.E.compute(C, fn)

        # Forces acting on nuclei
        F = np.array(self.sirius_dft_gs.forces().calc_forces_total()).T

        return F, Eks, Hx
