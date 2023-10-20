from sirius import set_atom_positions
from sirius import DFT_ground_state
from sirius.coefficient_array import PwCoeffs, CoefficientArray, identity_like, zeros_like, l2norm
import numpy as np
from sirius.ot import Energy, ApplyHamiltonian
import logging as log
import copy as cp


def shake(Cn, C, etol=5e-15, max_iter=100):
    """Add the Lagrange multipliers to the current wave function coefficients (wfc).
    The notation follows Tuckerman & Parrinello (T&P)
    "Implementing the Car-Parrinello equations I" Section IV.B.Velocity Verlet
    """
    A = Cn.H @ Cn
    B = C.H @ Cn
    I = identity_like(A)
    Xn = 0.5 * (I - A)
    for _ in range(max_iter + 1):  # TODO: Add the number of iteration in input file
        Xn = 0.5 * (I - A + Xn @ (I - B) + (I - B.H) @ Xn - (Xn @ Xn))
        XC = C @ Xn.H
        Cp = Cn + XC  # eq. (4.3) in T&P
        error = np.max(np.abs((Cp.H @ Cp - I)[0, 0]))  # TODO: generalize
        log.debug(f"error shake : {error}")
        if error < etol:
            log.debug(f"plane wave norms: {np.linalg.norm(Cp[0,0], axis=0)}")
            return Cp, XC
    raise Exception("shake failed to converge")


def g_dot(A: PwCoeffs, B: PwCoeffs):
    AB = zeros_like(A.H @ B)

    A_matrix = A[0, 0]
    B_matrix = B[0, 0]

    IJ = np.conj(A_matrix[1:, :]).T @ B_matrix[1:, :]

    S = np.outer(np.conj(A_matrix[0, :]), B_matrix[0, :]) + IJ + np.conj(IJ)

    AB[0, 0] = S
    return AB


def g_shake(Cn, C, etol=5e-15, max_iter=100):
    """Add the Lagrange multipliers to the current wave function coefficients (wfc).
    The notation follows Tuckerman & Parrinello (T&P)
    "Implementing the Car-Parrinello equations I" Section IV.B.Velocity Verlet
    """
    A = g_dot(Cn, Cn)
    B = g_dot(C, Cn)
    I = identity_like(A)
    Xn = 0.5 * (I - A)
    for _ in range(max_iter + 1):  # TODO: Add the number of iteration in input file
        Xn = 0.5 * (I - A + Xn @ (I - B) + (I - B.H) @ Xn - (Xn @ Xn))
        XC = C @ Xn.H
        Cp = Cn + XC  # eq. (4.3) in T&P
        CpHCp = g_dot(Cp, Cp)
        error = np.max(np.abs((CpHCp - I)[0, 0]))  # TODO: generalize
        log.debug(f"error shake : {error}")
        if error < etol:
            log.debug(f"plane wave norms in gamme point: {np.diag(CpHCp[0,0])}")
            return Cp, XC
    raise Exception("shake failed to converge")


def rattle(un, Cn, XC, dt):
    un = (
        un + XC / dt
    )  # eq. (4.9) in T&P. Note that XC = (dt^2/2 me)\sum_j\Lambda_{ij}C_j
    D = Cn.H @ un  # in T&P, C is used instead of D
    Y = -0.5 * (D + D.H)
    YC = Cn @ Y.H
    un = un + YC  # eq. (4.11) in T&P
    error = np.max(np.abs((un.H @ Cn + Cn.H @ un)[0, 0]))
    log.debug(f"error rattle : {error}")
    return un


def g_rattle(un, Cn, XC, dt):
    un = (
        un + XC / dt
    )  # eq. (4.9) in T&P. Note that XC = (dt^2/2 me)\sum_j\Lambda_{ij}C_j
    D = g_dot(Cn, un)  # in T&P, C is used instead of D
    Y = -0.5 * (D + D.H)
    YC = Cn @ Y.H
    un = un + YC  # eq. (4.11) in T&P
    error = np.max(np.abs((g_dot(un, Cn) + g_dot(Cn, un))[0, 0]))
    log.debug(f"error rattle : {error}")
    return un


def update_sirius(sirius_dft_gs: DFT_ground_state):
    kset = sirius_dft_gs.k_point_set()
    H = ApplyHamiltonian(sirius_dft_gs.potential(), sirius_dft_gs.k_point_set())
    E = Energy(
        sirius_dft_gs.k_point_set(),
        sirius_dft_gs.potential(),
        sirius_dft_gs.density(),
        H,
    )
    E.compute(kset.C, kset.fn)


class CPMDForce:
    """Helper class to compute force and Hx for CPMD."""

    def __init__(self, sirius_dft_gs: DFT_ground_state):
        self.sirius_dft_gs = sirius_dft_gs
        self.unit_cell = self.sirius_dft_gs.k_point_set().ctx().unit_cell()
        self.L = self.unit_cell.lattice_vectors()
        self.Lh = np.linalg.inv(self.L)

        self.H = ApplyHamiltonian(
            sirius_dft_gs.potential(), sirius_dft_gs.k_point_set()
        )
        # create object to compute the total energy
        self.E = Energy(
            sirius_dft_gs.k_point_set(),
            sirius_dft_gs.potential(),
            sirius_dft_gs.density(),
            self.H,
        )

    def __call__(self, C: PwCoeffs, fn: CoefficientArray, pos):
        kset = self.sirius_dft_gs.k_point_set()

        # set new ion positions
        unit_cell = kset.ctx().unit_cell()
        pos = np.mod(pos, 1)
        set_atom_positions(unit_cell, pos)
        self.sirius_dft_gs.update()

        # Kohn-Sham energy and H*Psi
        Eks, Hx = self.E.compute(C, fn)

        # Forces acting on nuclei
        F = np.array(
            self.sirius_dft_gs.forces().calc_forces_total(add_scf_corr=False)
        ).T

        return F @ self.Lh.T, Eks, Hx
