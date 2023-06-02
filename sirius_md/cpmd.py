from sirius import set_atom_positions
from sirius import DFT_ground_state
from sirius.coefficient_array import PwCoeffs, CoefficientArray
import numpy as np
from sirius.ot import Energy, ApplyHamiltonian
import logging as log

#def initialize_

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
