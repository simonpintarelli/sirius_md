"""Wrappers for SIRIUS DFT_ground_state class"""
import numpy as np
from scipy.special import binom

from dft_direct_minimizer import OTMethod
from sirius import set_atom_positions, spdiag


def loewdin(X):
    """ Apply Loewdin orthogonalization to wfct."""
    S = X.H @ X
    w, U = S.eigh()
    Sm2 = U @ spdiag(1 / np.sqrt(w)) @ U.H
    return X @ Sm2


class DftGroundState:
    """plain SCF. No extrapolation"""

    def __init__(self, solver, **kwargs):
        self.dft_obj = solver
        self.potential_tol = kwargs["potential_tol"]
        self.energy_tol = kwargs["energy_tol"]
        self.num_dft_iter = kwargs["num_dft_iter"]

    def _generate_density_potential(self, kset):
        density = self.dft_obj.density()
        potential = self.dft_obj.potential()
        ctx = kset.ctx()
        density.generate(kset)
        if ctx.use_symmetry():
            density.symmetrize()
            density.symmetrize_density_matrix()

        density.generate_paw_loc_density()
        density.fft_transform(1)
        potential.generate(density)
        if ctx.use_symmetry():
            potential.symmetrize()
        potential.fft_transform(1)

    def update_and_find(self, pos):
        """
        Arguments:
        pos -- atom positions in reduced coordinates
        """
        kset = self.dft_obj.k_point_set()
        unit_cell = kset.ctx().unit_cell()
        pos = np.mod(pos, 1)
        set_atom_positions(unit_cell, pos)

        self.dft_obj.update()

        return self.dft_obj.find(
            potential_tol=self.potential_tol,
            energy_tol=self.energy_tol,
            initial_tol=1e-2,
            num_dft_iter=self.num_dft_iter,
            write_state=False,
        )


class DftWfExtrapolate(DftGroundState):
    """extrapolate wave functions."""

    def __init__(self, solver, order=3, **kwargs):
        super().__init__(solver, **kwargs)
        self.Cs = []
        self.order = order

    def update_and_find(self, pos):
        """
        Arguments:
        pos -- atom positions in reduced coordinates
        """

        kset = self.dft_obj.k_point_set()
        # obtain current wave function coefficients
        C = kset.C
        self.Cs.append(C)

        K = self.order

        if len(self.Cs) >= K:
            # this is Eq (36) from:
            # Kühne, T. D. Ab-Initio Molecular Dynamics. , 4(4), 391–406.
            # http://dx.doi.org/10.1002/wcms.1176
            Cp = binom(K, 1) * self.Cs[-1] @ (self.Cs[-1].H @ self.Cs[-1])
            for m in range(2, K + 1):
                Cp += ((-1) ** (m + 1) * m * binom(2 * K, K - m) / binom(2 * K - 2, K - 1)
                       * self.Cs[-m]
                       @ (self.Cs[-m].H @ self.Cs[-1]))
            # orthogonalize
            Cp = loewdin(Cp)
            # truncate wave function history
            self.Cs = self.Cs[1:]
            # TODO remove phase
            # store extrapolated value
            kset.C = Cp
            self._generate_density_potential(kset)

        return super().update_and_find(pos)


def make_dft(solver, parameters):
    """DFT object factory."""

    num_dft_iter = parameters["parameters"]["num_dft_iter"]
    potential_tol = parameters["parameters"]["potential_tol"]
    energy_tol = parameters["parameters"]["energy_tol"]

    # TODO: clean this up
    if "solver" in parameters["parameters"]:
        if parameters["parameters"]["solver"] == "ot":
            solver = OTMethod(solver)

    if parameters["parameters"]["method"]["type"] == "plain":
        return DftGroundState(
            solver,
            energy_tol=energy_tol,
            potential_tol=potential_tol,
            num_dft_iter=num_dft_iter,
        )
    if parameters["parameters"]["method"]["type"] == "wfct":
        order = parameters["parameters"]["method"]["order"]
        return DftWfExtrapolate(
            solver,
            order=order,
            energy_tol=energy_tol,
            potential_tol=potential_tol,
            num_dft_iter=num_dft_iter,
        )

    raise ValueError("invalid extrapolation method")
