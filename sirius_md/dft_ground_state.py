"""Wrappers for SIRIUS DFT_ground_state class"""
import numpy as np
from scipy.special import binom

from .dft_direct_minimizer import OTMethod
from sirius import set_atom_positions, spdiag, l2norm, diag
from sirius.coefficient_array import threaded
from scipy import linalg as la


def loewdin(X):
    """ Apply Loewdin orthogonalization to wfct."""
    S = X.H @ X
    w, U = S.eigh()
    Sm2 = U @ spdiag(1 / np.sqrt(w)) @ U.H
    return X @ Sm2

def _solve(A, X):
    """
    returns A⁻¹ X
    """
    out = type(X)(dtype=X.dtype, ctype=X.ctype)
    for k in X.keys():
        out[k] = np.linalg.solve(A[k], X[k])
    return out

@threaded
def chol(X):
    return la.cholesky(X)

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
        Update positions and compute ground state
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


def Bm(K, j):
    """Extrapolation coefficients from Kolafa 0 < j < K+2"""
    return (-1)**(j+1) * j * binom(2*K +2, K+1-j) / binom(2*K, K)


class DftWfExtrapolate(DftGroundState):
    """extrapolate wave functions."""

    def __init__(self, solver, order=3, **kwargs):
        super().__init__(solver, **kwargs)
        self.Cs = []
        self.order = order
        # extrapolation coefficients
        self.Bm = [Bm(order, j) for j in range(1, order+2)]
        print('Extrapolation coefficients: ', self.Bm)
        print('Extrapolation order: ', len(self.Bm))
        assert np.isclose(np.sum(self.Bm), 1)

    def update_and_find(self, pos):
        """
        Arguments:
        pos -- atom positions in reduced coordinates
        """

        kset = self.dft_obj.k_point_set()
        # obtain current wave function coefficients
        C = kset.C
        self.Cs.append(C)

        if len(self.Cs) >= self.order+1:
            print('extrpolate')

            # this is Eq (19) from:
            # Kolafa, J., Time-reversible always stable predictor–corrector method
            #             for molecular dynamics of polarizable molecules,
            # 25(3), 335–342 ().  http://dx.doi.org/10.1002/jcc.10385
            Cp = self.Bm[0] * self.Cs[-1]
            for j in range(1, self.order+1):
                Cp += self.Bm[j] * self.Cs[-(j+1)] @ (self.Cs[-(j+1)].H @ self.Cs[-1])
            # orthogonalize
            Cp = loewdin(Cp)
            # truncate wave function history
            self.Cs = self.Cs[1:]
            # store extrapolated value
            kset.C = Cp
            self._generate_density_potential(kset)

            res = super().update_and_find(pos)

            # Subspace alignment
            # C <- C U
            # where U = (O O^H)^(-1/2) O, O = C^H Cp
            # according to (11) in:
            # Steneteg, P., Abrikosov, I. A., Weber, V., & Niklasson, A. M. N.  Wave
            # function extended Lagrangian Born-Oppenheimer molecular dynamics. , 82(7),
            # 075110. http://dx.doi.org/10.1103/PhysRevB.82.075110
            C = kset.C
            Om = C.H @ Cp
            U = _solve(chol(Om@Om.H), Om)
            C_phase = C @ U
            kset.C = C_phase
            print('U', diag(U))
            print('U offdiag', l2norm(U-diag(diag(U))))
            print('aligned: %.5e' % l2norm(C_phase-C))
            print('unaligned: %.5e' % l2norm(C_phase-C))
            print('diff: %.5e' % l2norm(C_phase-C))
            # obtain current wave function coefficients
            C = kset.C
            self.Cs.append(C)

            return res

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
