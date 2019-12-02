"""Wrappers for SIRIUS DFT_ground_state class"""
import numpy as np
from scipy.special import binom

from .dft_direct_minimizer import OTMethod, MVP2Method
from sirius import set_atom_positions, spdiag, l2norm
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
def cholesky(X):
    return la.cholesky(X)


def is_insulator(fn):
    @threaded
    def _is_insulator(fn):
        return np.linalg.norm(fn-np.mean(fn))
    # check if bands are not constant
    return np.sum(_is_insulator(fn)) < 1e-8


def align_subspace(C, Cp):
    """Aligns subspaces.

    Computes: U = argmin_Z || C@Z - Cp ||
    and returns C@U.

    U is given by (O@ O.H)^(-1/2) O,
    where O = C.H@Cp.

    U can be computed using svd:
    W, s, Vh = svd(O),
    then U = W@Vh.

    For derivation see: http://dx.doi.org/10.1103/PhysRevB.45.1538.

    Arguments:
    C  -- wave function
    Cp -- wave function
    """

    # Arias, T. A., Payne, M. C., & Joannopoulos, J. D.,
    # Ab initio molecular-dynamics techniques extended to large-length-scale systems,
    # 45(4), 1538–1549.
    # http://dx.doi.org/10.1103/PhysRevB.45.1538
    # See Appendix A: subspace alignment

    Om = C.H @ Cp
    U, _, Vh = Om.svd(full_matrices=False)
    C_phase = C @ (U @ Vh)
    # print('U offdiag', l2norm(U-diag(diag(U))))
    print('aligned: %.5e' % l2norm(C_phase-Cp))
    print('unaligned: %.5e' % l2norm(C-Cp))
    # obtain current wave function coefficients
    return C_phase


def align_occupied_subspace(C, Cp, fn):
    """Aligns only occupied part of subspace.

    Computes: U = argmin_Z || C@Z - Cp ||
    and returns C@U.

    U is given by (O@ O.H)^(-1/2) O,
    where O = C.H@Cp.

    U can be computed using svd:
    W, s, Vh = svd(O),
    then U = W@Vh.

    For derivation see: http://dx.doi.org/10.1103/PhysRevB.45.1538.

    Arguments:
    C  -- wave function
    Cp -- wave function
    """
    # Arias, T. A., Payne, M. C., & Joannopoulos, J. D.,
    # Ab initio molecular-dynamics techniques extended to large-length-scale systems,
    # 45(4), 1538–1549.
    # http://dx.doi.org/10.1103/PhysRevB.45.1538
    # See Appendix A: subspace alignment

    if is_insulator(fn):
        return align_subspace(C, Cp)

    from copy import deepcopy
    C_phase = deepcopy(C)

    for k in C.keys():
        Ck = C[k]
        Cpk = Cp[k]
        fnk = fn[k]

        ids, = np.where(np.isclose(fnk, 0))
        jocc = ids[0]  # first empty band

        # assert all occupation numbers are equal
        assert np.linalg.norm(fnk[:jocc]-np.mean(fnk[:jocc])) < 1e-9

        Om = Ck[:, :jocc].H @ Cpk[:, :jocc]
        U, _, Vh = np.linalg.svd(Om, full_matrices=False)
        C_phase[k][:, :jocc] = C_phase[k][:, :jocc] @ (U @ Vh)

    return C_phase


class DftGroundState:
    """plain SCF. No extrapolation"""

    def __init__(self, solver, **kwargs):
        self.dft_obj = solver
        self.potential_tol = kwargs["potential_tol"]
        self.energy_tol = kwargs["energy_tol"]
        self.maxiter = kwargs["maxiter"]

    def _generate_density_potential(self, kset):
        density = self.dft_obj.density()
        potential = self.dft_obj.potential()
        ctx = kset.ctx()

        density.generate(kset)
        density.fft_transform(1)

        potential.generate(density)
        potential.fft_transform(1)

    def update_and_find(self, pos, tol=None):
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

        print('DEBUG:: sum(fn) = %.9f' % np.sum(kset.fn))
        print('DEBUG:: fn',  kset.fn)

        return self.dft_obj.find(
            potential_tol=self.potential_tol if tol is None else tol,
            energy_tol=self.energy_tol if tol is None else tol,
            initial_tol=1e-2,
            num_dft_iter=self.maxiter,
            write_state=False,
        )


def Bm(K, j):
    """Extrapolation coefficients from Kolafa 0 < j < K+2"""
    return (-1)**(j+1) * j * binom(2*K + 2, K+1-j) / binom(2*K, K)


class DftWfExtrapolate(DftGroundState):
    """extrapolate wave functions."""

    def __init__(self, solver, order=3, **kwargs):
        super().__init__(solver, **kwargs)
        self.Cs = [self.dft_obj.k_point_set().C]
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
        if len(self.Cs) >= self.order+1:
            print('extrapolate')
            # this is Eq (19) from:
            # Kolafa, J., Time-reversible always stable predictor–corrector method
            #             for molecular dynamics of polarizable molecules,
            # 25(3), 335–342 ().  http://dx.doi.org/10.1002/jcc.10385
            Cp = self.Bm[0] * self.Cs[-1]
            for j in range(1, len(self.Bm)):
                # print('Bm', 'j', j, ':', self.Bm[j])
                Cp += self.Bm[j] * self.Cs[-(j+1)] @ (self.Cs[-(j+1)].H @ self.Cs[-1])
            # orthogonalize
            Cp = loewdin(Cp)
            # truncate wave function history
            self.Cs = self.Cs[1:]
            # store extrapolated value
            align_occupied_subspace(Cp, kset.C, kset.fn)
            kset.C = Cp
            self._generate_density_potential(kset)
            res = super().update_and_find(pos)

            # Subspace alignment
            C = kset.C
            print('align C,Cp')
            C_phase = align_occupied_subspace(C, Cp, kset.fn)
            kset.C = C_phase
            omega = (self.order+1) / (2*self.order + 1)
            # apply corrector and append to history
            print('align to previous iterate..')
            self.Cs.append(align_occupied_subspace(loewdin(omega*C_phase + (1-omega)*Cp), self.Cs[-1], kset.fn))

            return res

        # initial steps with higher tolerance
        res = super().update_and_find(pos, tol=1e-9)
        C = kset.C
        self.Cs.append(align_occupied_subspace(C, self.Cs[-1], kset.fn))
        return res


class NiklassonWfExtrapolate(DftGroundState):
    """Niklasson wave function extrapolation.

    Steneteg, P., Abrikosov, I. A., Weber, V., & Niklasson, A. M. N.,
    Wave function extended Lagrangian Born-Oppenheimer molecular dynamics,
    82(7), 075110
    http://dx.doi.org/10.1103/PhysRevB.82.075110
    """

    def __init__(self, solver, order, **kwargs):
        super().__init__(solver, **kwargs)
        self.Cps = [self.dft_obj.k_point_set().C]
        self.order = order

        # Niklasson, A. M. N., Steneteg, P., Odell, A., Bock, N., Challacombe, M., Tymczak, C. J., Holmström, E.,
        # Extended Lagrangian Born–Oppenheimer molecular dynamics with dissipation,
        # 130(21), 214109 ().  http://dx.doi.org/10.1063/1.3148075
        self.coeffs = {
            0: {'kappa': 2, 'a': 0, 'c': []},
            3: {'kappa': 1.69, 'a': 0.15, 'c': [-2, 3, 0, -1]},
            4: {'kappa': 1.75, 'a': 0.057, 'c': [-3, 6, -2, -2, 1]},
            5: {'kappa': 1.82, 'a': 0.018, 'c': [-6, 14, -8, -3, 4, -1]},
            6: {'kappa': 1.84, 'a': 0.0055, 'c': [-14, 36, -27, -2, 12, -6, 1]},
            7: {'kappa': 1.86, 'a': 0.0016, 'c': [-36, 99, -88, 11, 32, -25, 8, -1]},
            8: {'kappa': 1.88, 'a': 0.00044, 'c': [-99, 286, -286, 78, 78, -90, 42, -10, 1]},
            9: {'kappa': 1.89, 'a': 0.00012, 'c': [-286, 858, -936, 364, 168, -300, 184, -63, 12, -1]}
        }

        if order not in self.coeffs:
            raise ValueError('invalid order given.')

    def update_and_find(self, pos):
        """
        Arguments:
        pos -- atom positions in reduced coordinates
        """

        kset = self.dft_obj.k_point_set()
        if len(self.Cps) >= max(2, self.order+1):
            print('niklasson extrapolate')
            C = kset.C
            CU = align_subspace(C, self.Cps[-1])
            Cp = 2*self.Cps[-1] - self.Cps[-2] + self.coeffs[self.order]['kappa']*(CU-self.Cps[-1])
            cm = self.coeffs[self.order]['c']
            if self.order > 0:
                for i in range(self.order+1):
                    # others
                    Cp += self.coeffs[self.order]['a'] * cm[i] * self.Cps[-(i+1)]
            Cp = loewdin(Cp)

            # append history
            self.Cps = self.Cps[1:] + [Cp, ]

            kset.C = Cp
            res = super().update_and_find(pos)
            return res

        # not enough previous values to extrapolate
        res = super().update_and_find(pos, tol=1e-9)
        C = kset.C

        # subspace alignment for initial, non-extrapolated steps
        self.Cps.append(align_subspace(C, self.Cps[-1]))
        return res


def make_dft(solver, parameters):
    """DFT object factory.

    Arguments:
    solver     -- plain DFT_ground_state from SIRIUS
    parameters -- parameter dictionary
    """

    maxiter = parameters["parameters"]["maxiter"]
    potential_tol = parameters["parameters"]["potential_tol"]
    energy_tol = parameters["parameters"]["energy_tol"]

    # replace solver if OTMethod or MVP2 is used
    if "solver" in parameters["parameters"]:
        if parameters["parameters"]["solver"] == "ot":
            solver = OTMethod(solver)
        if parameters["parameters"]["solver"] == "mvp2":
            # this one is for metallic systems, only implemented for magnetic sytems
            solver = MVP2Method(solver)

    if parameters["parameters"]["method"]["type"] == "plain":
        return DftGroundState(
            solver,
            energy_tol=energy_tol,
            potential_tol=potential_tol,
            maxiter=maxiter,
        )
    if parameters["parameters"]["method"]["type"] == "kolafa":
        order = parameters["parameters"]["method"]["order"]
        return DftWfExtrapolate(
            solver,
            order=order,
            energy_tol=energy_tol,
            potential_tol=potential_tol,
            maxiter=maxiter,
        )
    if parameters["parameters"]["method"]["type"] == "niklasson_wf":
        order = parameters["parameters"]["method"]["order"]
        return NiklassonWfExtrapolate(
            solver,
            order=order,
            energy_tol=energy_tol,
            potential_tol=potential_tol,
            maxiter=maxiter,
        )

    raise ValueError("invalid extrapolation method")
