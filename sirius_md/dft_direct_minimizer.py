"""Wrappers exposing interface find(..) like SIRIUS C++ DFT_ground_state does."""

from sirius.ot.minimize import minimize
from sirius.ot import Energy, ApplyHamiltonian, c
from sirius.ot import ConstrainedGradient, make_kinetic_precond
from sirius import get_c0_x
import numpy as np

from sirius import Logger as pprinter
pprint = pprinter()


class solver_base:
    def __init__(self, dft_obj):
        self.dft_obj = dft_obj

    def density(self):
        """Return SIRIUS density obj."""
        return self.dft_obj.density()

    def potential(self):
        """Return SIRIUS potential obj."""
        return self.dft_obj.potential()

    def forces(self):
        """Returns SIRIUS forces obj."""
        return self.dft_obj.forces()

    def update(self):
        """Call DFT_ground_state.update"""
        return self.dft_obj.update()

    def k_point_set(self):
        """Return SIRIUS k-point set."""
        return self.dft_obj.k_point_set()

    def initial_state(self):
        """Return SIRIUS k-point set."""
        self.dft_obj.initial_state()

    def serialize(self):
        """Return SIRIUS k-point set."""
        return self.dft_obj.serialize()


class OTMethod(solver_base):
    """Orbital transformation method adaptor."""
    def __init__(self, dft_obj):
        super().__init__(dft_obj)
        self.kset = dft_obj.k_point_set()
        potential = dft_obj.potential()
        density = dft_obj.density()
        # Hamiltonian, provides gradient H|Ψ>
        self.H = ApplyHamiltonian(potential, self.kset)
        # create object to compute the total energy
        self.E = Energy(self.kset, potential, density, self.H)

    def find(self, energy_tol, num_dft_iter, **_):
        """Find ground state by the orbital transformation method."""
        c0, x = get_c0_x(self.kset)
        # prepare a simple kinetic preconditioner
        M = make_kinetic_precond(self.kset, c0, asPwCoeffs=True, eps=1e-3)
        # run NLCG
        _, niter, success, histE = minimize(
            x,
            f=lambda x: self.E(c(x, c0)),
            df=ConstrainedGradient(self.H, c0),
            maxiter=num_dft_iter,
            restart=10,
            mtype='PR',
            verbose=True,
            log=True,
            M=M,
            tol=float(energy_tol))

        return {'converged': success, 'num_scf_iterations': niter,
                'band_gap': -1, 'energy': {'total': histE[-1]}}


class MVP2Method(solver_base):
    """Marzari-Vanderbilt-Payne pseudo-Hamiltonian method."""
    def __init__(self, dft_obj):
        super().__init__(dft_obj)

        self.kset = dft_obj.k_point_set()
        potential = dft_obj.potential()
        density = dft_obj.density()
        self.H = ApplyHamiltonian(potential, self.kset)
        # create object to compute the total energy
        self.E = Energy(self.kset, potential, density, self.H)

    def find(self, energy_tol, num_dft_iter, **_):
        """Find ground state by the orbital transformation method."""
        from sirius.edft import NeugebaurCG as CG, FreeEnergy, make_fermi_dirac_smearing
        from sirius.edft.preconditioner import make_kinetic_precond2
        import time

        T = 300
        ctx = self.kset.ctx()

        smearing = make_fermi_dirac_smearing(T, ctx, self.kset)

        M = FreeEnergy(E=self.E, T=T, smearing=smearing)
        cg = CG(M)
        K = make_kinetic_precond2(self.kset)

        def make_callback(histE):
            def _callback(**kwargs):
                histE.append(kwargs['FE'])
            return _callback

        tstart = time.time()
        X = self.kset.C
        fn = self.kset.fn
        histE = []
        X, fn, FE, success = cg.run(X,
                                    fn,
                                    tol=energy_tol,
                                    K=K,
                                    maxiter=num_dft_iter,
                                    kappa=0.3,
                                    restart=10,
                                    cgtype='FR',
                                    tau=0.1,
                                    callback=make_callback(histE))
        tstop = time.time()
        pprint('MVP2 took: ', tstop-tstart, ' seconds')
        pprint('number of steps found by callback:', len(histE))

        return {
            'converged': success,
            'num_scf_iterations': len(histE),
            'band_gap': -1,
            'energy': {
                'total': FE
            }
        }
