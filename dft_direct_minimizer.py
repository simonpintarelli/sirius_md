from sirius.ot.minimize import minimize
from sirius.ot import Energy, ApplyHamiltonian, c
from sirius.ot import ConstrainedGradient, make_kinetic_precond
from sirius import DFT_ground_state_find, get_c0_x


class OTMethod:
    """Orbital transformation method adaptor."""
    def __init__(self, dft_obj):
        self.dft_obj = dft_obj
        self.kset = dft_obj.k_point_set()
        potential = dft_obj.potential()
        density = dft_obj.density()
        # Hamiltonian, provides gradient H|Ψ>
        self.H = ApplyHamiltonian(potential, self.kset)
        # create object to compute the total energy
        self.E = Energy(self.kset, potential, density, self.H)


    def find(self, energy_tol, num_dft_iter, **kwargs):
        c0, x = get_c0_x(self.kset)
        """Find ground state by CG iterations."""

        # prepare a simple kinetic preconditioner
        M = make_kinetic_precond(self.kset, c0, asPwCoeffs=True, eps=1e-3)

        # run NLCG
        x, niter, success, histE = minimize(
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

        return {'converged': success, 'num_scf_iterations': niter, 'band_gap': -1, 'energy': {'total': histE[-1]}}

    def density(self):
        """return SIRIUS density obj."""
        return self.dft_obj.density()

    def potential(self):
        """return SIRIUS potential obj."""
        return self.dft_obj.potential()


    def forces(self):
        """returns SIRIUS forces obj."""
        return self.dft_obj.forces()

    def update(self):
        return self.dft_obj.update()

    def k_point_set(self):
        return self.dft_obj.k_point_set()
