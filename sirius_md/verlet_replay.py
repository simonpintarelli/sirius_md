import json
import yaml
import numpy as np
from .verlet import (initialize, Force, atom_masses, make_dft, atom_positions,
                     velocity_verlet, Logger, to_cart, from_cart)


def run(md_results_fname='md_results.json'):
    """
    """
    md_results = json.load(open(md_results_fname, 'r'))
    x_list = [d['x'] for d in md_results]
    v_list = [d['v'] for d in md_results]
    F_list = [d['F'] for d in md_results]

    # use different output name
    Logger().output = 'md_replay.json'

    input_vars = yaml.load(open('input.yml', 'r'))
    N = input_vars['parameters']['N']
    dt = input_vars['parameters']['dt']

    kset, _, _, dft_ = initialize(input_vars['parameters']['energy_tol'])

    dft = make_dft(dft_, input_vars)

    unit_cell = kset.ctx().unit_cell()
    lattice_vectors = np.array(unit_cell.lattice_vectors())

    Fh = Force(dft)

    x0 = atom_positions(unit_cell)
    F, EKS = Fh(x0)
    v0 = np.zeros_like(x0)
    na = len(x0)  # number of atoms
    atom_types = [unit_cell.atom(i).label for i in range(na)]

    # masses in A_r
    m = np.array([atom_masses[label] for label in atom_types])
    with Logger():
        # Velocity Verlet time-stepping
        for i, (xnext, vnext, Fnext) in enumerate(zip(x_list, v_list, F_list)):

            print('iteration: ', i, '\n')

            xn, vn, Fn, EKS = velocity_verlet(x0, v0, F, dt, Fh, m)
            print("displacement: %.2e" % np.linalg.norm(xn - x0))
            vc = to_cart(vn, lattice_vectors)
            ekin = 0.5 * np.sum(vc**2 * m[:, np.newaxis])
            print("Etot: %10.8f, Ekin: %10.8f, Eks: %10.8f" %
                  (EKS + ekin, ekin, EKS))
            Logger().insert({
                "i": i,
                "v": to_cart(vn, lattice_vectors),
                "x": to_cart(xn, lattice_vectors),
                "F": Fn,
                "E": EKS + ekin,
                "EKS": EKS,
                "ekin": ekin,
                't': i * dt
            })

            # load from input, and convert to fractional coordinates
            xnext = from_cart(xnext, lattice_vectors)
            vnext = from_cart(vnext, lattice_vectors)

            xerr = np.linalg.norm(xn-xnext, ord='fro')
            verr = np.linalg.norm(vn-vnext, ord='fro')
            Logger().insert({'xerr': xerr, 'verr': verr})
            x0 = xnext
            v0 = vnext
            F = np.array(Fnext)
