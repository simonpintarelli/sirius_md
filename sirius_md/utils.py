import numpy as np
from sirius import DFT_ground_state_find
from sirius import CoefficientArray
from mpi4py import MPI
import json
import h5py
import logging as log

def to_cart(x, L):
    assert L.shape == (3, 3)
    return x @ L.T


def from_cart(x, L):
    assert L.shape == (3, 3)
    return x @ np.linalg.inv(L).T


def sirius_save_state(objs_dict, prefix):
    """
    Save SIRIUS CoefficientArrays to hdf5.

    Arguments:
    objs_dict -- dictionary(string: CoefficientArray), example: {'Z': Z, 'G': G}
    kset      -- SIRIUS kpointset
    prefix    --
    """
    rank = MPI.COMM_WORLD.rank

    with h5py.File(prefix + "%d.h5" % rank, "w") as fh5:
        for key in objs_dict:
            # assume it is a string
            name = key
            sirius_save_h5(fh5, name, objs_dict[key])


def sirius_save_h5(fh5, label, obj):
    """
    Arguments:
    fh5  -- h5py.File
    label -- name for the object
    obj  -- np.array like / CoefficientArray
    """

    if isinstance(obj, CoefficientArray):
        grp = fh5.create_group(label)
        for key, val in obj.items():
            dset = grp.create_dataset(
                name=",".join(map(str, key)), shape=val.shape, dtype=val.dtype, data=val
            )
            dset.attrs["key"] = key
    else:
        grp = fh5.create_dataset(name=label, data=obj)
    return grp


def sirius_load_state(prefix, name, dst):
    """
    Arguments:
    prefix  -- path/to/prefix*.hdf5 file
    name      -- name of the dataset to be loaded
    dst       -- obj of type CoefficientArray or PwCoeffs
    """
    import glob

    rank = MPI.COMM_WORLD.rank

    with h5py.File(prefix + "%d.h5" % rank, "r") as fh5:
        for key in fh5[name].keys():
            kp_index, spin_index = tuple(fh5[name][key].attrs["key"])
            dst[(kp_index, spin_index)] = dst.ctype(fh5[name][key])
    return dst


class ScfConvergenceError(Exception):
    """Custom exception."""


def initialize(tol=None, atom_positions=None, num_dft_iter=10000):
    """Initialize DFT_ground_state object."""
    sirius_config = json.load(open("sirius.json", "r"))

    if tol is not None:
        sirius_config["parameters"]["density_tol"] = tol
        sirius_config["parameters"]["energy_tol"] = tol
    else:
        sirius_config["parameters"]["density_tol"] = 1e-14
        sirius_config["parameters"]["energy_tol"] = 1e-14

    if atom_positions:
        sirius_config["unit_cell"]["atom_coordinate_units"] = "au"
        for atom in sirius_config["unit_cell"]["atoms"]:
            n = len(sirius_config["unit_cell"]["atoms"][atom])
            positions = [atom_positions.pop(0) for _ in range(n)]
            sirius_config["unit_cell"]["atoms"][atom] = positions
    
    log.debug(f"number of dft iterations: {num_dft_iter}")
    res = DFT_ground_state_find(num_dft_iter, config=sirius_config)

    if not res["E"]["converged"]:
        raise ScfConvergenceError

    return res["kpointset"], res["density"], res["potential"], res["dft_gs"]
