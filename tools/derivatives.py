import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
import scipy
import time
from scipy.sparse import diags


def ddx_fwd(f, dx, periodic=False):
    # return the first derivative of f in x using a first-order forward difference.
    A = diags([-1, 1], [0, 1], shape=(f.shape[0], f.shape[0])).toarray()
    if periodic:
        A[-1, 0] = 1
    else:
        A[-1, -1] = 1
        A[-1, -2] = -1
    A /= dx
    return A @ f


def ddx_bwd(f, dx, periodic=False):
    # return the first derivative of f in x using a first-order backward difference.
    A = diags([-1, 1], [-1, 0], shape=(f.shape[0], f.shape[0])).toarray()
    if periodic:
        A[0, -1] = -1
        A /= dx
    else:
        A[0, 0] = -1
        A[0, 1] = 1
    A /= dx
    return A @ f

def ddx_central(f, dx, periodic=False):
    # return the first derivative of f in x using a first-order central difference.
    A = diags([-1, 1], [-1, 1], shape=(f.shape[0], f.shape[0])).toarray()
    if periodic:
        A[0, -1] = -1
        A[-1, 0] = 1
    else:
        A[0, 0] = -3
        A[0, 1] = 4
        A[0, 2] = -1
        A[-1, -1] = 3
        A[-1, -2] = -4
        A[-1, -3] = 1
    A /= (2 * dx)
    return A @ f