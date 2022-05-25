import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
import scipy
import time
from scipy.sparse import diags


def ddx_fwd(f, dx, periodic=True, order=1):
    # return the first derivative of f in x using a first-order forward difference.
    if order == 1:
        A = diags([-1, 1], [0, 1], shape=(f.shape[0], f.shape[0])).toarray()
        if periodic:
            A[-1, 0] = 1
        else:
            A[-1, -1] = 1
            A[-1, -2] = -1
        A /= dx
    elif order == 2:
        A = diags([-3/2, 2, -1/2], [0, 1, 2], shape=(f.shape[0], f.shape[0])).toarray()
        if periodic:
            A[-1, 0] = 2
            A[-1, 1] = -1/2
            A[-2, 0] = -1/2
        else:
            A[-1, -1] = 1/2
            A[-1, -2] = -2
            A[-1, -3] = 3/2
            A[-2, -1] = 1/2
            A[-2, -2] = -1/2
    elif order == 3:
        A = diags([-11/6, 3, -3/2, 1/3], [0, 1, 2, 3], shape=(f.shape[0], f.shape[0])).toarray()
        if periodic:
            A[-1, 0] = 3
            A[-1, 1] = -3/2
            A[-1, 2] = 1/3
            A[-2, 0] = -3/2
            A[-2, 1] = 1/3
        else:
            return ArithmeticError
        A /= (dx)
    return A @ f


def ddx_bwd(f, dx, periodic=False, order=1):
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