"""Module includes finite difference operators


Authors: Opal Issan
Version: August 19, 2022
"""

import numpy as np
from scipy.sparse import diags


def ddx_fwd(f, dx, periodic=True, order=1):
    """ return the first derivative of f(x) in x using a first-order forward difference.
    --- assuming a uniform mesh discretization in x, i.e. dx = const. ---
    df/dx = (f_{i+1} - f_{i})/dx (for order=1)

     Parameters
    ----------
    f: array
        a 2d or 1d array
    dx: float
        the spatial discretization of x, i.e. dx = x_{i+1} - x_{i}
    periodic: bool
        periodic boundary conditions (True/False).
    order: float
        order of accuracy.

    Returns
    -------
    array
        df/dx
    """
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
    """
    return the first derivative of f(x) in x using a first-order backwards difference.
    --- assuming a uniform mesh discretization in x, i.e. dx = const. ---
    df/dx = (f_{i} - f_{i-1})/dx (for order=1)

     Parameters
    ----------
    f: array
        a 2d or 1d array
    dx: float
        the spatial discretization of x, i.e. dx = x_{i+1} - x_{i}
    periodic: bool
        periodic boundary conditions (True/False).
    order: float
        order of accuracy.

    Returns
    -------
    array
        df/dx
    """
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
    """
    return the first derivative of f(x) in x using a second-order accurate central difference.
    --- assuming a uniform mesh discretization in x, i.e. dx = const. ---
    df/dx = (f_{i+1} - f_{i-1})/(2*dx)

     Parameters
    ----------
    f: array
        a 2d or 1d array
    dx: float
        the spatial discretization of x, i.e. dx = x_{i+1} - x_{i}
    periodic: bool
        periodic boundary conditions (True/False).

    Returns
    -------
    array
        df/dx
    """
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


def d2dx2_central(f, dx, periodic=False):
    """
    return the second derivative of f(x) in x using a second-order accurate central difference.
    --- assuming a uniform mesh discretization in x, i.e. dx = const. ---
    d2f/dx2 = (f_{i+1} - 2f_{i} + f_{i-1})/(dx^2)

    on the boundaries (for non-periodic):
    (forward)
    d2f/dx2 = 2f_{i} - 5 f_{i+1} + 4 f_{i+2} - f_{i+3}/(dx^3)
    (backward)
    d2f/dx2 = 2f_{i} - 5f_{i-1} + 4 f_{i-2} -f_{i-3}/(dx^3)

     Parameters
    ----------
    f: array
        a 2d or 1d array
    dx: float
        the spatial discretization of x, i.e. dx = x_{i+1} - x_{i}
    periodic: bool
        periodic boundary conditions (True/False).

    Returns
    -------
    array
        df/dx
    """
    A = diags([-1, 0, 1], [1, -2,  1], shape=(f.shape[0], f.shape[0])).toarray()
    if periodic:
        A[0, -1] = 1
        A[-1, 0] = 1
    else:
        # forward
        A[0, 0] = 2/dx
        A[0, 1] = -5/dx
        A[0, 2] = 4/dx
        A[0, 3] = -1/dx
        # backward
        A[-1, -1] = 2/dx
        A[-1, -2] = -5/dx
        A[-1, -3] = 4/dx
        A[-1, -4] = -1/dx
    A /= (dx**2)
    return A @ f