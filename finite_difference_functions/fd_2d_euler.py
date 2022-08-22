"""
Module that includes finite difference (1st) order schemes to
propagate Pizzo 2D hydrodynamic model of the solar wind functions

Authors: Opal Issan
Version: August 21, 2022
"""

import numpy as np
from tools.derivatives import ddx_fwd, ddx_bwd
from operator_functions.functions_2d import HdUdp, G_vector
import matplotlib.pyplot as plt


def forward_euler_pizzo_2d(U, dp, dr, r, theta):
    """forward Euler scheme q_{i+1} = q_{i} + dr * (dH/dp + G)

    Parameters
    ----------
    U: 2d array
        contains primitive variables = [vr, rho, pressure, vp]
    dp: float
        spacing in longitude grid (uniform).
    dr: float
        spacing in radial grid.
    r: float
        radial distance from the Sun.
    theta: float
        theta slice

    Returns
    -------
    2d array
        a 2D array of the flow quantities at (r + dr).
    """
    # derivative with respect to phi
    dUdp = np.array([ddx_fwd(U[0], dp, periodic=True),
                     ddx_fwd(U[1], dp, periodic=True),
                     ddx_fwd(U[2], dp, periodic=True),
                     ddx_fwd(U[3], dp, periodic=True)])
    # H operator times dU/dp
    H = HdUdp(U=U, dUdp=dUdp, r=r, theta=theta)
    # gravitational forces vector
    G = G_vector(U=U, r=r)
    return U + dr * (G + H)


def backward_euler_pizzo_2d(U, dp, dr, r, theta):
    """backward Euler scheme q_{i-1} = q_{i} + dr * (dH/dp + G)

    Parameters
    ----------
    U: 2d array
        contains primitive variables = [vr, rho, pressure, vp]
    dp: float
        spacing in longitude grid (uniform).
    dr: float
        spacing in radial grid.
    r: float
        radial distance from the Sun.
    theta: float
        theta slice

    Returns
    -------
    2d array
        a 2D array of the flow quantities at (r + dr).
    """
    # derivative with respect to phi
    dUdp = np.array([ddx_bwd(U[0], dp, periodic=True),
                     ddx_bwd(U[1], dp, periodic=True),
                     ddx_bwd(U[2], dp, periodic=True),
                     ddx_bwd(U[3], dp, periodic=True)])
    # H operator times dU/dp
    H = HdUdp(U=U, dUdp=dUdp, r=r, theta=theta)
    # gravitational forces vector
    G = G_vector(U=U, r=r)
    return U + dr * (G + H)
