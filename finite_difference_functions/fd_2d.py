"""
Module that includes finite difference (1st/2nd) order schemes to
propagate Pizzo 2D hydrodynamic model of the solar wind functions

Authors: Opal Issan
Version: August 21, 2022
"""

import numpy as np
from tools.derivatives import ddx_fwd, ddx_bwd
from operator_functions.functions_2d import HdUdp, G_vector


def maccormack_pizzo_2d(U, dp, dr, r, theta):
    """MacCormack scheme (predictor, corrector steps)

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
    # predictor step
    U_pred = euler_pizzo_2d(U=U, dp=dp, dr=dr, r=r, theta=theta)

    # corrector step
    # derivative with respect to phi
    dUdp_pred = np.array([ddx_bwd(U_pred[0], dp, periodic=True),
                     ddx_bwd(U_pred[1], dp, periodic=True),
                     ddx_bwd(U_pred[2], dp, periodic=True),
                     ddx_bwd(U_pred[3], dp, periodic=True)])
    # H operator times dU/dp
    HdUdp_pred = HdUdp(U=U_pred, dUdp=dUdp_pred, r=r, theta=theta)
    # gravitational forces vector
    G_pred = G_vector(U=U_pred, r=r, theta=theta)
    # return corrector output
    return 0.5 * (U + U_pred) + 0.5 * dr * (HdUdp_pred + G_pred)


def euler_pizzo_2d(U, dp, dr, r, theta):
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
    G = G_vector(U=U, r=r, theta=theta)
    return U + dr * (G + H)
