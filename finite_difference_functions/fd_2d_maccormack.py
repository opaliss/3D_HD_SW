"""
Module that includes finite difference (2nd) order schemes to
propagate Pizzo 2D hydrodynamic model of the solar wind functions

Authors: Opal Issan
Version: August 21, 2022
"""

import numpy as np
from tools.derivatives import ddx_fwd, ddx_bwd, d2dx2_central, ddx_central
from operator_functions.functions_2d import HdUdp, G_vector
from finite_difference_functions.fd_2d_euler import forward_euler_pizzo_2d
import matplotlib.pyplot as plt
import astropy.units as u


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
    U_pred = forward_euler_pizzo_2d(U=U, dp=dp, dr=dr, r=r, theta=theta)

    # corrector step
    # derivative with respect to phi
    dUdp_pred = np.array([ddx_bwd(U_pred[0], dp, periodic=True),
                          ddx_bwd(U_pred[1], dp, periodic=True),
                          ddx_bwd(U_pred[2], dp, periodic=True),
                          ddx_bwd(U_pred[3], dp, periodic=True)])
    # H operator times dU/dp
    HdUdp_pred = HdUdp(U=U_pred, dUdp=dUdp_pred, r=r, theta=theta)
    # gravitational forces vector
    G_pred = G_vector(U=U, r=r + (dr*u.km)/2, theta=theta)
    # return corrector output
    return 0.5 * (U + U_pred) + 0.5 * dr * (HdUdp_pred + G_pred)


def modified_maccormack_pizzo_2d(U, dp, dr, r, theta, epsilon):
    """MacCormack scheme with viscous term d2U/dp2 (predictor, corrector steps)

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
    epsilon: float
        artificial viscosity

    Returns
    -------
    2d array
        a 2D array of the flow quantities at (r + dr).
    """
    # predictor step
    dUdp = np.array([ddx_fwd(U[0], dp, periodic=True),
                     ddx_fwd(U[1], dp, periodic=True),
                     ddx_fwd(U[2], dp, periodic=True),
                     ddx_fwd(U[3], dp, periodic=True)])
    # H operator times dU/dp
    H = HdUdp(U=U, dUdp=dUdp, r=r, theta=theta)
    # gravitational forces vector
    G = G_vector(U=U, r=r, theta=theta)
    # viscosity
    vis = np.array([d2dx2_central(U[0], 0, periodic=True),
                    d2dx2_central(U[1], 0, periodic=True),
                    d2dx2_central(U[2], 0, periodic=True),
                    d2dx2_central(U[3], 0, periodic=True)])
    vis = epsilon * vis * viscosity(pressure=U[2])

    U_pred = U + dr * (H + G) + vis

    # corrector step
    # derivative with respect to phi
    dUdp_pred = np.array([ddx_bwd(U_pred[0], dp, periodic=True),
                          ddx_bwd(U_pred[1], dp, periodic=True),
                          ddx_bwd(U_pred[2], dp, periodic=True),
                          ddx_bwd(U_pred[3], dp, periodic=True)])
    # H operator times dU/dp
    HdUdp_pred = HdUdp(U=U_pred, dUdp=dUdp_pred, r=r + dr/2 * u.km, theta=theta)
    # gravitational forces vector
    G_pred = G_vector(U=U_pred, r=r + dr/2 * u.km, theta=theta)
    # viscosity
    vis_pred = np.array([d2dx2_central(U_pred[0], 0, periodic=True),
                        d2dx2_central(U_pred[1], 0, periodic=True),
                        d2dx2_central(U_pred[2], 0, periodic=True),
                        d2dx2_central(U_pred[3], 0, periodic=True)])
    vis_pred = epsilon * vis_pred * viscosity(pressure=U_pred[2])
    # return corrector output
    U_corr = 0.5 * (U + U_pred) + 0.5 * dr * (HdUdp_pred + G_pred) + vis_pred
    return U_corr


def viscosity(pressure):
    """return modified MacCormack & Baldwin pressure term.

    """
    vec = np.zeros(len(pressure))
    for ii in range(len(pressure)):
        if ii == len(pressure) - 1:
            num = pressure[0] - 2 * pressure[ii] + pressure[ii - 1]
            den = pressure[0] + 2 * pressure[ii] + pressure[ii - 1]
        else:
            num = pressure[ii + 1] - 2 * pressure[ii] + pressure[ii - 1]
            den = pressure[0] + 2 * pressure[ii] + pressure[ii - 1]
        vec[ii] = np.abs(num)/den
    return vec