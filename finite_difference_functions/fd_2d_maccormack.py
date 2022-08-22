"""
Module that includes finite difference (2nd) order schemes to
propagate Pizzo 2D hydrodynamic model of the solar wind functions

Authors: Opal Issan
Version: August 21, 2022
"""

import numpy as np
from tools.derivatives import ddx_fwd, ddx_bwd, d2dx2_central
from operator_functions.functions_2d import HdUdp, G_vector
from finite_difference_functions.fd_2d_euler import forward_euler_pizzo_2d
import matplotlib.pyplot as plt


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
    dUdp_corr = np.array([ddx_bwd(f=U_pred[0], dx=dp, periodic=True),
                          ddx_bwd(f=U_pred[1], dx=dp, periodic=True),
                          ddx_bwd(f=U_pred[2], dx=dp, periodic=True),
                          ddx_bwd(f=U_pred[3], dx=dp, periodic=True)])
    # H operator times dU/dp
    HdUdp_corr = HdUdp(U=U_pred, dUdp=dUdp_corr, r=r, theta=theta)
    # gravitational forces vector
    G_corr = G_vector(U=U_pred, r=r + dr)
    # return corrector output
    return 0.5 * (U + U_pred + dr * (HdUdp_corr + G_corr))


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
    dUdp = np.array([ddx_fwd(f=U[0], dx=dp, periodic=True),
                     ddx_fwd(f=U[1], dx=dp, periodic=True),
                     ddx_fwd(f=U[2], dx=dp, periodic=True),
                     ddx_fwd(f=U[3], dx=dp, periodic=True)])
    # H operator times dU/dp
    H = HdUdp(U=U, dUdp=dUdp, r=r, theta=theta)
    # gravitational forces vector
    G = G_vector(U=U, r=r)
    # viscosity
    vis = epsilon * np.array([d2dx2_central(f=U[0], dx=0, periodic=True),
                              d2dx2_central(f=U[1], dx=0, periodic=True),
                              d2dx2_central(f=U[2], dx=0, periodic=True),
                              d2dx2_central(f=U[3], dx=0, periodic=True)]) * viscosity(pressure=U[2])
    # predicted + viscosity
    U_pred = U + dr * (H + G) + vis

    # corrector step
    # derivative with respect to phi
    dUdp_corr = np.array([ddx_bwd(f=U_pred[0], dx=dp, periodic=True),
                          ddx_bwd(f=U_pred[1], dx=dp, periodic=True),
                          ddx_bwd(f=U_pred[2], dx=dp, periodic=True),
                          ddx_bwd(f=U_pred[3], dx=dp, periodic=True)])
    # H operator times dU/dp
    HdUdp_corr = HdUdp(U=U_pred, dUdp=dUdp_corr, r=r + dr, theta=theta)
    # gravitational forces vector
    G_corr = G_vector(U=U_pred, r=r + dr)
    # viscosity
    vis_corr = epsilon * np.array([d2dx2_central(U_pred[0], 0, periodic=True),
                         d2dx2_central(U_pred[1], 0, periodic=True),
                         d2dx2_central(U_pred[2], 0, periodic=True),
                         d2dx2_central(U_pred[3], 0, periodic=True)]) * viscosity(pressure=U_pred[2])

    # return corrector output
    U_corr = 0.5 * (U + U_pred + dr * (HdUdp_corr + G_corr)) + vis_corr
    return U_corr


def viscosity(pressure):
    """return modified MacCormack artificial viscosity term  Eq. 6.58 in
    https://www.airloads.net/Downloads/Textbooks/Computational-Fluid-Dynamics-the-Basics-With-Applications-Anderson-J-D.pdf

    Parameters
    ----------
    pressure: 1d array
        contains primitive variable pressure

    Returns
    -------
    1d array
        |p_{i+1} - 2*p_{i} + p_{i-1}| / p_{i+1} + 2*p_{i} + p_{i-1}
    """
    vec = np.zeros(len(pressure))
    for ii in range(len(pressure)):
        # enforce periodicity
        if ii == len(pressure) - 1:
            num = pressure[0] - 2 * pressure[ii] + pressure[ii - 1]
            den = pressure[0] + 2 * pressure[ii] + pressure[ii - 1]
        else:
            num = pressure[ii + 1] - 2 * pressure[ii] + pressure[ii - 1]
            den = pressure[ii + 1] + 2 * pressure[ii] + pressure[ii - 1]
        vec[ii] = np.abs(num) / den
    return vec
