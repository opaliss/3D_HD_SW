"""
Module that includes finite difference (2nd) order schemes to
propagate Pizzo 3D hydrodynamic model of the solar wind functions

Authors: Opal Issan
Version: August 22, 2022
"""

import numpy as np
from tools.derivatives import ddx_bwd
from finite_difference_functions.fd_3d_euler import forward_euler_pizzo_3d
from operator_functions.functions_3d import FdUdt, HdUdp, g_vector, boundary_conditions


def maccormack_pizzo_3d(U, dt, dp, dr, THETA, r):
    """maccormack scheme (3d) equations.

    Parameters
    ----------
    U: tensor
        contains primitive variables = [vr, rho, pressure, vp]
    dp: float
        spacing in longitude grid (uniform).
    dt: float
        spacing in latitude gird (uniform).
    dr: float
        spacing in radial grid.
    r: float
        radial distance from the Sun.
    THETA: 2d array
        THETA grid

    Returns
    -------
    tensor
        a tensor of the flow quantities at (r + dr).
    """
    # predictor
    U_pred = forward_euler_pizzo_3d(U=U, dt=dt, dp=dp, dr=dr, THETA=THETA, r=r)
    # corrector
    dUdt_pred = np.array([
        ddx_bwd(U_pred[0].T, dt).T,
        ddx_bwd(U_pred[1].T, dt).T,
        ddx_bwd(U_pred[2].T, dt).T,
        ddx_bwd(U_pred[3].T, dt).T,
        ddx_bwd(U_pred[4].T, dt).T])

    V1_pred = FdUdt(U=U_pred, dUdt=dUdt_pred, r=r + dr)

    # derivative with respect to phi
    dUdp_pred = np.array([
        ddx_bwd(U_pred[0], dp, periodic=True),
        ddx_bwd(U_pred[1], dp, periodic=True),
        ddx_bwd(U_pred[2], dp, periodic=True),
        ddx_bwd(U_pred[3], dp, periodic=True),
        ddx_bwd(U_pred[4], dp, periodic=True)])

    V2_pred = HdUdp(U=U_pred, dUdp=dUdp_pred, r=r + dr, THETA=THETA)
    # gravitational forces vector
    G_pred = g_vector(U=U_pred, THETA=THETA, r=r + dr)
    # corrector step
    U_final = 0.5 * (U_pred + U + dr * (G_pred + V1_pred + V2_pred))
    return boundary_conditions(U=U_final)
