"""
Module that includes finite difference (1st) order schemes to
propagate Pizzo 3D hydrodynamic model of the solar wind functions

Authors: Opal Issan
Version: August 22, 2022
"""
import numpy as np
from tools.derivatives import ddx_fwd, ddx_bwd
from operator_functions.functions_3d import FdUdt, HdUdp, g_vector, boundary_conditions


def forward_euler_pizzo_3d(U, dt, dp, dr, THETA, r):
    """forward Euler scheme q_{i+1} = q_{i} + dr * (H * dq_{i}/dp + F * dq_{i}/dt  + G)

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
    dUdt = np.array([ddx_fwd(f=U[0].T, dx=dt).T,
                     ddx_fwd(f=U[1].T, dx=dt).T,
                     ddx_fwd(f=U[2].T, dx=dt).T,
                     ddx_fwd(f=U[3].T, dx=dt).T,
                     ddx_fwd(f=U[4].T, dx=dt).T])
    # F * dU/dt
    V1 = FdUdt(U=U, dUdt=dUdt, r=r)

    # derivative with respect to phi
    dUdp = np.array([ddx_fwd(f=U[0], dx=dp, periodic=True),
                     ddx_fwd(f=U[1], dx=dp, periodic=True),
                     ddx_fwd(f=U[2], dx=dp, periodic=True),
                     ddx_fwd(f=U[3], dx=dp, periodic=True),
                     ddx_fwd(f=U[4], dx=dp, periodic=True)])
    # H * dU/dp
    V2 = HdUdp(U=U, dUdp=dUdp, r=r, THETA=THETA)

    # gravitational forces vector
    G = g_vector(U=U, THETA=THETA, r=r)

    # predictor step
    U_pred = U + dr * (G + V1 + V2)
    return boundary_conditions(U=U_pred)


def backward_euler_pizzo_3d(U, dt, dp, dr, THETA, r):
    """backward Euler scheme q_{i-1} = q_{i} + dr * (H * dq_{i}/dp + F * dq_{i}/dt  + G)

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
    dUdt = np.array([ddx_bwd(f=U[0].T, dx=dt).T,
                     ddx_bwd(f=U[1].T, dx=dt).T,
                     ddx_bwd(f=U[2].T, dx=dt).T,
                     ddx_bwd(f=U[3].T, dx=dt).T,
                     ddx_bwd(f=U[4].T, dx=dt).T])
    # F * dU/dt
    V1 = FdUdt(U=U, dUdt=dUdt, r=r)

    # derivative with respect to phi
    dUdp = np.array([ddx_bwd(f=U[0], dx=dp, periodic=True),
                     ddx_bwd(f=U[1], dx=dp, periodic=True),
                     ddx_bwd(f=U[2], dx=dp, periodic=True),
                     ddx_bwd(f=U[3], dx=dp, periodic=True),
                     ddx_bwd(f=U[4], dx=dp, periodic=True)])
    # H * dU/dp
    V2 = HdUdp(U=U, dUdp=dUdp, r=r, THETA=THETA)

    # gravitational forces vector
    G = g_vector(U=U, THETA=THETA, r=r)

    # predictor step
    U_pred = U + dr * (G + V1 + V2)
    return boundary_conditions(U=U_pred)
