"""
Module that includes Pizzo 3D hydrodynamic model of the solar wind functions


Authors: Opal Issan
Version: August 19, 2022
"""
import numpy as np
from astropy.constants import G
from astropy import constants as const
import astropy.units as u
import matplotlib.pyplot as plt


def cs_variable(P, rho, gamma=5/3):
    """cs = gamma * p / rho

    Parameters
    ----------
    P: 2d array
        pressure
    rho: 2d array
        density
    gamma: float
        polytropic index, default = 5/3.

    Returns
    -------
    2d array
        a 2D array of cs variable
    """
    return gamma * P / rho


def cg_variable(vt, vp, r, Ms=const.M_sun.to(u.kg).value, G=G.to(u.km ** 3 / (u.kg * u.s * u.s)).value):
    """cg = vt^2 + vp^2 - G*Ms/r

    Parameters
    ----------
    vt: 2d array
        velocity in latitude
    vp: 2d array
        velocity in longitude
    G: float
        gravitational constant
    Ms: float
        mass of the Sun.
    r: float
        radial distance from the Sun.

    Returns
    -------
    2d array
        a 2D array of cg variable
    """
    return (1/r) * (vt**2 + vp**2 - G*Ms/r)


def alpha_square_variable(ur, cs):
    """alpha^2 = vr^2 - cs^2

    Parameters
    ----------
    ur: 2d array
        radial velocity
    cs: 2d array
        cs = gamma * p / rho

    Returns
    -------
    2d array
        a 2D array of alpha^2 variable
    """
    return ur ** 2 - cs


def u_velocity(up, r, THETA, omega_rot=(2 * np.pi / (25.38 * 86400))):
    """u = vp - omega * r * sin(theta)

    Parameters
    ----------
    up: 2d array
        longitudinal velocity
    r: float
        radial distance from the Sun.
    omega_rot: float
        rotational rate of the Sun, default is set to 25.38 (days).

    Returns
    -------
    2d array
        a 2D array of u variable
    """
    return up - omega_rot * r * np.sin(THETA)


def c_vector(U, THETA, r):
    """c = [c1, c2, c3] coefficients.

    Parameters
    ----------
    U: tensor
        contains primitive variables [velocity radial, density, pressure, velocity longitudinal, velocity latitude]
    THETA: 2d array
        Theta mesh grid
    r: float
        radial distance from the Sun.

    Returns
    -------
    tensor
        a tensor with c coefficents
    """
    # primitive variables
    vr, rho, Pr, vp, vt = U

    # cotangant of theta
    cot = np.cos(THETA) / np.sin(THETA)

    # c- components
    C1 = (1 / r) * (-2 * rho * vr - rho * vt * cot)
    C2 = (1 / r) * (-vp * vr - vp * vt * cot)
    C3 = (1 / r) * ((vp ** 2) * cot - vr * vt)
    return np.array([C1, C2, C3])


def g_vector(U, THETA, r):
    """gravitational term

    Parameters
    ----------
    U: tensor
        contains primitive variables [velocity radial, density, pressure, velocity longitudinal, velocity latitude]
    THETA: 2d array
        Theta mesh grid
    r: float
        radial distance from the Sun.

    Returns
    -------
    tensor
        a tensor with G
    """
    # primitive variables
    vr, rho, Pr, vp, vt = U
    # cs^2
    cs = cs_variable(P=Pr, rho=rho)
    # cg
    cg = cg_variable(vt=vt, vp=vp, r=r)
    # 1/alpha^2
    alpha_coeff = 1 / alpha_square_variable(ur=vr, cs=cs)
    # c-vector elements
    C1, C2, C3 = c_vector(U=U, THETA=THETA, r=r)

    G1 = alpha_coeff * (vr * cg - cs * C1 / rho)
    G2 = alpha_coeff * (vr * C1 - rho * cg)
    G3 = alpha_coeff * (cs * vr * C1 - cs * rho * cg)
    G4 = alpha_coeff * (C2 / vr)
    G5 = alpha_coeff * (C3 / vr)
    return np.array([G1, G2, G3, G4, G5])


def FdUdt(U, dUdt, r):
    """F* dU/d(theta)

    Parameters
    ----------
    U: tensor
        contains primitive variables [velocity radial, density, pressure, velocity longitudinal, velocity latitude]
    dUdt: tensor
        contains primitive variables (U) first-derivative with respect to theta.
    r: float
        radial distance from the Sun.

    Returns
    -------
    tensor
        a tensor with F*dU/dt
    """
    # primitive variables
    vr, rho, Pr, vp, vt = U
    # cs
    cs = cs_variable(P=Pr, rho=rho)
    # alpha
    alpha = alpha_square_variable(ur=vr, cs=cs)
    # coefficient of F matrix
    coeff = -1 / (alpha * r)
    # F matrix times dUdt components
    F1 = coeff * (vt * vr * dUdt[0] - (vt / rho) * dUdt[2] - cs * dUdt[4])
    F2 = coeff * (-rho * vt * dUdt[0] + alpha * (vt / vr) * dUdt[1] + (vt / vr) * dUdt[2] + rho * vr * dUdt[4])
    F3 = coeff * (-cs * rho * vt * dUdt[0] + vt * vr * dUdt[2] + cs * rho * vr * dUdt[4])
    F4 = coeff * (alpha * (vt / vr) * dUdt[3])
    F5 = coeff * (alpha / (rho * vr) * dUdt[2] + alpha * (vt / vr) * dUdt[4])
    return np.array([F1, F2, F3, F4, F5])


def HdUdp(U, dUdp, r, THETA):
    """H* dU/d(phi)

    Parameters
    ----------
    U: tensor
        contains primitive variables [velocity radial, density, pressure, velocity longitudinal, velocity latitude]
    dUdp: tensor
        contains primitive variables (U) first-derivative with respect to phi.
    THETA: 2d array
        Theta mesh grid
    r: float
        radial distance from the Sun.

    Returns
    -------
    tensor
        a tensor with H*dU/dp

    """
    # primitive variables
    vr, rho, Pr, vp, vt = U
    # cs^2
    cs = cs_variable(P=Pr, rho=rho)
    # alpha
    alpha = alpha_square_variable(ur=vr, cs=cs)
    # coefficient of the H matrix
    coeff = -1 / (alpha * r * np.sin(THETA))
    # new velocity in the phi direction
    u = u_velocity(up=vp, r=r, THETA=THETA)
    # H * dUdp components
    H1 = coeff * (u * vr * dUdp[0] + -(u / rho) * dUdp[2] - cs * dUdp[3])
    H2 = coeff * (-rho * u * dUdp[0] + alpha * (u / vr) * dUdp[1] + (u / vr) * dUdp[2] + rho * vr * dUdp[3])
    H3 = coeff * (-cs * rho * u * dUdp[0] + u * vr * dUdp[2] + cs * rho * vr * dUdp[3])
    H4 = coeff * (alpha / (rho * vr) * dUdp[2] + alpha * (u / vr) * dUdp[3])
    H5 = coeff * (alpha * (u / vr) * dUdp[4])
    return np.array([H1, H2, H3, H4, H5])


def boundary_conditions(U):
    """enforce boundary conditions: dUdt = 0 on boundaries theta=0, pi.

     Parameters
    ----------
    U: tensor
        contains primitive variables [velocity radial, density, pressure, velocity longitudinal, velocity latitude]

    Returns
    -------
    tensor
        a tensor with U
    """
    # primitive variables
    vr, rho, Pr, vp, vt = U
    # left side (second order accuracy)
    vr[:, -1] = (-vr[:, -2] + 4 * vr[:, -3]) / 3
    rho[:, -1] = (-rho[:, -2] + 4 * rho[:, -3]) / 3
    Pr[:, -1] = (-Pr[:, -2] + 4 * Pr[:, -3]) / 3
    vp[:, -1] = (-vp[:, -2] + 4 * vp[:, -3]) / 3
    vt[:, -1] = (-vt[:, -2] + 4 * vt[:, -3]) / 3
    # right side (second order accuracy)
    vr[:, 0] = (-vr[:, 2] + 4 * vr[:, 1]) / 3
    rho[:, 0] = (-rho[:, 2] + 4 * rho[:, 1]) / 3
    Pr[:, 0] = (-Pr[:, 2] + 4 * Pr[:, 1]) / 3
    vp[:, 0] = (-vp[:, 2] + 4 * vp[:, 1]) / 3
    vt[:, 0] = (-vt[:, 2] + 4 * vt[:, 1]) / 3
    return np.array([vr, rho, Pr, vp, vt])

