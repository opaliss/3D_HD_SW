"""
Module that includes Pizzo 2D hydrodynamic model of the solar wind functions


Authors: Opal Issan
Version: August 19, 2022
"""
from tools.derivatives import ddx_fwd
import numpy as np
from astropy.constants import G
from astropy import constants as const
from operator_functions.functions_3d import cs_variable, alpha_square_variable
import astropy.units as u


def cg_variable(vp, r, Ms=const.M_sun.to(u.kg).value, G=G.to(u.km ** 3 / (u.kg * u.s * u.s)).value):
    """cg = vt^2 + vp^2 - G*Ms/r

    Parameters
    ----------
    vp: 1d array
        velocity in longitude
    G: float
        gravitational constant
    Ms: float
        mass of the Sun.
    r: float
        radial distance from the Sun.

    Returns
    -------
    1d array
        a 1D array of cg variable
    """
    return (1 / r.value) * (vp ** 2 - G * Ms / r.value)


def u_velocity(vp, r, theta, omega_rot=((2 * np.pi) / (25.38 * 86400) * (1 / u.s)).value):
    """u = vp - omega * r * sin(theta)

    Parameters
    ----------
    vp: 1d array
        longitudinal velocity
    r: float
        radial distance from the Sun.
    theta: float
        theta slice (radians).
    omega_rot: float
        rotational rate of the Sun, default is set to 25.38 (days).

    Returns
    -------
    1d array
        a 1D array of u variable
    """
    if np.sin(theta) != 0:
        return vp - omega_rot * r.value * np.sin(theta)
    else:
        return vp - omega_rot * r.value


def c_vector(U, r):
    """c = [c1, c2, c3] coefficients.

    Parameters
    ----------
    U: tensor
        contains primitive variables [velocity radial, density, pressure, velocity longitudinal, velocity latitude]
    r: float
        radial distance from the Sun.

    Returns
    -------
    tensor
        a tensor with c coefficents
    """
    # primitive variables
    vr, rho, pressure, vp = U
    # c-vector elements
    C1 = (1 / r.value) * (-2 * rho * vr)
    C2 = (1 / r.value) * (-vp * vr)
    return np.array([C1, C2])


def G_vector(U, r, theta):
    """gravitational term

    Parameters
    ----------
    U: tensor
        contains primitive variables [velocity radial, density, pressure, velocity longitudinal, velocity latitude]
    theta: float
        theta slice (radians).
    r: float
        radial distance from the Sun.

    Returns
    -------
    tensor
        a tensor with G
    """
    # primitive variables
    vr, rho, pressure, vp = U
    # cs^2
    cs = cs_variable(P=pressure, rho=rho)
    # alpha^2
    alpha = alpha_square_variable(ur=vr, cs=cs)
    # cg
    cg = cg_variable(vp=vp, r=r)
    # c-vector
    C1, C2 = c_vector(U=U, r=r)
    # g-vector elements
    G1 = (1 / alpha) * (vr * cg - cs * C1 / rho)
    G2 = (1 / alpha) * (vr * C1 - rho * cg)
    G3 = (1 / alpha) * (cs * vr * C1 - cs * rho * cg)
    G4 = (1 / alpha) * (C2 / vr)
    return np.array([G1, G2, G3, G4])


def HdUdp(dUdp, U, r, theta):
    """H* dU/d(phi)

    Parameters
    ----------
    U: tensor
        contains primitive variables [velocity radial, density, pressure, velocity longitudinal, velocity latitude]
    dUdp: tensor
        contains primitive variables (U) first-derivative with respect to phi.
    theta: float
        theta slice.
    r: float
        radial distance from the Sun.

    Returns
    -------
    tensor
        a tensor with H*dU/dp
    """
    # primitive variables
    vr, rho, pressure, vp = U
    # cs^2
    cs = cs_variable(P=pressure, rho=rho)
    # alpha^2
    alpha = alpha_square_variable(ur=vr, cs=cs)
    # H matrix coefficient
    if np.sin(theta) != 0:
        coeff = -1 / (r * alpha * np.sin(theta))
    else:
        coeff = -1 / (r * alpha)

    # u velocity
    u = u_velocity(vp=vp, r=r, theta=theta)

    # H dU/dp elements
    H1 = coeff * (u * vr * dUdp[0] - (u / rho) * dUdp[2] - cs * dUdp[3])
    H2 = coeff * (-rho * u * dUdp[0] + alpha * (u / vr) * dUdp[1] + (u / vr) * dUdp[2] + rho * vr * dUdp[3])
    H3 = coeff * (-cs * rho * u * dUdp[0] + u * vr * dUdp[2] + cs * rho * vr * dUdp[3])
    H4 = coeff * (alpha / (rho * vr) * dUdp[2] + alpha * (u / vr) * dUdp[3])
    return np.array([H1, H2, H3, H4])

