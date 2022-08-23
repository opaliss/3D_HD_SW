"""module to solve for the time stepping that satisfies the CFL condition.

Authors: Opal Issan
Version: August 22, 2022
"""
import numpy as np
from operator_functions.functions_2d import u_velocity, cs_variable


def lam_1(u, r, theta, vr):
    """lam_{1} = u / (r * sin(theta) * vr)

    Parameters
    ----------
    vr: 1d/2d array
        radial velocity
    u: 1d/2d array
        u = vp - omega * r * sin(theta)
    r: float
        radial distance from the Sun.
    theta: float/1d array
        theta slice or grid

    Returns
    -------
    1d/2d array
        a 2D array of lam_{1}
    """
    return u / (r * np.sin(theta) * vr)


def lam_2(vr, u, cs, theta, r):
    """lam_{2}(plus/minus) = vr*u +- sqrt(vr^2 + u^2 -cs^2) / (r sin(theta) (vr^2 -cs^2)

    Parameters
    ----------
    vr: 1d/2d array
        radial velocity
    u: 1d/2d array
        u = vp - omega * r * sin(theta)
    cs: 1d/2d
        cs = gamma * P / rho
    r: float
        radial distance from the Sun.
    theta: float/1d array
        theta slice or grid

    Returns
    -------
    tensor
        a 2D array of lam_{2}
    """
    sqrt_var = np.sqrt(vr**2 + u**2 - cs)
    denom = vr**2 - cs
    lam_2_plus = (vr * u + cs * sqrt_var)/(r * np.sin(theta) * denom)
    lam_2_minus = (vr * u - cs * sqrt_var) / (r * np.sin(theta) * denom)
    return lam_2_plus, lam_2_minus


def epsilon_1(vt, r, vr):
    """epsilon= vt/(r*vr)
    Parameters
    ----------
    vt: 1d/2d array
        latitudinal velocity
    vr: 1d/2d array
        radial velocity
    r: float
        radial distance from the Sun.

    Returns
    -------
    1d/2d array
        a 2D array of epsilon_{1}
    """
    return vt/(r*vr)


def epsilon_2(vr, vt, cs, r):
    """epsilon_{2}(plus/minus) = (vr*vt +- cs^2 \sqrt(vr^2 + vt^2 - cs^2))/(vr^2 -cs^2)r

    Parameters
    ----------
    vr: 1d/2d array
        radial velocity
    vt: 1d/2d array
        latitudinal velocity
    cs: 1d/2d
        cs = gamma * P / rho
    r: float
        radial distance from the Sun.

    Returns
    -------
    tensor
        a 2D array of lam_{1}
    """
    sqrt_var = vr**2 + vt**2 - cs
    epsilon_2_plus = (vr * vt + cs * sqrt_var) / ((vr ** 2 - cs) * r)
    epsilon_2_minus = (vr * vt - cs * sqrt_var) / ((vr ** 2 - cs) * r)
    return epsilon_2_plus, epsilon_2_minus


def max_dr_2d(U, r, dp, theta):
    """return upper limit for radial stepping

    Parameters
    ----------
    U: tensor
        contains primitive variables [velocity radial, density, pressure, velocity longitudinal, velocity latitude]
    r: float
        radial distance from the Sun.
    dp: float
        longitude grid spacing (uniform).
    theta: float/1d array
        theta slice or grid

    Returns
    -------
    float
        a float with max dr
    """
    vr, rho, pressure, vp = U
    u = u_velocity(vp=vp, r=r, theta=theta)
    cs = cs_variable(P=pressure, rho=rho)
    val1 = dp/np.max(lam_1(u=u, r=r, theta=theta, vr=vr))
    val2 = dp/np.max(lam_2(u=u, r=r, theta=theta, vr=vr, cs=cs)[0])
    val3 = dp/np.max(lam_2(u=u, r=r, theta=theta, vr=vr, cs=cs)[1])
    return np.max([val1, val2, val3])