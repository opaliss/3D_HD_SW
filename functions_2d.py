from tools.derivatives import ddx_fwd, ddx_bwd, ddx_central
import numpy as np
from astropy.constants import G
from astropy import constants as const
from functions_3d import cs_variable, alpha_variable
import astropy.units as u
import matplotlib.pyplot as plt


def u_velocity(up, r, omega_rot=((2 * np.pi) / (25.38 * 86400) * (1 / u.s)).value):
    return up - omega_rot * r.value


def c_vector(U, r, Minf=const.M_sun.to(u.kg).value, G=G.to(u.km ** 3 / (u.kg * u.s * u.s)).value):
    ur, rho, Pr, up = U
    C1 = (1 / r.value) * (up ** 2 - G * Minf / r.value)
    C2 = (1 / r.value) * (-2 * rho * ur)
    C3 = np.zeros(C1.shape)
    C4 = (1 / r.value) * (-up * ur)
    return np.array([C1, C2, C3, C4])


def G_vector(U, r):
    ur, rho, Pr, up = U
    cs = cs_variable(P=Pr, rho=rho)
    alpha = alpha_variable(ur=ur, cs=cs)
    C1, C2, C3, C4 = c_vector(U=U, r=r)
    G1 = (1 / alpha) * (ur * C1 - cs * C2 / rho)
    G2 = (1 / alpha) * (ur * C2 - rho * C1)
    G3 = (1 / alpha) * (cs * ur * C2 - cs * rho * C1)
    G4 = (1 / alpha) * (C4 / ur)
    return np.array([G1, G2, G3, G4])


def HdUdp(dUdp, U, r):
    ur, rho, Pr, up = U
    cs = cs_variable(P=Pr, rho=rho)
    alpha = alpha_variable(ur=ur, cs=cs)
    coeff = -1 / (r * alpha)
    VP = u_velocity(up=up, r=r)
    H1 = coeff * (VP*ur*dUdp[0] - (VP/rho)*dUdp[2] - cs*dUdp[3])
    H2 = coeff * (-rho*VP*dUdp[0] + alpha*(VP/ur)*dUdp[1] + (VP/ur)*dUdp[2] + rho*ur*dUdp[3])
    H3 = coeff * (-cs*rho*VP*dUdp[0] + VP*ur*dUdp[2] + cs*rho*ur*dUdp[3])
    H4 = coeff * (alpha/(rho*ur)*dUdp[2] + alpha*(VP/ur)*dUdp[3])
    return np.array([H1, H2, H3, H4])



def pizzo_forward_euler_2d(U, dp, dr, r):
    # predictor
    # derivative with respect to phi
    dUdp = np.array([ddx_fwd(U[0], dp, periodic=True),
                     ddx_fwd(U[1], dp, periodic=True),
                     ddx_fwd(U[2], dp, periodic=True),
                     ddx_fwd(U[3], dp, periodic=True)])

    H1 = HdUdp(U=U, dUdp=dUdp, r=r)
    # gravitational forces vector
    G = G_vector(U=U, r=r)

    # predictor step
    U_pred = U + dr.value * (G + H1)
    return U_pred
