from tools.derivatives import ddx_fwd, ddx_bwd
import numpy as np
from astropy.constants import G
from astropy import constants as const
import astropy.units as u
import matplotlib.pyplot as plt


def cs_variable(P, rho, gamma=5 / 3):
    return gamma * P / rho


def alpha_variable(ur, cs):
    return ur**2 - cs


def u_velocity(up, r, THETA, omega_rot=(2 * np.pi / (25.38 * 86400))):
    return up - omega_rot * r.value * np.sin(THETA)


def c_vector(U, THETA, r, Minf=const.M_sun.to(u.kg).value, G=G.to(u.km ** 3 / (u.kg * u.s * u.s)).value):
    # primitive variables
    vr, rho, Pr, vp, vt = U
    # cotangant of theta
    cot = np.cos(THETA) / np.sin(THETA)
    coeff = 1 / r.value
    # c- components
    C1 = coeff * (vt ** 2 + vp ** 2 - (G * Minf) / r.value)
    C2 = coeff * (-2 * rho * vr - rho * vt * cot)
    C3 = np.zeros(C1.shape)
    C4 = coeff * (-vp * vr - vp * vt * cot)
    C5 = coeff * ((vp ** 2) * cot - vr * vt)
    return np.array([C1, C2, C3, C4, C5])


def g_vector(U, THETA, r):
    # primitive variables
    vr, rho, Pr, vp, vt = U
    # cs^2
    cs = cs_variable(P=Pr, rho=rho)
    # 1/alpha^2
    alpha_coeff = 1 / alpha_variable(ur=vr, cs=cs)

    C1, C2, C3, C4, C5 = c_vector(U=U, THETA=THETA, r=r)

    G1 = alpha_coeff * (vr * C1 - cs * C2 / rho)
    G2 = alpha_coeff * (vr * C2 - rho * C1)
    G3 = alpha_coeff * (cs * vr * C2 - cs * rho * C1)
    G4 = alpha_coeff * (C4 / vr)
    G5 = alpha_coeff * (C5 / vr)
    return np.array([G1, G2, G3, G4, G5])


def FdUdt(U, dUdt, r):
    # primitive variables
    vr, rho, Pr, vp, vt = U
    # cs
    cs = cs_variable(P=Pr, rho=rho)
    # alpha
    alpha = alpha_variable(ur=vr, cs=cs)
    # coefficient of F matrix
    coeff = -1 / (alpha * r.value)
    # F matrix times dUdt components
    F1 = coeff * (vt * vr * dUdt[0] - (vt / rho) * dUdt[2] - cs * dUdt[4])
    F2 = coeff * (-rho * vt * dUdt[0] + alpha * (vt / vr) * dUdt[1] + (vt / vr) * dUdt[2] + rho * vr * dUdt[4])
    F3 = coeff * (-cs * rho * vt * dUdt[0] + vt * vr * dUdt[2] + cs * rho * vr * dUdt[4])
    F4 = coeff * (alpha * (vt / vr) * dUdt[3])
    F5 = coeff * (alpha / (rho * vr) * dUdt[2] + alpha * (vt / vr) * dUdt[4])
    return np.array([F1, F2, F3, F4, F5])


def HdUdp(U, dUdp, r, THETA):
    # primitive variables
    vr, rho, Pr, vp, vt = U
    # cs
    cs = cs_variable(P=Pr, rho=rho)
    # alpha
    alpha = alpha_variable(ur=vr, cs=cs)
    # coefficient of the H matrix
    coeff = -1 / (alpha * r.value * np.sin(THETA))
    # new velocity in the phi direction
    VP = u_velocity(up=vp, r=r, THETA=THETA)
    # H * dUdp components
    H1 = coeff * (VP * vr * dUdp[0] + -(VP / rho) * dUdp[2] - cs * dUdp[3])
    H2 = coeff * (-rho * VP * dUdp[0] + alpha * (VP / vr) * dUdp[1] + (VP / vr) * dUdp[2] + rho * vr * dUdp[3])
    H3 = coeff * (-cs * rho * VP * dUdp[0] + VP * vr * dUdp[2] + cs * rho * vr * dUdp[3])
    H4 = coeff * (alpha / (rho * vr) * dUdp[2] + alpha * (VP / vr) * dUdp[3])
    H5 = coeff * (alpha * (VP / vr) * dUdp[4])
    return np.array([H1, H2, H3, H4, H5])

def boundary_conditions(U):
    # primitive variables
    vr, rho, Pr, vp, vt = U
    # left side second order
    vr[:, -1] = (-vr[:, -2] + 4 * vr[:, -3])/3
    rho[:, -1] = (-rho[:, -2] + 4 * rho[:, -3])/3
    Pr[:, -1] = (-Pr[:, -2] + 4 * Pr[:, -3])/3
    vp[:, -1] = (-vp[:, -2] + 4 * vp[:, -3])/3
    vt[:, -1] = (-vt[:, -2] + 4 * vt[:, -3])/3
    # right side second order
    vr[:, 0] = (-vr[:, 2] + 4 * vr[:, 1])/3
    rho[:, 0] = (-rho[:, 2] + 4 * rho[:, 1])/3
    Pr[:, 0] = (-Pr[:, 2] + 4 * Pr[:, 1])/3
    vp[:, 0] = (-vp[:, 2] + 4 * vp[:, 1])/3
    vt[:, 0] = (-vt[:, 2] + 4 * vt[:, 1])/3
    return np.array([vr, rho, Pr, vp, vt])


def pizzo_maccormack(U, dt, dp, dr, THETA, r):
    # predictor
    # derivative with respect to theta
    dUdt = np.array([ddx_fwd(U[0].T, dt).T,
                     ddx_fwd(U[1].T, dt).T,
                     ddx_fwd(U[2].T, dt).T,
                     ddx_fwd(U[3].T, dt).T,
                     ddx_fwd(U[4].T, dt).T])

    V1 = FdUdt(U=U, dUdt=dUdt, r=r)

    # derivative with respect to phi
    dUdp = np.array([ddx_fwd(U[0], dp, periodic=True),
                     ddx_fwd(U[1], dp, periodic=True),
                     ddx_fwd(U[2], dp, periodic=True),
                     ddx_fwd(U[3], dp, periodic=True),
                     ddx_fwd(U[4], dp, periodic=True)])

    V2 = HdUdp(U=U, dUdp=dUdp, r=r, THETA=THETA)
    # gravitational forces vector
    G = g_vector(U=U, THETA=THETA, r=r)

    # predictor step
    U_pred = U + dr.value * (G + V1 + V2)
    U_pred = boundary_conditions(U=U_pred)

    # corrector
    # dUdt_pred = np.array([
    #     ddx_bwd(U_pred[0].T, dt).T,
    #     ddx_bwd(U_pred[1].T, dt).T,
    #     ddx_bwd(U_pred[2].T, dt).T,
    #     ddx_bwd(U_pred[3].T, dt).T,
    #     ddx_bwd(U_pred[4].T, dt).T])
    #
    # V1_pred = FdUdt(U=U_pred, dUdt=dUdt_pred, r=r + dr/2)
    #
    # # derivative with respect to phi
    # dUdp_pred = np.array([
    #     ddx_bwd(U_pred[0], dp, periodic=True),
    #     ddx_bwd(U_pred[1], dp, periodic=True),
    #     ddx_bwd(U_pred[2], dp, periodic=True),
    #     ddx_bwd(U_pred[3], dp, periodic=True),
    #     ddx_bwd(U_pred[4], dp, periodic=True)])
    #
    # V2_pred = HdUdp(U=U_pred, dUdp=dUdp_pred, r=r + dr, THETA=THETA)
    # # gravitational forces vector
    # G_pred = g_vector(U=U_pred, THETA=THETA, r=r + dr)
    # # corrector step
    # U_final = 0.5 * (U_pred + U + dr.value * (G_pred + V1_pred + V2_pred))

    dUdt = np.array([ddx_fwd(U[0].T, dt, order=2).T,
                     ddx_fwd(U[1].T, dt, order=2).T,
                     ddx_fwd(U[2].T, dt, order=2).T,
                     ddx_fwd(U[3].T, dt, order=2).T,
                     ddx_fwd(U[4].T, dt, order=2).T])

    V1 = FdUdt(U=U, dUdt=dUdt, r=r)

    # derivative with respect to phi
    dUdp = np.array([ddx_fwd(U[0], dp, periodic=True, order=2),
                     ddx_fwd(U[1], dp, periodic=True, order=2),
                     ddx_fwd(U[2], dp, periodic=True, order=2),
                     ddx_fwd(U[3], dp, periodic=True, order=2),
                     ddx_fwd(U[4], dp, periodic=True, order=2)])

    V2 = HdUdp(U=U, dUdp=dUdp, r=r, THETA=THETA)
    # gravitational forces vector
    G = g_vector(U=U, THETA=THETA, r=r)

    # runge kutta
    U_pred_star = U + (dr.value)/2 * (G + V1 + V2)

    # dUdt = np.array([ddx_fwd(U_pred_star[0].T, dt, order=2).T,
    #                  ddx_fwd(U_pred_star[1].T, dt, order=2).T,
    #                  ddx_fwd(U_pred_star[2].T, dt, order=2).T,
    #                  ddx_fwd(U_pred_star[3].T, dt, order=2).T,
    #                  ddx_fwd(U_pred_star[4].T, dt, order=2).T])
    #
    # V1 = FdUdt(U=U_pred_star, dUdt=dUdt, r=r + (dr)/2)
    #
    # # derivative with respect to phi
    # dUdp = np.array([ddx_fwd(U_pred_star[0], dp, periodic=True, order=2),
    #                  ddx_fwd(U_pred_star[1], dp, periodic=True, order=2),
    #                  ddx_fwd(U_pred_star[2], dp, periodic=True, order=2),
    #                  ddx_fwd(U_pred_star[3], dp, periodic=True, order=2),
    #                  ddx_fwd(U_pred_star[4], dp, periodic=True, order=2)])
    #
    # V2 = HdUdp(U=U_pred_star, dUdp=dUdp, r=r + (dr)/2, THETA=THETA)
    # # gravitational forces vector
    # G = g_vector(U=U_pred_star, THETA=THETA, r=r + (dr)/2)
    #
    # U_pred_2 = U + (dr.value) * (G + V1 + V2)

    # # flux limiter
    phi = phi_flux_limiter(q=U)
    U_sol = U_pred + phi*(U_pred_star - U_pred)
    U_sol = boundary_conditions(U=U_sol)
    return U_sol


def minmod(theta):
    return max(0, min(1, theta))


def vanleer(theta):
    return (theta + abs(theta))/(1 + abs(theta))


def phi_flux_limiter(q):
    flux_limiter = np.ones(q.shape)
    for quantity in range(q.shape[0]):
        for ii in range(q.shape[1]-1):
            for jj in range(q.shape[2]-1):
                if q[quantity, ii+1, jj] == q[quantity, ii, jj]:
                    frac1 = 1
                else:
                    frac1 = (q[quantity, ii, jj] - q[quantity, ii-1, jj])/(q[quantity, ii+1, jj] - q[quantity, ii, jj])
                if q[quantity, ii, jj+1] == q[quantity, ii, jj]:
                    frac2 = 1
                else:
                    frac2 = (q[quantity, ii, jj] - q[quantity, ii, jj-1])/(q[quantity, ii, jj+1] - q[quantity, ii, jj])
                flux_limiter[quantity, ii, jj] = minmod(theta=min(frac1, frac2))
    return flux_limiter

