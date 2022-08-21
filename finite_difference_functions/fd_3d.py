import numpy as np
from astropy.constants import G
from astropy import constants as const
import astropy.units as u
from tools.derivatives import ddx_fwd, ddx_bwd
from operator_functions.functions_3d import FdUdt, HdUdp, g_vector, boundary_conditions


def upwind(U, dt, dp, dr, THETA, r, order=1):
    dUdt = np.array([ddx_fwd(U[0].T, dt, order=order).T,
                     ddx_fwd(U[1].T, dt, order=order).T,
                     ddx_fwd(U[2].T, dt, order=order).T,
                     ddx_fwd(U[3].T, dt, order=order).T,
                     ddx_fwd(U[4].T, dt, order=order).T])

    V1 = FdUdt(U=U, dUdt=dUdt, r=r)

    # derivative with respect to phi
    dUdp = np.array([ddx_fwd(U[0], dp, periodic=True, order=order),
                     ddx_fwd(U[1], dp, periodic=True, order=order),
                     ddx_fwd(U[2], dp, periodic=True, order=order),
                     ddx_fwd(U[3], dp, periodic=True, order=order),
                     ddx_fwd(U[4], dp, periodic=True, order=order)])

    V2 = HdUdp(U=U, dUdp=dUdp, r=r, THETA=THETA)
    # gravitational forces vector
    G = g_vector(U=U, THETA=THETA, r=r)

    # predictor step
    U_pred = U + dr.value * (G + V1 + V2)
    U_pred = boundary_conditions(U=U_pred)
    return U_pred


def f_function(U, dt, dp, THETA, r, order=1):
    dUdt = np.array([ddx_fwd(U[0].T, dt, order=order).T,
                     ddx_fwd(U[1].T, dt, order=order).T,
                     ddx_fwd(U[2].T, dt, order=order).T,
                     ddx_fwd(U[3].T, dt, order=order).T,
                     ddx_fwd(U[4].T, dt, order=order).T])

    V1 = FdUdt(U=U, dUdt=dUdt, r=r)

    # derivative with respect to phi
    dUdp = np.array([ddx_fwd(U[0], dp, periodic=True, order=order),
                     ddx_fwd(U[1], dp, periodic=True, order=order),
                     ddx_fwd(U[2], dp, periodic=True, order=order),
                     ddx_fwd(U[3], dp, periodic=True, order=order),
                     ddx_fwd(U[4], dp, periodic=True, order=order)])

    V2 = HdUdp(U=U, dUdp=dUdp, r=r, THETA=THETA)
    # gravitational forces vector
    G = g_vector(U=U, THETA=THETA, r=r)
    return G + V1 + V2

def corrector(U, U_pred, dt, dp, dr, THETA, r):
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
    U_final = 0.5 * (U_pred + U + dr.value * (G_pred + V1_pred + V2_pred))
    U_final = boundary_conditions(U=U_final)
    return U_final


def pizzo_maccormack(U, dt, dp, dr, THETA, r, reg_coeff=1e-2):
    # predictor
    U_pred = upwind(U=U, dt=dt, dp=dp, dr=dr, THETA=THETA, r=r)
    # corrector
    U_final = corrector(U=U, U_pred=U_pred, dt=dt, dp=dp, dr=dr, THETA=THETA, r=r)
    # flux limiter
    phi = reg_coeff * phi_flux_limiter(q=U)
    U_sol = U_pred + phi * (U_final - U_pred)
    U_sol = boundary_conditions(U=U_sol)
    return U_sol, phi


def maccormack(U, dt, dp, dr, THETA, r):
    # predictor
    U_pred = upwind(U=U, dt=dt, dp=dp, dr=dr, THETA=THETA, r=r)
    # corrector
    U_final = corrector(U=U, U_pred=U_pred, dt=dt, dp=dp, dr=dr, THETA=THETA, r=r)
    return U_final


def flux_limiter_upwinds(U, dt, dp, dr, THETA, r):
    # upwind order 1
    U_pred_1 = upwind(U=U, dt=dt, dp=dp, dr=dr, THETA=THETA, r=r, order=1)

    # upwind order 2
    U_pred_2 = upwind(U=U, dt=dt, dp=dp, dr=dr, THETA=THETA, r=r, order=2)

    # # flux limiter
    phi = phi_flux_limiter(q=U)
    U_sol = U_pred_1 + phi * (U_pred_2 - U_pred_1)
    U_sol = boundary_conditions(U=U_sol)
    return U_sol


def minmod(theta):
    return max(0, min(1, theta))


def vanleer(theta):
    return (theta + abs(theta)) / (1 + abs(theta))


def phi_flux_limiter(q):
    flux_limiter = np.ones(q.shape)
    for quantity in range(q.shape[0]):
        for ii in range(q.shape[1] - 1):
            for jj in range(q.shape[2] - 1):
                if q[quantity, ii + 1, jj] == q[quantity, ii, jj]:
                    frac1 = 1
                elif q[quantity, ii - 1, jj] == q[quantity, ii, jj]:
                    frac1 = 1
                else:
                    frac1 = (q[quantity, ii, jj] - q[quantity, ii - 1, jj]) / (
                                q[quantity, ii + 1, jj] - q[quantity, ii, jj])
                if q[quantity, ii, jj + 1] == q[quantity, ii, jj]:
                    frac2 = 1
                elif q[quantity, ii, jj - 1] == q[quantity, ii, jj]:
                    frac2 = 1
                else:
                    frac2 = (q[quantity, ii, jj] - q[quantity, ii, jj - 1]) / (
                                q[quantity, ii, jj + 1] - q[quantity, ii, jj])
                flux_limiter[quantity, ii, jj] = minmod(theta=min(frac1, frac2))
    return flux_limiter
