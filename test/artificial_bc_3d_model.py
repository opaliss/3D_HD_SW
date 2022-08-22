"""Module that implements the artificial conditions at 30Rs in Pizzo 1979 (verification of implementation)

Authors: Opal Issan
Version: August 21, 2022
"""
import numpy as np
from astropy.constants import m_p, m_e
from finite_difference_functions.fd_3d_euler import forward_euler_pizzo_3d
import time
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'serif',
        'size': 14}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)

# set up grid
nphi = int(360 / 5)
nt = int(180 / 2)
nr = 100
p = np.linspace(0, 2 * np.pi, nphi)
t = np.linspace(-np.pi / 2, np.pi / 2, nt)
r = (np.linspace(35, 220, nr) * u.solRad).to(u.km)

PHI, THETA = np.meshgrid(p, np.linspace(0.05, np.pi - 0.05, len(t)))

# set up initial condition
u_max = 580
u_min = 290

A = (u_max - u_min) / u_min

f = np.zeros((PHI.T).shape)
for ii in range(len(p)):
    for jj in range(len(t)):
        L = np.sqrt((p[ii] - np.pi) ** 2 + (t[jj]) ** 2)
        f[ii, jj] = (np.sin(np.pi * L) / (np.pi * L)) ** 2
vr = u_min * (np.ones(PHI.T.shape) + A * f)

# plot initial condition
fig, ax = plt.subplots(figsize=(8, 5))
pos = ax.contourf(180 / np.pi * PHI, 180 / np.pi * THETA,
                  vr.T, cmap="viridis")
cbar = fig.colorbar(pos, ax=ax)
cbar.ax.set_ylabel(r'km/s', rotation=90, fontsize=14)
ax.set_title(r"$v_{r}$")

ax.set_xlabel("Longitude (Deg.)")
ax.set_ylabel("Latitude (Deg.)")
ax.set_xticks([0, 90, 180, 270, 360])
ax.set_yticks([0, 90, 180])

# initialize primitive variables
rho_0 = (370 * m_p / u.cm ** 3).to(u.kg / u.km ** 3)  # *m_p
p_0 = (1.14 * 1e-9 * u.dyne / (u.cm ** 2)).to(u.kg / (u.s ** 2 * u.km))
idx = 1

dr = r[1] - r[0]
dp = p[1] - p[0]
dt = t[1] - t[0]

U_SOL = np.zeros((5, len(p), len(t), len(r)))

U_SOL[:, :, :, 0] = np.array((vr,
                              rho_0.value * np.ones(vr.shape),
                              p_0.value * np.ones(vr.shape),
                              np.zeros(vr.shape),
                              np.zeros(vr.shape)))

# define mesh grid
PHI, THETA = np.meshgrid(p, np.linspace(0.05, np.pi - 0.05, len(t)))

jj = 0
# numerically propagte in the radial direction.
for ii in range(len(r) - 1):
    U_SOL[:, :, :, ii + 1] = forward_euler_pizzo_3d(U=U_SOL[:, :, :, ii],
                                                    dr=dr.value,
                                                    dp=dp,
                                                    dt=dt,
                                                    r=r[ii].value,
                                                    THETA=THETA.T)
    if ii % 25 == 0:
        fig, ax = plt.subplots(nrows=5, sharex=True, sharey=True, figsize=(5, 15))
        pos = ax[0].contourf(180 / np.pi * PHI, 180 / np.pi * THETA, U_SOL[0, :, :, ii + 1].T,
                             cmap="viridis")
        cbar = fig.colorbar(pos, ax=ax[0])
        cbar.ax.set_ylabel(r'km/s', rotation=90, fontsize=14)
        ax[0].set_title(r"$v_{r}$")

        pos = ax[1].contourf(180 / np.pi * PHI, 180 / np.pi * THETA,
                             (U_SOL[1, :, :, ii + 1].T / m_p.value * 1 / u.km ** 3).to(1 / u.cm ** 3),
                             cmap="viridis")
        cbar = fig.colorbar(pos, ax=ax[1])
        cbar.ax.set_ylabel(r'$\frac{1}{cm^3}$', rotation=90, fontsize=14)
        ax[1].set_title(r"$n_{\rho}$")

        pos = ax[2].contourf(180 / np.pi * PHI, 180 / np.pi * THETA,
                             (U_SOL[2, :, :, ii + 1].T * u.kg / (u.s ** 2 * u.km)).to(u.dyne / u.cm ** 2),
                             cmap="viridis")
        cbar = fig.colorbar(pos, ax=ax[2])
        cbar.ax.set_ylabel(r'$\frac{dyne}{cm^2}$', rotation=90, fontsize=14)
        ax[2].set_title(r"$P$")

        pos = ax[3].contourf(180 / np.pi * PHI, 180 / np.pi * THETA, U_SOL[3, :, :, ii + 1].T,
                             cmap="viridis")
        cbar = fig.colorbar(pos, ax=ax[3])
        cbar.ax.set_ylabel(r'km/s', rotation=90, fontsize=14)
        ax[3].set_title(r"$v_{\phi}$")

        pos = ax[4].contourf(180 / np.pi * PHI, 180 / np.pi * THETA, U_SOL[4, :, :, ii + 1].T,
                             cmap="viridis")
        cbar = fig.colorbar(pos, ax=ax[4])
        cbar.ax.set_ylabel(r'km/s', rotation=90, fontsize=14)
        ax[4].set_title(r"$v_{\theta}$")
        ax[4].set_xticks([0, 90, 180, 270, 360])
        ax[4].set_yticks([0, 90, 180])
        ax[4].set_xlabel("Longitude (Deg.)")
        ax[4].set_ylabel("Latitude (Deg.)")
        ax[3].set_ylabel("Latitude (Deg.)")
        ax[2].set_ylabel("Latitude (Deg.)")
        ax[1].set_ylabel("Latitude (Deg.)")
        ax[0].set_ylabel("Latitude (Deg.)")
        fig.suptitle("r = " + str(round(((r[ii]).to(u.AU)).value, 2)) + "AU")
        fig.suptitle("r = " + str(round((r[ii]).to(u.AU).value, 1)) + "AU")
        plt.tight_layout()
        plt.show()
        jj += 1
        # fig.savefig("figs/artificial_bc/fig_" + str(jj) + ".png", dpi=400)
        plt.close()
