"""Module that uses the 3D MAS coronal solutions for the primitive
 variables at 30Rs and propagates forward to 1.1AU.

Authors: Opal Issan
Version: August 22, 2022
"""
from tools.MASweb import get_mas_path
from psipy.model import MASOutput
from scipy.interpolate import RegularGridInterpolator
from finite_difference_functions.fd_3d_maccormack import maccormack_pizzo_3d
from finite_difference_functions.fd_3d_euler import forward_euler_pizzo_3d
import numpy as np
from astropy.constants import m_p
import astropy.units as u

import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'serif',
        'size': 14}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)

# GLOBAL UNITS: KM, S, KG.

# load data
# mas output - br, rho, vr
cr = "2210"  # carrington rotation number.
mas_path = get_mas_path(cr=cr)
model = MASOutput(mas_path)
print(model.variables)

# interpolate so all quantities are on the same mesh
pp, tt, rr = np.meshgrid(model["vp"].phi_coords, model["vp"].theta_coords, model["vp"].r_coords, indexing='ij')
coordinate_grid = np.array([pp.T, tt.T, rr.T]).T

# linear interpolation
interp_function = RegularGridInterpolator(
    points=(model["vt"].phi_coords, model["vt"].theta_coords, model["vt"].r_coords),
    values=np.array(model["vt"].data), bounds_error=False, fill_value=None)

data_vt = interp_function(coordinate_grid)

# save MHD mesh coordinates Heliographic (rotating) Coordinate System (HG)
# phi - (0, 2pi)
p = model["vr"].phi_coords

# delta phi
dp = p[1] - p[0]

# theta - (-pi/2, pi/2)
t = model["vr"].theta_coords
t = np.linspace(0.05, np.pi - 0.05, len(t))
# delta t
dt = t[1] - t[0]

# 30 solar radii to approximately 1 AU, 1 solar radii = 695,700 km
r = (model["vr"].r_coords * u.solRad).to(u.km)
new_r = np.linspace(r[0], r[-1], int(300))
# change in r
dr = new_r[1] - new_r[0]

# since the last phi index is less than 2*pi, then we will append 2*pi to phi scale.
p = np.append(p, 2 * np.pi)
vr = model["vr"].data * (u.km / u.s)
vr = np.append(vr, [vr[0, :, :]], axis=0)

vp = model["vp"].data * (u.km / u.s)
vp = np.append(vp, [vp[0, :, :]], axis=0)

vt = data_vt * (u.km / u.s)
vt = np.append(vt, [vt[0, :, :]], axis=0)

rho = np.array(model["rho"].data) * m_p  # multiply by kg
rho = (rho * (1 / u.cm ** 3)).to(u.kg / u.km ** 3).value  # convert to mks (km)
rho = np.append(rho, [rho[0, :, :]], axis=0)

Pr = np.array(model["p"].data)
Pr = ((Pr * (u.dyne / u.cm ** 2)).to(u.kg / (u.s ** 2 * u.km))) * 1e-1  # convert to mks (km)
Pr = np.append(Pr, [Pr[0, :, :]], axis=0)

PHI, THETA = np.meshgrid(p, t)

U_SOL = np.zeros((5, len(p), len(t), len(new_r)))
U_SOL[:, :, :, 0] = np.array((vr[:, :, 0] * (1 + 0.25 * (1 - np.exp(-30 / 50))),
                              rho[:, :, 0],
                              Pr[:, :, 0],
                              vp[:, :, 0] * (1 + 10 * (1 - np.exp(-30 / 50))),
                              vt[:, :, 0] * (1 + 10 * (1 - np.exp(-30 / 50)))))

for ii in range(len(new_r) - 1):
    if ii % 10 == 0:
        print("debug ==> " + str(ii))
    U_SOL[:, :, :, ii + 1] = forward_euler_pizzo_3d(U=U_SOL[:, :, :, ii],
                                                    dr=dr.value,
                                                    dp=dp,
                                                    dt=dt,
                                                    r=new_r[ii].value,
                                                    THETA=THETA.T)
    if ii % 10 == 0:
        print(ii)
        print((new_r[ii]).to(u.AU))
        fig, ax = plt.subplots(nrows=5, sharex=True, sharey=True, figsize=(5, 10))
        pos = ax[0].pcolormesh(180 / np.pi * PHI, 180 / np.pi * THETA, U_SOL[0, :, :, ii + 1].T, shading='gouraud',
                               cmap="viridis")
        cbar = fig.colorbar(pos, ax=ax[0])
        cbar.ax.set_ylabel(r'km/s', rotation=90, fontsize=14)
        ax[0].set_title(r"$v_{r}$")

        pos = ax[1].pcolormesh(180 / np.pi * PHI, 180 / np.pi * THETA,
                               (U_SOL[1, :, :, ii + 1].T / m_p.value * 1 / u.km ** 3).to(1 / u.cm ** 3),
                               shading='gouraud',
                               cmap="viridis")
        cbar = fig.colorbar(pos, ax=ax[1])
        cbar.ax.set_ylabel(r'$\frac{1}{cm^3}$', rotation=90, fontsize=14)
        ax[1].set_title(r"$n_{\rho}$")

        pos = ax[2].pcolormesh(180 / np.pi * PHI, 180 / np.pi * THETA,
                               (U_SOL[2, :, :, ii + 1].T * u.kg / (u.s ** 2 * u.km)).to(u.dyne / u.cm ** 2),
                               shading='gouraud',
                               cmap="viridis")
        cbar = fig.colorbar(pos, ax=ax[2])
        cbar.ax.set_ylabel(r'$\frac{dyne}{cm^2}$', rotation=90, fontsize=14)
        ax[2].set_title(r"$P$")

        pos = ax[3].pcolormesh(180 / np.pi * PHI, 180 / np.pi * THETA, U_SOL[3, :, :, ii + 1].T, shading='gouraud',
                               cmap="viridis")
        cbar = fig.colorbar(pos, ax=ax[3])
        cbar.ax.set_ylabel(r'km/s', rotation=90, fontsize=14)
        ax[3].set_title(r"$v_{\phi}$")

        pos = ax[4].pcolormesh(180 / np.pi * PHI, 180 / np.pi * THETA, U_SOL[4, :, :, ii + 1].T, shading='gouraud',
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
        plt.tight_layout()
        plt.show()
        plt.close()
