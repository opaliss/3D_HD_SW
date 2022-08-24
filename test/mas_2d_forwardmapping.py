"""Module that uses the (2D) equatorial slice MAS coronal solutions
for the primitive variables at 30RS and propagates forward to 1.1AU.

Authors: Opal Issan
Version: August 22, 2022
"""

from tools.MASweb import get_mas_path
from psipy.model import MASOutput
from finite_difference_functions.fd_2d_euler import forward_euler_pizzo_2d
import numpy as np
from astropy.constants import m_p
import matplotlib.pyplot as plt
import matplotlib
import astropy.units as u

font = {'family': 'serif',
        'size': 14}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)

# GLOBAL UNITS: KM, S, KG.
# load data
# mas output - br, rho, vr
cr = "2210"
mas_path = get_mas_path(cr=cr)
model = MASOutput(mas_path)
print(model.variables)

# save MHD mesh coordinates Heliographic (rotating) Coordinate System (HG)
# phi - (0, 2pi)
p = model["vr"].phi_coords

# delta phi
dp = p[1] - p[0]

# 30 solar radii to approximately 1 AU, 1 solar radii = 695,700 km
r = (model["vr"].r_coords * u.solRad).to(u.km)
new_r = r.value
# change in r
dr = (new_r[1] - new_r[0])

# since the last phi index is less than 2*pi, then we will append 2*pi to phi scale.
p = np.append(p, 2 * np.pi)
vr = model["vr"].data * (u.km / u.s)
vr = np.append(vr, [vr[0, :, :]], axis=0)
# boost velocity from HUX paper.
vr_ic = vr + 0.15 * vr * (1 - np.exp(-30/50))

vp = model["vp"].data * (u.km / u.s)
vp = np.append(vp, [vp[0, :, :]], axis=0)

rho = np.array(model["rho"].data) * m_p  # multiply by kg
rho = (rho * (1 / u.cm ** 3)).to(u.kg / u.km ** 3).value  # convert to mks (km)
rho = np.append(rho, [rho[0, :, :]], axis=0)

Pr = np.array(model["p"].data)
Pr = ((Pr * (u.dyne / u.cm ** 2)).to(u.kg / (u.s ** 2 * u.km)))  # convert to mks (km)
Pr = np.append(Pr, [Pr[0, :, :]], axis=0)

U_SOL_E = np.zeros((4, len(p), len(new_r)))
U_SOL_E[:, :, 0] = np.array((vr_ic[:, 55, 0], rho[:, 55, 0], Pr[:, 55, 0], vp[:, 55, 0]))

for ii in range(len(new_r) - 1):
    U_SOL_E[:, :, ii + 1] = forward_euler_pizzo_2d(U=U_SOL_E[:, :, ii],
                                                   dr=np.abs(new_r[ii + 1] - new_r[ii]),
                                                   dp=dp, r=new_r[ii], theta=np.pi / 2)

sample_columns = np.arange(0, len(new_r), int(len(new_r) // 5))
sample_columns = np.append(sample_columns, len(new_r) - 1)
fig, ax = plt.subplots(4, 1, figsize=(10, 15), sharex=True)

color = iter(plt.cm.viridis_r(np.linspace(0, 1, len(sample_columns))))

for j in sample_columns:
    curr_color = next(color)
    # vr
    ax[0].plot(180 / np.pi * p, U_SOL_E[0, :, j], color=curr_color, linewidth=1.0,
               label="$v_{r}(\phi, r$ = " + str(round(((new_r[j] * u.km).to(u.AU)).value, 2)) + " AU)")
    ax[0].plot(180 / np.pi * p, vr[:, 55, j], color=curr_color, linewidth=5.0, alpha=0.2)
    # rho
    ax[1].plot(180 / np.pi * p, ((U_SOL_E[1, :, j] / m_p.value) * (1 / u.km ** 3)).to(1 / u.cm ** 3), color=curr_color,
               linewidth=1.0, label="$n(\phi, r$ = " + str(round(((new_r[j] * u.km).to(u.AU)).value, 2)) + " AU)")
    ax[1].plot(180 / np.pi * p, ((rho[:, 55, j] / m_p.value) * (1 / u.km ** 3)).to(1 / u.cm ** 3),
               color=curr_color, linewidth=5.0, alpha=0.2)
    # pressure
    ax[2].plot(180 / np.pi * p, (U_SOL_E[2, :, j] * (u.kg / ((u.s ** 2) * u.km))).to(u.dyne / (u.cm ** 2)),
               color=curr_color,
               linewidth=1.0, label="$p(\phi, r$ = " + str(round(((new_r[j] * u.km).to(u.AU)).value, 2)) + ")")
    ax[2].plot(180 / np.pi * p, (Pr[:, 55, j]).to(u.dyne / (u.cm ** 2)),
               color=curr_color, linewidth=5.0, alpha=0.2)
    # vp
    ax[3].plot(180 / np.pi * p, U_SOL_E[3, :, j], color=curr_color, linewidth=1.0,
               label="$v_{\phi}(\phi, r$ = " + str(round(((new_r[j] * u.km).to(u.AU)).value, 2)) + ")")
    ax[3].plot(180 / np.pi * p, vp[:, 55, j], color=curr_color, linewidth=5.0, alpha=0.2)

fig.suptitle("MAS vs. HD")
ax[0].set_ylabel(r"$v_{r}$ (km/s)")
ax[1].set_ylabel(r"$n$ (1/cm$^3$)")
ax[2].set_ylabel(r"$p$ (dyne/cm$^2$)")
ax[3].set_ylabel(r"$v_{\phi}$ (km/s)")
ax[0].legend(loc=(1.05, .05), fontsize=13)
ax[1].legend(loc=(1.05, .05), fontsize=13)
ax[2].legend(loc=(1.05, .05), fontsize=13)
ax[3].legend(loc=(1.05, .05), fontsize=13)
ax[1].set_yscale("log")
ax[2].set_yscale("log")
ax[0].spines["right"].set_visible(False)
ax[0].spines["top"].set_visible(False)
_ = ax[0].tick_params(axis='both', which='major')
plt.tight_layout()
plt.show()
