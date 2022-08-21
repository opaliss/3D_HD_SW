from tools.MASweb import get_mas_path
from psipy.model import MASOutput
from finite_difference_functions.fd_2d import maccormack_pizzo_2d, euler_pizzo_2d
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
new_r = np.linspace(r[10], r[-1], int(400))
# change in r
dr = (new_r[1] - new_r[0]).value

# since the last phi index is less than 2*pi, then we will append 2*pi to phi scale.
p = np.append(p, 2 * np.pi)
vr = model["vr"].data * (u.km / u.s)
vr = np.append(vr, [vr[0, :, :]], axis=0)

vp = model["vp"].data * (u.km / u.s)
vp = np.append(vp, [vp[0, :, :]], axis=0)

rho = np.array(model["rho"].data) * m_p  # multiply by kg
rho = (rho * (1 / u.cm ** 3)).to(u.kg / u.km ** 3).value  # convert to mks (km)
rho = np.append(rho, [rho[0, :, :]], axis=0)

Pr = np.array(model["p"].data)
Pr = ((Pr * (u.dyne / u.cm ** 2)).to(u.kg / (u.s ** 2 * u.km)))  # convert to mks (km)
Pr = np.append(Pr, [Pr[0, :, :]], axis=0)

U_SOL = np.zeros((4, len(p), len(new_r)))
U_SOL[:, :, 0] = np.array((vr[:, 55, 0],
                           rho[:, 55, 0],
                           Pr[:, 55, 0],
                           vp[:, 55, 0]))

for ii in range(len(new_r) - 1):
    U_SOL[:, :, ii + 1] = euler_pizzo_2d(U=U_SOL[:, :, ii], dr=dr, dp=dp, r=new_r[ii], theta=0)
    if ii % 25 == 0:
        print(ii)
        print((new_r[ii]).to(u.AU))
        fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(5, 10))
        pos = ax[0].plot(180 / np.pi * p, U_SOL[0, :, ii + 1])
        ax[0].set_ylabel(r'$\frac{km}{s}$')
        ax[0].set_title(r"$v_{r}$")

        pos = ax[1].plot(180 / np.pi * p,
                         ((U_SOL[1, :, ii + 1] / m_p.value) * (1 / u.km ** 3)).to(1 / u.cm ** 3))
        ax[1].set_ylabel(r'$\frac{1}{cm^3}$')
        ax[1].set_title(r"$n_{p}$")

        pos = ax[2].plot(180 / np.pi * p,
                         (U_SOL[2, :, ii + 1] * (u.kg / (u.s ** 2 * u.km))).to(u.dyne / (u.cm ** 2)))
        ax[2].set_ylabel(r'$\frac{dyne}{cm^2}$')
        ax[2].set_title(r"$P$")

        pos = ax[3].plot(180 / np.pi * p, U_SOL[3, :, ii + 1])
        ax[3].set_ylabel(r'km/s')
        ax[3].set_title(r"$v_{\phi}$")
        ax[3].set_xticks([0, 90, 180, 270, 360])
        ax[3].set_xlabel(" Carrington Longitude (Deg.)")
        fig.suptitle("r = " + str(round(new_r[ii + 1].to(u.AU).value, 3)))
        plt.tight_layout()

        plt.show()
        plt.close()
