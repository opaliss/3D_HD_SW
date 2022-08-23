import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import scipy
from scipy import ndimage
from astropy.constants import m_p, k_B
import heliopy.spice as spice
import astropy.units as u
from sunpy.coordinates.sun import carrington_rotation_time
from heliopy.data import omni
from finite_difference_functions.fd_2d_euler import backward_euler_pizzo_2d
import matplotlib
plt.rcParams['savefig.facecolor'] = 'white'
font = {'family': 'serif',
        'size': 15}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)

# time period
print(carrington_rotation_time(2240))
print(carrington_rotation_time(2241))


starttime = dt.datetime(year=2021, month=1, day=22)
endtime = dt.datetime(year=2021, month=2, day=18)
deltatime = dt.timedelta(hours=1)

times = np.arange(starttime, endtime, deltatime)

# Earth's Trajectory
earth_traj = spice.Trajectory('Earth')
earth_traj.generate_positions(times=times, observing_body='Sun', frame='IAU_SUN')
earth_coords = earth_traj.coords

# get 1hr cadence
omni_data = omni.h0_mrg1hr(starttime, endtime)
p_interp = np.linspace(0, 360, 100)

#interpolate data
V_interp = np.interp(p_interp, earth_coords.lon[:-1].value, omni_data.quantity('V'), period=360)
P_interp = np.interp(p_interp, earth_coords.lon[:-1].value, omni_data.quantity('Pressure'), period=360)
N_interp = np.interp(p_interp, earth_coords.lon[:-1].value, omni_data.quantity('N'), period=360)
T_interp = np.interp(p_interp, earth_coords.lon[:-1].value, omni_data.quantity('T'), period=360)

#convolve data (smoothing)
kernel_size = 3
kernel = np.ones(kernel_size) / kernel_size
V_convolved = scipy.ndimage.convolve(V_interp, kernel, mode='wrap')
P_convolved = scipy.ndimage.convolve(P_interp, kernel, mode='wrap')
N_convolved = scipy.ndimage.convolve(N_interp, kernel, mode='wrap')
T_convolved = scipy.ndimage.convolve(T_interp, kernel, mode='wrap')


fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(10, 18))


ax[0].scatter(earth_coords.lon[:-1], omni_data.quantity('V'), s=10, color="g", label="data")
ax[0].plot(p_interp, V_interp, color="g", label="linear-interp")
ax[0].plot(p_interp, V_convolved, color="k", ls="--", label="smoothed")
ax[0].legend()

ax[1].scatter(earth_coords.lon[:-1], omni_data.quantity('Pressure'),s=2, color="b")
ax[1].plot(p_interp, P_interp, color="b")
ax[1].plot(p_interp, P_convolved, color="k", ls="--")

ax[2].scatter(earth_coords.lon[:-1], omni_data.quantity('N'),s=2, color="r")
ax[2].plot(p_interp, N_interp, color="r")
ax[2].plot(p_interp, N_convolved, color="k", ls="--")


ax[3].scatter(earth_coords.lon[:-1], omni_data.quantity('T'),s=2, color="orange")
ax[3].plot(p_interp, T_interp, color="orange")
ax[3].plot(p_interp, T_convolved, color="k", ls="--")

ax[0].set_ylabel(r'bulk flow speed (km/s)')
ax[1].set_ylabel(r'pressure (nPa)')
ax[2].set_ylabel(r"proton density (1/cm$^3$)")
ax[3].set_ylabel(r"proton temperature (K)")

ax[2].set_xlabel("Carrington Longitude (Deg.)")
ax[2].set_xticks([0, 90, 180, 270, 360])
ax[2].set_xlim(0, 360)
ax[0].set_title("OMNI in-situ observations")
fig.autofmt_xdate()
plt.show()

# convert units
N_convolved_ = (N_convolved * m_p/u.cm**3).to(u.kg/(u.km**3))
#P_convolved_ = (P_convolved * u.nPa).to(u.kg/((u.s**2) * u.km))
P_convolved_ = ((2 * N_convolved_ * k_B * T_convolved * u.K) / m_p).to(u.kg/((u.s**2) * u.km))
#P_convolved_ = scipy.ndimage.convolve(P_convolved_, kernel, mode='wrap')

r_0 = np.mean(earth_coords.radius[:-1].to(u.km)).value
r_f = ((50*u.solRad).to(u.km)).value

r_vec = np.linspace(r_0, r_f, int(1e4))
dr = r_vec[1] - r_vec[0]

theta_avg = np.mean(earth_coords.lat).to(u.rad).value + np.pi/2


U_SOL = np.zeros((4, len(p_interp), len(r_vec)))
U_SOL[:, :, 0] = np.array([
                    V_convolved,
                    N_convolved_,
                    1e-1 * P_convolved_,
                    0 * np.ones(len(p_interp))
])

for ii in range(len(r_vec) - 1):
    print((r_vec[ii+1]*u.km).to(u.AU).value)

    U_SOL[:, :, ii + 1] = backward_euler_pizzo_2d(U=U_SOL[:, :, ii],
                                                  dr=np.abs(dr),
                                                  dp=(p_interp[1] - p_interp[0])*(np.pi/180),
                                                  r=r_vec[ii],
                                                  theta=theta_avg)
   # U_SOL[-1, :, ii+1] = 0 * p_interp

    # if (r_vec[ii]*u.km).to(u.AU).value < 0.178:
    #     print("debug")
    #
    # if ii % 15 == 0:
    #     fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(5, 10))
    #     ax[0].plot(p_interp, U_SOL[0, :, ii], c="r")
    #     ax[0].set_ylabel(r'$\frac{km}{s}$')
    #     ax[0].set_title(r"$v_{r}$")
    #
    #     ax[1].plot(p_interp, ((U_SOL[1, :, ii] / m_p.value) * (1 / u.km ** 3)).to(1 / u.cm ** 3), c="r")
    #     ax[1].set_ylabel(r'$\frac{1}{cm^3}$')
    #     ax[1].set_title(r"$n_{p}$")
    #
    #     pos = ax[2].plot(p_interp, (U_SOL[2, :, ii] * (u.kg / (u.s ** 2 * u.km))).to(u.dyne / (u.cm ** 2)), c="b")
    #     ax[2].set_ylabel(r'$\frac{dyne}{cm^2}$')
    #     ax[2].set_title(r"$P$")
    #
    #     pos = ax[3].plot(p_interp, U_SOL[3, :, ii], c="k")
    #     ax[3].set_ylabel(r'km/s')
    #     ax[3].set_title(r"$v_{\phi}$")
    #     ax[3].set_xticks([0, 90, 180, 270, 360])
    #     ax[3].set_xlabel(" Carrington Longitude (Deg.)")
    #     fig.suptitle("r = " + str(round((r_vec[ii]*(u.km)).to(u.AU).value, 3)) + " AU")
    #     plt.tight_layout()
    #     plt.show()

sample_columns = np.arange(0, len(r_vec), int(len(r_vec) // 5))
sample_columns = np.append(sample_columns, len(r_vec) - 1)
fig, ax = plt.subplots(4, 1, figsize=(10, 15), sharex=True)

color = iter(plt.cm.viridis_r(np.linspace(0, 1, len(sample_columns))))

for j in sample_columns:
    curr_color = next(color)
    ax[0].plot(p_interp, U_SOL[0, :, j], color=curr_color, linewidth=1.0,
               label="$v_{r}(\phi, r$ = " + str(round(((r_vec[j] * u.km).to(u.AU)).value, 2)) + " AU)")
    ax[1].plot(p_interp, ((U_SOL[1, :, j] / m_p.value) * (1 / u.km ** 3)).to(1 / u.cm ** 3), color=curr_color,
               linewidth=1.0, label="$n(\phi, r$ = " + str(round(((r_vec[j] * u.km).to(u.AU)).value, 2)) + " AU)")
    ax[2].plot(p_interp, (U_SOL[2, :, j] * (u.kg / ((u.s ** 2) * u.km))).to(u.dyne / (u.cm ** 2)), color=curr_color,
               linewidth=1.0, label="$p(\phi, r$ = " + str(round(((r_vec[j] * u.km).to(u.AU)).value, 2)) + ")")
    ax[3].plot(p_interp, U_SOL[3, :, j], color=curr_color, linewidth=1.0,
               label="$v_{\phi}(\phi, r$ = " + str(round(((r_vec[j] * u.km).to(u.AU)).value, 2)) + ")")

ax[0].set_ylabel(r"$v_{r}$ (km/s)")
ax[1].set_ylabel(r"$n$ (1/cm$^3$)")
ax[2].set_ylabel(r"$p$ (dyne/cm$^2$)")
ax[3].set_ylabel(r"$v_{\phi}$ (km/s)")
ax[0].legend(loc=(1.05, .05), fontsize=13)
ax[1].legend(loc=(1.05, .05), fontsize=13)
ax[2].legend(loc=(1.05, .05), fontsize=13)
ax[3].legend(loc=(1.05, .05), fontsize=13)

ax[0].spines["right"].set_visible(False)
ax[0].spines["top"].set_visible(False)
ax[2].set_yscale("log")
_ = ax[0].tick_params(axis='both', which='major')
plt.tight_layout()
plt.show()