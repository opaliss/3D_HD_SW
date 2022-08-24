"""Module contains functions to interpolate 2d initial condition at 30Rs.

Authors: Opal Issan
Version: August 21, 2022
"""
from scipy.interpolate import RegularGridInterpolator
import numpy as np


def interpolate_initial_condition(data, p_coord, t_coord, r_coord, p_interp, t_interp, r_interp):
    # coordinate grid
    pp, tt, rr = np.meshgrid(p_interp, t_interp, r_interp, indexing='ij')
    coordinate_grid = np.array([pp.T, tt.T, rr.T]).T
    # interpolate
    interp_function = RegularGridInterpolator(
        points=(p_coord, t_coord, r_coord),
        values=np.array(data),
        method="linear",
        bounds_error=False,
        fill_value=None)

    return interp_function(coordinate_grid)


def interpolate_ace_data(x, xp, fp, period):
    """return the interpolate values on new grid x of data (xp, fp)

    Parameters
    ----------
    x: array
        1d array of new grid coordinate
    xp: float
        1d array of old grid coordinate
    fp: bool
        1d array data on xp coordinate
    period: float
        periodicity

    Returns
    -------
    array
        fp on x coordinate
    """
    non_nan_idx = np.where(fp != np.nan)
    return np.interp(x=x, xp=xp[~np.isnan(fp)], fp=fp[~np.isnan(fp)], period=period)