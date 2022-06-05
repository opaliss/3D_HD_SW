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