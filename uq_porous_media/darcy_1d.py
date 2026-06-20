"""
1D single-phase Darcy flow (no-source, steady state).

PDE:   -d/dx[ k(x) dp/dx ] = 0,   x in [0, 1]
BC:    p(0) = p_left,   p(1) = p_right

Analytic solution (valid when no volumetric source term):
    p(x) = p_left + (p_right - p_left) * R(x) / R(1)
    R(x) = integral_0^x  1/k(s) ds      <- cumulative hydraulic resistance
    q     = (p_left - p_right) / R(1)   <- constant Darcy flux throughout

This is used instead of the general FD solver for MCMC speed.
For problems with sources or 2D, switch to the FD assembly in solve_darcy_fd().

Observation operator H:
    d = H p = [ p(x_s1), ..., p(x_sN) ]  (pressure at sensor locations)
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid


# ---------------------------------------------------------------------------
# Analytic 1D solver (fast, exact for no-source case)
# ---------------------------------------------------------------------------

def solve_darcy_1d(k_field, x, p_left=1.0, p_right=0.0):
    """
    Exact 1D Darcy solution via cumulative integration of 1/k.

    Parameters
    ----------
    k_field : (n,) permeability at grid points x  (strictly positive)
    x       : (n,) monotone grid in [0, 1]

    Returns
    -------
    p : (n,) pressure field
    q : float, constant Darcy flux
    """
    inv_k = 1.0 / (k_field + 1e-30)
    R = np.concatenate([[0.0], cumulative_trapezoid(inv_k, x)])
    R_total = max(R[-1], 1e-30)
    p = p_left + (p_right - p_left) * R / R_total
    q = (p_left - p_right) / R_total
    return p, q


# ---------------------------------------------------------------------------
# Forward model: xi -> observations
# ---------------------------------------------------------------------------

class DarcyForwardModel:
    """
    Forward model  G : KL coefficients xi  ->  pressure observations d.

    This is the operator that appears in the Gaussian likelihood:
        d_obs ~ N(G(xi), sigma2 * I)

    The difference  d_obs - G(xi)  is the "comparison to experiment."
    """

    def __init__(self, kl, sensor_x=None, p_left=1.0, p_right=0.0):
        self.kl = kl
        self.x = kl.x
        self.p_left = p_left
        self.p_right = p_right

        if sensor_x is None:
            sensor_x = np.linspace(0.1, 0.9, 7)
        self.sensor_x = np.asarray(sensor_x, dtype=float)
        self.n_obs = len(self.sensor_x)

    def __call__(self, xi):
        """xi (M,) -> predicted pressure at sensor locations (n_obs,)."""
        k = self.kl.xi_to_k(xi)
        p, _ = solve_darcy_1d(k, self.x, self.p_left, self.p_right)
        return np.interp(self.sensor_x, self.x, p)

    def full_solution(self, xi):
        """Return (p, q) over the full grid for a given xi."""
        k = self.kl.xi_to_k(xi)
        return solve_darcy_1d(k, self.x, self.p_left, self.p_right)
