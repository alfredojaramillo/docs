"""
Gaussian Random Field (GRF) for log-permeability via Karhunen-Loève expansion.

The infinite-dimensional field Y = log k is represented as

    Y(x) = mu_Y  +  sum_{i=1}^{M}  sqrt(lambda_i) * phi_i(x) * xi_i
           ^^^^       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           mean       KL expansion  (truncated at M modes)

where (lambda_i, phi_i) are eigenpairs of the discretised covariance matrix
    C_ij = sigma_Y^2 * K(x_i, x_j),
and xi = (xi_1, ..., xi_M) ~ N(0, I_M)  are the inferred KL coefficients.

Benefits
--------
  - k(x) = exp(Y(x)) > 0  by construction  (log-normal marginals)
  - Spatial correlation structure encoded in the kernel K
  - Dimension reduction: M << n_grid, typically M=20 captures > 99% variance
  - xi ~ N(0, I) enables dimension-robust pCN MCMC (prior is standard Gaussian)

Wiener-Askey PCE note
---------------------
If you use polynomial chaos expansion instead of MCMC, the distribution of the
input coefficients dictates the orthogonal polynomial basis:
    xi ~ Normal  ->  Hermite polynomials
    xi ~ Gamma   ->  Laguerre polynomials
    xi ~ Uniform ->  Legendre polynomials
    xi ~ Beta    ->  Jacobi polynomials
"""

import numpy as np


# ---------------------------------------------------------------------------
# Covariance kernels
# ---------------------------------------------------------------------------

def exp_kernel(x, sigma2=1.0, ell=0.3):
    """
    Exponential (Ornstein-Uhlenbeck) covariance.
    C(xi, xj) = sigma2 * exp(-|xi - xj| / ell)
    Produces continuous but non-differentiable sample paths.
    """
    xi, xj = np.meshgrid(x, x, indexing='ij')
    return sigma2 * np.exp(-np.abs(xi - xj) / ell)


def squared_exp_kernel(x, sigma2=1.0, ell=0.3):
    """
    Squared-exponential (RBF) covariance.
    C(xi, xj) = sigma2 * exp(-|xi - xj|^2 / (2 * ell^2))
    Produces infinitely differentiable (very smooth) sample paths.
    """
    xi, xj = np.meshgrid(x, x, indexing='ij')
    return sigma2 * np.exp(-0.5 * (xi - xj) ** 2 / ell ** 2)


# ---------------------------------------------------------------------------
# KL expansion
# ---------------------------------------------------------------------------

class KLExpansion:
    """
    Karhunen-Loève expansion for 1D log-permeability on a uniform grid.

    Parameters
    ----------
    x        : 1D array, grid points in [0, 1]
    mu_Y     : mean of log-permeability field (median k = exp(mu_Y))
    sigma2_Y : variance of log-permeability field
    ell      : correlation length (controls spatial variability)
    n_modes  : number of KL modes M to retain
    kernel   : 'exponential' or 'squared_exp'
    """

    def __init__(self, x, mu_Y=0.0, sigma2_Y=1.0, ell=0.3,
                 n_modes=20, kernel='exponential'):
        self.x = np.asarray(x, dtype=float)
        self.n = len(x)
        self.mu_Y = mu_Y
        self.sigma2_Y = sigma2_Y
        self.ell = ell
        self.n_modes = n_modes

        if kernel == 'exponential':
            C = exp_kernel(self.x, sigma2_Y, ell)
        else:
            C = squared_exp_kernel(self.x, sigma2_Y, ell)

        # Symmetric eigendecomposition — returns ascending order
        lam, phi = np.linalg.eigh(C)
        idx = np.argsort(lam)[::-1]   # sort descending
        lam, phi = lam[idx], phi[:, idx]

        self.eigenvalues = np.maximum(lam[:n_modes], 0.0)   # (M,)
        self.eigenvectors = phi[:, :n_modes]                  # (n, M)
        self.sqrt_lam = np.sqrt(self.eigenvalues)             # (M,)

        total_var = lam.sum()
        self.frac_var = float(self.eigenvalues.sum() / max(total_var, 1e-14))

    def xi_to_log_k(self, xi):
        """
        Map KL coefficients xi in R^M  ->  log-permeability on grid.
            Y(x) = mu_Y + Phi @ (sqrt_lam * xi)
        """
        return self.mu_Y + self.eigenvectors @ (self.sqrt_lam * xi)

    def xi_to_k(self, xi):
        """Map xi  ->  permeability field  k = exp(Y). Always positive."""
        return np.exp(self.xi_to_log_k(xi))

    def sample_xi(self, n=1, rng=None):
        """Draw n sets of KL coefficients from the prior N(0, I)."""
        rng = np.random.default_rng() if rng is None else rng
        return rng.standard_normal((n, self.n_modes))

    def sample_fields(self, n=1, rng=None):
        """Draw n log-k field realisations from the GRF prior. Returns (n, n_grid)."""
        xi = self.sample_xi(n, rng)
        return np.array([self.xi_to_log_k(xi[i]) for i in range(n)])
