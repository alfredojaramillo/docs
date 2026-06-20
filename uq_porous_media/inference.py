"""
Bayesian formulation for permeability inversion.

Posterior (Bayes' rule):
    p(xi, sigma2 | d)  ∝  p(d | xi, sigma2)  *  p(xi)  *  p(sigma2)
                           ^^^^^^^^^^^^^^^^^^     ^^^^^^^^     ^^^^^^^^^^^
                           Gaussian likelihood    N(0, I)     InvGamma prior
                           "comparison to exp"   (GRF prior   (noise variance)
                                                  via KL)

Likelihood
    d | xi, sigma2  ~  N(G(xi), sigma2 * I)
    log p = -n/2 * log(2*pi*sigma2) - ||d - G(xi)||^2 / (2*sigma2)

Prior on xi
    xi ~ N(0, I)    =>   log p(xi) = -||xi||^2 / 2   (+const)
    (The KL expansion absorbs the GRF covariance structure here.)

Prior on sigma2  (hierarchical, unknown noise level)
    sigma2 ~ InvGamma(alpha0, beta0)
    Conjugate to the Gaussian likelihood -> exact Gibbs update:
        sigma2 | xi, d ~ InvGamma(alpha0 + n/2,  beta0 + ||r||^2/2)
    where r = d - G(xi) are the residuals.
"""

import numpy as np
from .distributions import inv_gamma_logpdf, inv_gamma_posterior, inv_gamma_sample


def log_likelihood(d_obs, d_pred, sigma2):
    """log N(d_obs; d_pred, sigma2 * I)."""
    n = len(d_obs)
    r = d_obs - d_pred
    return -0.5 * n * np.log(2.0 * np.pi * sigma2) - 0.5 * np.dot(r, r) / sigma2


def log_prior_xi(xi):
    """log N(xi; 0, I) — prior on KL coefficients (up to constant)."""
    return -0.5 * np.dot(xi, xi)


def log_prior_sigma2(sigma2, alpha0, beta0):
    """log InvGamma(sigma2; alpha0, beta0) — hyperprior on noise variance."""
    return inv_gamma_logpdf(sigma2, alpha0, beta0)


def log_posterior(xi, sigma2, d_obs, G, alpha0, beta0):
    """Full unnormalised log-posterior log p(xi, sigma2 | d)."""
    return (log_likelihood(d_obs, G(xi), sigma2)
            + log_prior_xi(xi)
            + log_prior_sigma2(sigma2, alpha0, beta0))


def gibbs_sigma2(residuals, alpha0, beta0, rng=None):
    """
    Exact conjugate Gibbs sample from  p(sigma2 | xi, d).

    Given residuals r = d_obs - G(xi), the full conditional is
        sigma2 | r ~ InvGamma(alpha0 + n/2,  beta0 + ||r||^2/2)
    This is a single closed-form draw — no accept/reject needed.
    """
    rng = np.random.default_rng() if rng is None else rng
    alpha_post, beta_post = inv_gamma_posterior(alpha0, beta0, residuals)
    return float(inv_gamma_sample(alpha_post, beta_post, size=1, rng=rng).flat[0])
