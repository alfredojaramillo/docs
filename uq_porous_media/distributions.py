"""
Probability distributions used in Bayesian UQ for porous media.

UQ role of each distribution
------------------------------
Normal / Gaussian
  1. Gaussian likelihood  p(d | xi, sigma2) = N(G(xi), sigma2 I)
       -> models measurement error; "comparison to experiment" term
  2. Prior on log-k via Gaussian Random Field  Y(x) ~ GP(mu_Y, C)
       -> ensures k = exp(Y) > 0 and spatially correlated
  3. KL coefficients  xi ~ N(0, I)
       -> dimension reduction: infinite field -> finite R^M

Gamma
  Conjugate prior on precision  tau = 1/sigma^2
  tau ~ Gamma(alpha, beta)  with rate parametrisation (beta = 1/scale)
  E[tau] = alpha/beta

InvGamma
  Conjugate prior on variance  sigma^2
  sigma^2 ~ InvGamma(alpha, beta)  with scale parametrisation
  E[sigma^2] = beta / (alpha - 1)  for alpha > 1
  Posterior update (closed form, no MCMC needed):
      sigma^2 | xi, d ~ InvGamma(alpha + n/2,  beta + ||d - G(xi)||^2 / 2)

LogNormal
  Direct prior on k > 0:  k = exp(Y),  Y ~ Normal
"""

import numpy as np
from scipy import stats as spstats


# ---------------------------------------------------------------------------
# Normal / Gaussian
# ---------------------------------------------------------------------------

def normal_pdf(x, mu=0.0, sigma=1.0):
    return spstats.norm.pdf(x, loc=mu, scale=sigma)


def normal_logpdf(x, mu=0.0, sigma=1.0):
    return spstats.norm.logpdf(x, loc=mu, scale=sigma)


# ---------------------------------------------------------------------------
# Gamma  (conjugate prior on precision  tau = 1/sigma^2)
# ---------------------------------------------------------------------------

def gamma_pdf(x, alpha, beta):
    """Gamma(alpha, beta) with *rate* beta (scale = 1/beta)."""
    return spstats.gamma.pdf(x, a=alpha, scale=1.0 / beta)


def gamma_logpdf(x, alpha, beta):
    return spstats.gamma.logpdf(x, a=alpha, scale=1.0 / beta)


def gamma_sample(alpha, beta, size=1, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    return rng.gamma(shape=alpha, scale=1.0 / beta, size=size)


# ---------------------------------------------------------------------------
# InvGamma  (conjugate prior on variance  sigma^2)
# ---------------------------------------------------------------------------

def inv_gamma_pdf(x, alpha, beta):
    """InvGamma(alpha, beta) with *scale* beta. E[x] = beta/(alpha-1)."""
    return spstats.invgamma.pdf(x, a=alpha, scale=beta)


def inv_gamma_logpdf(x, alpha, beta):
    return spstats.invgamma.logpdf(x, a=alpha, scale=beta)


def inv_gamma_sample(alpha, beta, size=1, rng=None):
    """Draw from InvGamma(alpha, beta).  Equivalent to 1/Gamma(alpha, 1/beta)."""
    rng = np.random.default_rng() if rng is None else rng
    return 1.0 / rng.gamma(shape=alpha, scale=1.0 / beta, size=size)


def inv_gamma_posterior(alpha_prior, beta_prior, residuals):
    """
    Closed-form InvGamma posterior given Gaussian residuals r = d - G(xi).

    Returns (alpha_post, beta_post) for
        sigma^2 | xi, d ~ InvGamma(alpha_post, beta_post)
    """
    n = len(residuals)
    alpha_post = alpha_prior + n / 2.0
    beta_post = beta_prior + 0.5 * float(np.dot(residuals, residuals))
    return alpha_post, beta_post


# ---------------------------------------------------------------------------
# LogNormal  (direct positive prior on permeability k)
# ---------------------------------------------------------------------------

def lognormal_pdf(x, mu_log=0.0, sigma_log=1.0):
    return spstats.lognorm.pdf(x, s=sigma_log, scale=np.exp(mu_log))


def lognormal_logpdf(x, mu_log=0.0, sigma_log=1.0):
    return spstats.lognorm.logpdf(x, s=sigma_log, scale=np.exp(mu_log))
