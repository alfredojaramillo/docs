"""
Three conjugate prior pairs — exact Bayesian updates with no MCMC.

───────────────────────────────────────────────────────────────────────
DEFINITION
  A prior p(theta | h) is *conjugate* to the likelihood p(x | theta) if
  the posterior p(theta | x1,...,xn) is in the SAME parametric family
  as the prior. Only the hyperparameters h change; the family does not.

  Consequence: Bayesian updating reduces to a few arithmetic formulas.
  No numerical integration, no MCMC, no approximation.

───────────────────────────────────────────────────────────────────────
PSEUDO-DATA INTERPRETATION
  Every hyperparameter has a "virtual experiment" meaning:
    Normal-Normal  : tau0 encodes n0 = sigma^2/tau0^2 imaginary observations
    Normal-InvGamma: alpha0 encodes n0 = 2*alpha0 imaginary observations
                     beta0  encodes the imaginary sum-of-squares / 2

  Real data updates the same counters as imaginary data would.
  This is why the update formulas look like "running averages."

───────────────────────────────────────────────────────────────────────
THREE CASES IMPLEMENTED
  Case 1  NormalNormal        — infer mu, sigma known
  Case 2  NormalInvGamma      — infer sigma^2, mu known   ← Darcy Gibbs step
  Case 3  NormalNIG           — infer both mu AND sigma^2 jointly
"""

import numpy as np
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════
# Case 1: Normal likelihood + Gaussian prior on the mean
# ═══════════════════════════════════════════════════════════════════════

class NormalNormal:
    """
    Model:
        x_i | mu  ~iid  N(mu, sigma^2)        [sigma known, mu unknown]
        mu        ~     N(mu0, tau0^2)         [conjugate prior]

    Posterior:  mu | data  ~  N(mu_n, tau_n^2)

    Update formulas (PRECISIONS ADD):
        1/tau_n^2 = 1/tau0^2  +  n/sigma^2
        mu_n      = tau_n^2 * (mu0/tau0^2  +  n*x_bar/sigma^2)

    Key insight: posterior precision = prior precision + data precision.
    Information accumulates linearly in n — the posterior variance is
    a HARMONIC mean of prior and data variances, weighted by n.
    """

    def __init__(self, mu0: float, tau0: float, sigma: float):
        """
        mu0   : prior mean for mu
        tau0  : prior std for mu  (large = vague / weakly informative)
        sigma : FIXED data noise std (known; not inferred here)
        """
        self.mu0    = mu0
        self.tau0   = tau0
        self.sigma  = sigma
        # Posterior params initialised to prior
        self.mu_n   = mu0
        self.tau_n  = tau0
        self.n      = 0

    def update(self, data):
        """
        Incorporate new data.  Returns a NEW NormalNormal with updated params.
        Calling sequentially gives the same result as a single batch update.
        """
        data  = np.asarray(data).ravel()
        n_new = len(data)
        x_bar = data.mean()

        prec_prior = 1.0 / self.tau_n ** 2
        prec_data  = n_new / self.sigma ** 2
        prec_post  = prec_prior + prec_data

        tau_new = np.sqrt(1.0 / prec_post)
        mu_new  = (prec_prior * self.mu_n + prec_data * x_bar) / prec_post

        out = NormalNormal(self.mu0, self.tau0, self.sigma)
        out.mu_n = mu_new;  out.tau_n = tau_new;  out.n = self.n + n_new
        return out

    def pdf(self, mu_grid):
        return stats.norm.pdf(mu_grid, self.mu_n, self.tau_n)

    def sample(self, size=1, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        return rng.normal(self.mu_n, self.tau_n, size=size)

    def predictive_pdf(self, x_grid):
        """
        Posterior predictive p(x_new | data) = N(mu_n, tau_n^2 + sigma^2).
        Uncertainty in mu ADDS to observation noise in the predictive.
        """
        std_pred = np.sqrt(self.tau_n ** 2 + self.sigma ** 2)
        return stats.norm.pdf(x_grid, self.mu_n, std_pred)

    @property
    def mean(self):      return self.mu_n
    @property
    def std(self):       return self.tau_n
    @property
    def precision(self): return 1.0 / self.tau_n ** 2


# ═══════════════════════════════════════════════════════════════════════
# Case 2: Normal likelihood + InvGamma prior on the variance
# ═══════════════════════════════════════════════════════════════════════

class NormalInvGamma:
    """
    Model:
        x_i | sigma^2  ~iid  N(mu, sigma^2)   [mu known, sigma^2 unknown]
        sigma^2         ~    InvGamma(alpha0, beta0)   [conjugate prior]

    Posterior:  sigma^2 | data  ~  InvGamma(alpha_n, beta_n)

    Update formulas:
        alpha_n = alpha0  +  n/2
        beta_n  = beta0   +  sum((x_i - mu)^2) / 2

    Key insight: alpha counts "half-observations"; beta accumulates
    the sum-of-squared residuals.  Both updates are additive — you can
    process data one point at a time and get the same answer.

    ▶ THIS IS EXACTLY THE GIBBS STEP IN THE DARCY SAMPLER:
        xi fixed => residuals r = d_obs - G(xi)
        sigma^2 | xi, d  ~  InvGamma(alpha0 + n/2,  beta0 + ||r||^2/2)
        One sample from this InvGamma replaces one MH accept/reject step.
    """

    def __init__(self, alpha0: float, beta0: float, mu: float = 0.0):
        """
        alpha0 : InvGamma shape (> 1 for finite mean; > 2 for finite variance)
        beta0  : InvGamma scale  (same units as sigma^2)
        mu     : known mean of the Gaussian likelihood
        """
        self.alpha0   = alpha0
        self.beta0    = beta0
        self.mu       = mu
        self.alpha_n  = alpha0
        self.beta_n   = beta0
        self.n        = 0

    def update(self, data):
        data  = np.asarray(data).ravel()
        n_new = len(data)
        ss    = np.sum((data - self.mu) ** 2)

        out = NormalInvGamma(self.alpha0, self.beta0, self.mu)
        out.alpha_n = self.alpha_n + n_new / 2.0
        out.beta_n  = self.beta_n  + ss / 2.0
        out.n       = self.n + n_new
        return out

    def pdf(self, sigma2_grid):
        return stats.invgamma.pdf(sigma2_grid, a=self.alpha_n, scale=self.beta_n)

    def sample(self, size=1, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        return 1.0 / rng.gamma(self.alpha_n, scale=1.0 / self.beta_n, size=size)

    @property
    def mean(self):
        return self.beta_n / max(self.alpha_n - 1.0, 1e-12)

    @property
    def mode(self):
        return self.beta_n / (self.alpha_n + 1.0)


# ═══════════════════════════════════════════════════════════════════════
# Case 3: Normal likelihood + Normal-InvGamma joint prior (both unknown)
# ═══════════════════════════════════════════════════════════════════════

class NormalNIG:
    """
    Model:
        x_i | mu, sigma^2  ~iid  N(mu, sigma^2)
        (mu, sigma^2)       ~    NIG(mu0, kappa0, alpha0, beta0)

    The NIG joint prior factorises as:
        p(mu, sigma^2) = N(mu | mu0,  sigma^2/kappa0) * InvGamma(sigma^2 | alpha0, beta0)

    Posterior:  (mu, sigma^2) | data  ~  NIG(mu_n, kappa_n, alpha_n, beta_n)

    Update formulas:
        n_bar    = n                       (number of observations)
        x_bar    = mean(data)
        S        = sum((x_i - x_bar)^2)   (sum of squares around sample mean)

        kappa_n  = kappa0 + n
        mu_n     = (kappa0 * mu0 + n * x_bar) / kappa_n
        alpha_n  = alpha0 + n / 2
        beta_n   = beta0 + S/2 + kappa0*n*(x_bar - mu0)^2 / (2*kappa_n)

    Marginal posterior of mu (integrating out sigma^2):
        mu | data  ~  t_{2*alpha_n}  with mean mu_n and scale factor

    Pseudo-data interpretation:
        kappa0 : weight of prior on mu  (= # imaginary observations)
        2*alpha0: # imaginary observations for variance estimation
        mu0    : imaginary sample mean
        beta0  : half the imaginary sum-of-squares
    """

    def __init__(self, mu0: float, kappa0: float, alpha0: float, beta0: float):
        self.mu0 = mu0;  self.kappa0 = kappa0
        self.alpha0 = alpha0;  self.beta0 = beta0
        self.mu_n = mu0;  self.kappa_n = kappa0
        self.alpha_n = alpha0;  self.beta_n = beta0
        self.n = 0

    def update(self, data):
        data  = np.asarray(data).ravel()
        n_new = len(data)
        x_bar = data.mean()
        S     = np.sum((data - x_bar) ** 2)

        kn = self.kappa_n + n_new
        mn = (self.kappa_n * self.mu_n + n_new * x_bar) / kn
        an = self.alpha_n + n_new / 2.0
        bn = (self.beta_n + S / 2.0
              + self.kappa_n * n_new * (x_bar - self.mu_n) ** 2 / (2.0 * kn))

        out = NormalNIG(self.mu0, self.kappa0, self.alpha0, self.beta0)
        out.mu_n = mn;  out.kappa_n = kn
        out.alpha_n = an;  out.beta_n = bn
        out.n = self.n + n_new
        return out

    def marginal_mu_pdf(self, mu_grid):
        """Marginal p(mu | data) — Student-t after integrating out sigma^2."""
        df    = 2.0 * self.alpha_n
        scale = np.sqrt(self.beta_n * (self.kappa_n + 1.0)
                        / (self.alpha_n * self.kappa_n))
        return stats.t.pdf(mu_grid, df=df, loc=self.mu_n, scale=scale)

    def marginal_sigma2_pdf(self, sigma2_grid):
        """Marginal p(sigma^2 | data) = InvGamma(alpha_n, beta_n)."""
        return stats.invgamma.pdf(sigma2_grid, a=self.alpha_n, scale=self.beta_n)

    def joint_log_density(self, mu_2d, sigma2_2d):
        """
        Log joint density on a 2D grid (mu_2d, sigma2_2d), both shape (G, G).
        p(mu, sigma^2 | data) = N(mu | mu_n, sigma^2/kappa_n) * InvGamma(sigma^2 | alpha_n, beta_n)
        """
        log_norm = stats.norm.logpdf(
            mu_2d, loc=self.mu_n,
            scale=np.sqrt(sigma2_2d / self.kappa_n)
        )
        log_ig = stats.invgamma.logpdf(sigma2_2d, a=self.alpha_n, scale=self.beta_n)
        return log_norm + log_ig

    def sample(self, size=1, rng=None):
        """Draw (mu, sigma^2) jointly from the NIG posterior."""
        rng    = np.random.default_rng() if rng is None else rng
        sigma2 = 1.0 / rng.gamma(self.alpha_n, scale=1.0 / self.beta_n, size=size)
        mu     = rng.normal(self.mu_n, np.sqrt(sigma2 / self.kappa_n))
        return mu, sigma2

    @property
    def mean_mu(self):     return self.mu_n
    @property
    def mean_sigma2(self): return self.beta_n / max(self.alpha_n - 1.0, 1e-12)
