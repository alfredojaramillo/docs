"""
pCN (preconditioned Crank-Nicolson) + Gibbs MCMC.

pCN proposal for xi
-------------------
    xi_*  =  sqrt(1 - beta^2) * xi  +  beta * w,    w ~ N(0, I)

Key property: the N(0, I) prior is *invariant* under this map, so the
Metropolis acceptance ratio only involves the likelihood ratio:
    alpha = min(1,  p(d | xi_*) / p(d | xi))

This makes pCN *dimension-robust*: the same beta works regardless of the
number of KL modes M (unlike random-walk MH where beta must shrink as M grows).

Gibbs step for sigma2
---------------------
After each pCN sweep, draw sigma2 from its exact full conditional:
    sigma2 | xi, d  ~  InvGamma(alpha0 + n/2,  beta0 + ||r||^2/2)
where r = d_obs - G(xi). No accept/reject needed — conjugacy gives us
this closed-form draw for free.

Full sweep (one MCMC iteration):
    1. Propose xi_* via pCN; accept/reject on likelihood ratio.
    2. Draw sigma2 exactly from InvGamma full conditional.
"""

import numpy as np
from .inference import log_likelihood, gibbs_sigma2


def run_mcmc(G, d_obs, kl,
             n_iter=8000, n_burnin=2000,
             beta=0.15,
             alpha0=2.0, beta0=1e-3,
             sigma2_init=None, xi_init=None,
             thin=2, rng=None, verbose=True):
    """
    Run pCN + Gibbs for (xi, sigma2).

    Parameters
    ----------
    G         : forward model, callable  xi (M,) -> d_pred (n_obs,)
    d_obs     : observed / experimental data  (n_obs,)
    kl        : KLExpansion object (provides n_modes)
    n_iter    : post-burnin iterations to run
    n_burnin  : burn-in iterations (discarded)
    beta      : pCN step size in (0, 1); ~0.1-0.2 typically gives 20-40% acceptance
    alpha0, beta0 : InvGamma hyperprior parameters for sigma2
    thin      : store every `thin`-th sample
    rng       : numpy Generator (for reproducibility)

    Returns
    -------
    dict:
        'xi'         : (n_keep, M)   posterior KL coefficient samples
        'sigma2'     : (n_keep,)     posterior noise variance samples
        'log_like'   : (n_keep,)     log-likelihood trace
        'accept_rate': float         pCN acceptance rate (post-burnin)
    """
    rng = np.random.default_rng(42) if rng is None else rng
    M = kl.n_modes

    xi = rng.standard_normal(M) * 0.01 if xi_init is None else xi_init.copy()
    sigma2 = 1e-3 if sigma2_init is None else float(sigma2_init)

    # Cache current predicted data to avoid redundant forward call each step
    d_pred_cur = G(xi)
    ll_cur = log_likelihood(d_obs, d_pred_cur, sigma2)

    n_total = n_burnin + n_iter
    n_keep = n_iter // thin
    xi_samples = np.zeros((n_keep, M))
    s2_samples = np.zeros(n_keep)
    ll_samples = np.zeros(n_keep)
    n_accept = 0

    for t in range(n_total):
        # --- pCN proposal for xi (prior-reversible) ---
        w = rng.standard_normal(M)
        xi_prop = np.sqrt(1.0 - beta ** 2) * xi + beta * w
        d_pred_prop = G(xi_prop)
        ll_prop = log_likelihood(d_obs, d_pred_prop, sigma2)

        if np.log(rng.uniform()) < ll_prop - ll_cur:
            xi = xi_prop
            d_pred_cur = d_pred_prop
            ll_cur = ll_prop
            if t >= n_burnin:
                n_accept += 1

        # --- Gibbs update for sigma2 (exact InvGamma conjugate draw) ---
        residuals = d_obs - d_pred_cur
        sigma2 = gibbs_sigma2(residuals, alpha0, beta0, rng)
        ll_cur = log_likelihood(d_obs, d_pred_cur, sigma2)

        # --- Store post-burnin samples (with thinning) ---
        if t >= n_burnin and (t - n_burnin) % thin == 0:
            i = (t - n_burnin) // thin
            xi_samples[i] = xi
            s2_samples[i] = sigma2
            ll_samples[i] = ll_cur

        if verbose and (t + 1) % 2000 == 0:
            rate = n_accept / max(t - n_burnin + 1, 1)
            print(f"  step {t+1:5d}/{n_total} | "
                  f"accept={rate:.3f} | sigma2={sigma2:.2e} | ll={ll_cur:.2f}")

    return {
        'xi': xi_samples,
        'sigma2': s2_samples,
        'log_like': ll_samples,
        'accept_rate': n_accept / n_iter,
    }
