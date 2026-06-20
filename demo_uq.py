"""
Bayesian UQ for 1D single-phase Darcy permeability inversion.

Demonstrates every concept from the "textbook PDFs -> UQ" chain:

  Normal     -> (1) Gaussian likelihood: measurement noise model
                (2) GRF prior on log k(x) via covariance kernel
                (3) KL coefficients xi ~ N(0,I) for dimension reduction
  InvGamma   -> conjugate hyperprior on unknown noise variance sigma^2
  Gamma      -> equivalent conjugate prior on precision tau = 1/sigma^2
  pCN MCMC   -> dimension-robust posterior sampler for the GRF coefficients
  Gibbs      -> exact (no MCMC) conjugate update for sigma^2

Figures saved to ./figures/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from uq_porous_media import (
    KLExpansion, DarcyForwardModel, solve_darcy_1d,
    normal_pdf, gamma_pdf, inv_gamma_pdf, inv_gamma_posterior,
    run_mcmc,
)

# ============================================================
# CONFIG
# ============================================================
SEED       = 42
N_NODES    = 65       # FD / KL grid points
N_MODES    = 20       # KL modes retained
SIGMA2_Y   = 1.0      # log-k field variance
ELL        = 0.3      # correlation length
MU_Y       = 0.0      # mean log-k  (so median k = 1)
SIGMA_EPS  = 0.02     # true noise std (2% of unit pressure drop)
N_SENSORS  = 7
N_ITER     = 8000
N_BURNIN   = 2000
BETA       = 0.20     # pCN step size
ALPHA0     = 2.0      # InvGamma hyperprior shape
BETA0      = 1e-3     # InvGamma hyperprior scale  (prior mean = 1e-3)

OUT = Path("figures")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

# ============================================================
# 1.  SETUP: grid, KL expansion, forward model
# ============================================================
print("=" * 62)
print("  Bayesian permeability UQ — 1D single-phase Darcy")
print("=" * 62)

x = np.linspace(0.0, 1.0, N_NODES)
kl = KLExpansion(x, mu_Y=MU_Y, sigma2_Y=SIGMA2_Y, ell=ELL,
                 n_modes=N_MODES, kernel='exponential')
print(f"\n  KL expansion: {N_MODES} modes capture "
      f"{kl.frac_var * 100:.1f}% of field variance")

sensor_x = np.linspace(0.1, 0.9, N_SENSORS)
G = DarcyForwardModel(kl, sensor_x=sensor_x)

# ============================================================
# 2.  GENERATE SYNTHETIC "EXPERIMENTAL" DATA
# ============================================================
xi_true    = rng.standard_normal(N_MODES)
log_k_true = kl.xi_to_log_k(xi_true)
k_true     = np.exp(log_k_true)
d_noiseless = G(xi_true)
d_obs       = d_noiseless + rng.normal(0.0, SIGMA_EPS, size=N_SENSORS)
SIGMA2_TRUE = SIGMA_EPS ** 2

print(f"  True noise sigma_eps = {SIGMA_EPS:.4f}")
print(f"  True noise variance  = {SIGMA2_TRUE:.2e}")
print(f"  Noisy observations   = {np.array2string(d_obs, precision=4)}\n")

# ============================================================
# 3.  RUN pCN + GIBBS MCMC
# ============================================================
print("Running pCN + Gibbs MCMC...")
results  = run_mcmc(G, d_obs, kl,
                    n_iter=N_ITER, n_burnin=N_BURNIN,
                    beta=BETA, alpha0=ALPHA0, beta0=BETA0,
                    thin=2, rng=rng, verbose=True)
xi_post  = results['xi']      # (n_keep, M)
s2_post  = results['sigma2']  # (n_keep,)
ll_trace = results['log_like']
n_keep   = len(xi_post)

print(f"\n  Acceptance rate:   {results['accept_rate']:.3f}")
print(f"  Posterior sigma (noise): mean={np.sqrt(s2_post.mean()):.4f}  "
      f"true={SIGMA_EPS:.4f}")

# Posterior field statistics
log_k_post = np.array([kl.xi_to_log_k(xi_post[i]) for i in range(n_keep)])
log_k_mean = log_k_post.mean(axis=0)
log_k_std  = log_k_post.std(axis=0)

d_post_pred = np.array([G(xi_post[i]) for i in range(n_keep)])

# ============================================================
# FIGURE 1 — Probability distributions and their UQ roles
# ============================================================
print("\nSaving figures...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Textbook PDFs and Their Role in Bayesian Permeability UQ",
             fontsize=13, fontweight='bold')

# -- Normal: measurement noise (likelihood)
ax = axes[0]
xv = np.linspace(-0.15, 0.15, 400)
for sig, col in [(0.02, 'C0'), (0.05, 'C1'), (0.10, 'C2')]:
    ax.plot(xv, normal_pdf(xv, 0, sig), color=col,
            label=rf'$\sigma_\varepsilon={sig}$')
ax.axvline(0, color='k', lw=0.7, ls='--')
ax.set_title('Normal — Gaussian likelihood\n'
             r'$d_{obs} - G(\xi) \sim \mathcal{N}(0,\sigma_\varepsilon^2)$'
             '\nModels measurement noise; drives "fit to experiment"',
             fontsize=9)
ax.set_xlabel(r'residual  $d_{obs} - G(\xi)$')
ax.set_ylabel('pdf')
ax.legend(fontsize=8)

# -- InvGamma: hyperprior on sigma^2
ax = axes[1]
xv = np.linspace(1e-7, 8e-3, 600)
for a, b, col in [(2, 1e-3, 'C0'), (3, 2e-3, 'C1'), (5, 4e-3, 'C2')]:
    ax.plot(xv * 1e3, inv_gamma_pdf(xv, a, b) * 1e-3, color=col,
            label=rf'$\alpha={a},\beta={b:.0e}$')
ax.axvline(SIGMA2_TRUE * 1e3, color='k', ls='--', lw=1.5,
           label=rf'true $\sigma^2$={SIGMA2_TRUE:.1e}')
ax.set_title(r'InvGamma — hyperprior on $\sigma_\varepsilon^2$'
             '\n' r'$\sigma^2 \sim \mathrm{InvGamma}(\alpha_0,\beta_0)$'
             '\nConjugate: posterior is exact InvGamma',
             fontsize=9)
ax.set_xlabel(r'$\sigma_\varepsilon^2 \times 10^{-3}$')
ax.set_ylabel('pdf (scaled)')
ax.legend(fontsize=8)

# -- Gamma: conjugate prior on precision tau = 1/sigma^2
ax = axes[2]
xv_tau = np.linspace(1, 4000, 600)
for a, b, col in [(2, 1e-3, 'C0'), (3, 2e-3, 'C1'), (5, 4e-3, 'C2')]:
    ax.plot(xv_tau, gamma_pdf(xv_tau, a, b), color=col,
            label=rf'$\alpha={a},\beta={b:.0e}$')
ax.axvline(1.0 / SIGMA2_TRUE, color='k', ls='--', lw=1.5,
           label=rf'true $\tau=1/\sigma^2$={1/SIGMA2_TRUE:.0f}')
ax.set_title(r'Gamma — conjugate prior on precision $\tau=1/\sigma^2$'
             '\n' r'$\tau \sim \mathrm{Gamma}(\alpha_0,\beta_0)$'
             '\nDual of InvGamma (same hyperparameters)',
             fontsize=9)
ax.set_xlabel(r'precision $\tau = 1/\sigma_\varepsilon^2$')
ax.set_ylabel('pdf')
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUT / "fig1_distributions.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  fig1_distributions.png")

# ============================================================
# FIGURE 2 — KL expansion: eigenvalue decay + eigenfunctions + prior samples
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Gaussian Random Field Prior via Karhunen-Loève Expansion", fontsize=13)

# Eigenvalue decay
ax = axes[0]
ax.semilogy(np.arange(1, N_MODES + 1), kl.eigenvalues, 'C0.-', ms=6)
cumvar = np.cumsum(kl.eigenvalues) / kl.eigenvalues.sum()
ax2 = ax.twinx()
ax2.plot(np.arange(1, N_MODES + 1), cumvar * 100, 'C1--', lw=1.5,
         label='cumulative %')
ax2.set_ylabel('Cumulative variance (%)', color='C1')
ax2.tick_params(axis='y', labelcolor='C1')
ax.set_xlabel('KL mode index  $i$')
ax.set_ylabel(r'Eigenvalue $\lambda_i$  (log scale)', color='C0')
ax.set_title(f'Eigenvalue decay\n'
             f'{N_MODES} modes → {kl.frac_var * 100:.1f}% variance captured')

# First 5 eigenfunctions
ax = axes[1]
for i in range(5):
    ax.plot(x, kl.eigenvectors[:, i], label=rf'$\phi_{i+1}$')
ax.set_xlabel('x')
ax.set_title('KL eigenfunctions  $\\phi_i(x)$\n'
             r'(ordered by $\lambda_i$; each is $\xi_i$\'s spatial fingerprint)')
ax.legend(fontsize=8, ncol=2)
ax.axhline(0, color='k', lw=0.5)

# Prior log-k field samples
ax = axes[2]
prior_samples = kl.sample_fields(n=15, rng=rng)
for s in prior_samples:
    ax.plot(x, s, color='steelblue', alpha=0.35, lw=0.8)
ax.plot(x, prior_samples.mean(axis=0), 'k--', lw=1.5, label='sample mean')
ax.axhline(MU_Y, color='red', ls=':', lw=1.5,
           label=rf'prior mean $\mu_Y={MU_Y}$')
ax.set_xlabel('x')
ax.set_ylabel('log k(x)')
ax.set_title('15 prior log-k realisations\n'
             r'$\log k \sim \mathcal{GP}(\mu_Y, C_{\exp})$  (before any data)')
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUT / "fig2_kl_prior.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  fig2_kl_prior.png")

# ============================================================
# FIGURE 3 — Problem setup: truth + noisy "experimental" data
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle("Synthetic Experiment Setup  (the forward problem)", fontsize=13)

p_true, q_true = solve_darcy_1d(k_true, x)

ax = axes[0]
ax.plot(x, log_k_true, 'k-', lw=2.5, label='true log k(x)')
ax.fill_between(x,
                MU_Y - 2 * np.sqrt(SIGMA2_Y),
                MU_Y + 2 * np.sqrt(SIGMA2_Y),
                alpha=0.12, color='gray',
                label=r'prior $\pm 2\sigma_Y$')
ax.set_xlabel('x')
ax.set_ylabel('log k(x)')
ax.set_title('True log-permeability (one KL realisation)\n'
             r'$\log k(x) = \mu_Y + \sum_i\sqrt{\lambda_i}\phi_i(x)\xi_i^{true}$')
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(x, p_true, 'k-', lw=2.5, label='true pressure p(x)')
ax.scatter(sensor_x, d_noiseless, color='black', s=50, marker='+', zorder=6,
           label='noiseless sensor values')
ax.scatter(sensor_x, d_obs, color='red', s=70, zorder=7,
           label=rf'noisy observations ($\sigma_\varepsilon$={SIGMA_EPS})')
for xs, do in zip(sensor_x, d_obs):
    ax.axvline(xs, color='gray', lw=0.4, ls=':')
ax.set_xlabel('x')
ax.set_ylabel('p(x)')
ax.set_title(rf'Pressure field  (Darcy flux q={q_true:.4f})'
             '\nObservation operator H: p(x) at sensor locations')
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(OUT / "fig3_problem_setup.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  fig3_problem_setup.png")

# ============================================================
# FIGURE 4 — MCMC diagnostics
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("MCMC Diagnostics  (pCN + Gibbs sampler)", fontsize=13)

ax = axes[0]
ax.plot(ll_trace, color='steelblue', lw=0.6, alpha=0.85)
half = len(ll_trace) // 2
ax.axhline(ll_trace[half:].mean(), color='red', ls='--', lw=1.5,
           label=f'posterior mean (2nd half) = {ll_trace[half:].mean():.1f}')
ax.set_xlabel('post-burnin sample')
ax.set_ylabel('log-likelihood')
ax.set_title('Log-likelihood trace\n'
             r'$\log p(d|\xi,\sigma^2)$ — measures fit to experiment')
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(s2_post, color='darkorange', lw=0.6, alpha=0.85)
ax.axhline(SIGMA2_TRUE, color='red', ls='--', lw=2,
           label=rf'true $\sigma^2={SIGMA2_TRUE:.2e}$')
ax.axhline(s2_post.mean(), color='k', ls=':', lw=1.5,
           label=rf'posterior mean={s2_post.mean():.2e}')
ax.set_xlabel('post-burnin sample')
ax.set_ylabel(r'$\sigma_\varepsilon^2$')
ax.set_title(r'Noise variance $\sigma_\varepsilon^2$ trace'
             '\nGibbs step: exact InvGamma conjugate draw')
ax.legend(fontsize=8)

# sigma2 posterior histogram with analytical InvGamma posterior overlay
ax = axes[2]
xi_pm = xi_post.mean(axis=0)
r_pm  = d_obs - G(xi_pm)
ap, bp = inv_gamma_posterior(ALPHA0, BETA0, r_pm)

lo, hi = np.percentile(s2_post, [0.5, 99.5])
xv = np.linspace(max(lo * 0.5, 1e-8), hi * 1.5, 500)
from uq_porous_media import inv_gamma_pdf

ax.hist(s2_post, bins=80, density=True, color='darkorange', alpha=0.55,
        label='MCMC samples')
ax.plot(xv, inv_gamma_pdf(xv, ap, bp), 'r-', lw=2.0,
        label=rf'InvGamma($\alpha$={ap:.1f}, $\beta$={bp:.2e})'
              r' (conj. posterior at $\hat{\xi}$)')
ax.axvline(SIGMA2_TRUE, color='k', ls='--', lw=2,
           label=rf'true $\sigma^2={SIGMA2_TRUE:.2e}$')
ax.set_xlabel(r'$\sigma_\varepsilon^2$')
ax.set_title(r'Posterior of $\sigma_\varepsilon^2$'
             '\nInvGamma conjugate update (closed form)')
ax.legend(fontsize=7)
ax.set_xlim(0, hi * 1.5)

plt.tight_layout()
fig.savefig(OUT / "fig4_mcmc_diagnostics.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  fig4_mcmc_diagnostics.png")

# ============================================================
# FIGURE 5 — Posterior vs. true field (main UQ result)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Posterior vs. True Permeability Field  (main UQ result)", fontsize=13)

idx_sub = rng.choice(n_keep, size=min(300, n_keep), replace=False)

ax = axes[0]
for i in idx_sub[:150]:
    ax.plot(x, log_k_post[i], color='steelblue', alpha=0.04, lw=0.5)
ax.plot(x, log_k_mean, 'b-', lw=2.5, label='posterior mean')
ax.fill_between(x,
                log_k_mean - 2 * log_k_std,
                log_k_mean + 2 * log_k_std,
                alpha=0.25, color='blue', label='95% credible interval')
ax.plot(x, log_k_true, 'r-', lw=2.0, label='true log k(x)')
rmse_prior = np.sqrt(np.mean((MU_Y - log_k_true) ** 2))
rmse_post  = np.sqrt(np.mean((log_k_mean - log_k_true) ** 2))
coverage   = np.mean(np.abs(log_k_true - log_k_mean) < 2 * log_k_std)
ax.set_xlabel('x')
ax.set_ylabel('log k(x)')
ax.set_title(f'Log-permeability: prior RMSE={rmse_prior:.3f}  →  '
             f'posterior RMSE={rmse_post:.3f}\n'
             f'95% CI coverage = {coverage * 100:.0f}%')
ax.legend(fontsize=9)

ax = axes[1]
p_post_arr = []
for i in idx_sub:
    k_i = np.exp(log_k_post[i])
    p_i, _ = solve_darcy_1d(k_i, x)
    p_post_arr.append(p_i)
p_post_arr = np.array(p_post_arr)
p_pm   = p_post_arr.mean(axis=0)
p_pstd = p_post_arr.std(axis=0)

for i in range(min(80, len(p_post_arr))):
    ax.plot(x, p_post_arr[i], color='steelblue', alpha=0.06, lw=0.5)
ax.plot(x, p_pm, 'b-', lw=2.5, label='posterior mean pressure')
ax.fill_between(x, p_pm - 2 * p_pstd, p_pm + 2 * p_pstd,
                alpha=0.25, color='blue', label='95% credible interval')
ax.plot(x, p_true, 'r-', lw=2.0, label='true pressure')
ax.scatter(sensor_x, d_obs, color='red', s=80, zorder=7, marker='v',
           label=rf'observations ($\sigma_\varepsilon$={SIGMA_EPS})')
ax.set_xlabel('x')
ax.set_ylabel('p(x)')
ax.set_title('Posterior predictive pressure vs. experiment\n'
             r'Uncertainty in $k(x)$ propagates to $p(x)$')
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(OUT / "fig5_posterior_field.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  fig5_posterior_field.png")

# ============================================================
# FIGURE 6 — Posterior predictive check in data space
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Posterior Predictive Check  (data space)", fontsize=13)

d_pmean = d_post_pred.mean(axis=0)
d_pstd  = d_post_pred.std(axis=0)

ax.errorbar(sensor_x, d_pmean, yerr=2 * d_pstd,
            fmt='bs', ms=9, capsize=6, lw=1.5, capthick=1.5,
            label='posterior predictive  (mean ± 2σ)')
ax.scatter(sensor_x, d_obs, color='red', s=90, zorder=7, marker='v',
           label=rf'noisy observations ($\sigma_\varepsilon$={SIGMA_EPS})')
ax.scatter(sensor_x, d_noiseless, color='k', s=60, zorder=8, marker='+',
           linewidths=2, label='noiseless truth')
ax.set_xlabel('Sensor location  x')
ax.set_ylabel('Pressure  p(x)')
ax.set_title('Does the inferred permeability field reproduce the experiment?\n'
             'Posterior predictive ≈ observations  ✓  if calibrated')
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUT / "fig6_posterior_predictive.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  fig6_posterior_predictive.png")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 62)
print("  RESULTS SUMMARY")
print("=" * 62)
print(f"  True noise sigma_eps         : {SIGMA_EPS:.4f}")
print(f"  Posterior noise sigma (mean) : {np.sqrt(s2_post).mean():.4f}"
      f"  (std={np.sqrt(s2_post).std():.4f})")
print(f"  Log-k RMSE — prior mean      : {rmse_prior:.4f}")
print(f"  Log-k RMSE — posterior mean  : {rmse_post:.4f}")
print(f"  95% CI coverage (log-k)      : {coverage * 100:.1f}%  (target ~95%)")
print(f"  pCN acceptance rate          : {results['accept_rate']:.3f}")
print(f"\n  Figures saved to ./figures/")

# Wiener-Askey connection note
print("\n  ── Wiener-Askey PCE note ──────────────────────────────")
print("  For polynomial chaos surrogates the input distribution")
print("  fixes the orthogonal polynomial basis:")
print("    xi ~ Normal   ->  Hermite  (used here)")
print("    xi ~ Gamma    ->  Laguerre")
print("    xi ~ Uniform  ->  Legendre")
print("    xi ~ Beta     ->  Jacobi")
print("  ────────────────────────────────────────────────────────")
