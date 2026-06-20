"""
Conjugate prior demo — 5 figures that build the concept step by step.

Run:   python conjugate_priors/demo_conjugate.py

Figure layout
─────────────
  fig1_triptych.png         Prior × Likelihood ∝ Posterior (NN case)
  fig2_sequential_nn.png    Sequential updating — NormalNormal (mu)
  fig3_sequential_nig.png   Sequential updating — NormalInvGamma (sigma^2)
  fig4_precision.png        Why precisions add (information geometry)
  fig5_joint_nig.png        Joint NIG: 2D posterior contours
  fig6_darcy_gibbs.png      Connection back to the Darcy problem
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from conjugate_priors.conjugate_pairs import NormalNormal, NormalInvGamma, NormalNIG

# ── global settings ──────────────────────────────────────────────────
SEED       = 7
MU_TRUE    = 1.5        # true mean we are trying to infer
SIGMA_TRUE = 0.40       # true noise std
S2_TRUE    = SIGMA_TRUE ** 2
N_DATA     = 50         # total data available

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

rng  = np.random.default_rng(SEED)
data = rng.normal(MU_TRUE, SIGMA_TRUE, size=N_DATA)

# Priors for all three cases
NN0   = NormalNormal(mu0=0.0, tau0=2.0, sigma=SIGMA_TRUE)
NIG0  = NormalInvGamma(alpha0=2.0, beta0=0.5, mu=MU_TRUE)
NIGJ0 = NormalNIG(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=0.5)


# ════════════════════════════════════════════════════════════════════════
# Helper: build a list of posteriors after seeing data[0:n] for each n
# ════════════════════════════════════════════════════════════════════════

def sequential_posteriors(model0, checkpoints):
    """Return dict {n: updated_model} after seeing data[:n]."""
    out = {0: model0}
    cur = model0
    for n in range(1, max(checkpoints) + 1):
        cur = cur.update([data[n - 1]])
        if n in checkpoints:
            out[n] = cur
    return out


CHECKPOINTS = [0, 1, 3, 8, 20, 50]
nn_states  = sequential_posteriors(NN0,  CHECKPOINTS)
nig_states = sequential_posteriors(NIG0, CHECKPOINTS)

print("Generated posteriors for checkpoints:", CHECKPOINTS)


# ════════════════════════════════════════════════════════════════════════
# Figure 1  — The triptych: prior × likelihood ∝ posterior
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle(
    "Conjugate prior: prior  ×  likelihood  ∝  posterior  (SAME family)\n"
    "Case: Normal likelihood, Normal prior on  μ  (σ known)",
    fontsize=12, fontweight='bold'
)

mu_grid = np.linspace(-4, 6, 600)
n_demo  = 8                          # use 8 data points for illustration
d_demo  = data[:n_demo]
x_bar   = d_demo.mean()
nn_demo = nn_states[n_demo]

# --- panel 1: prior ---
ax = axes[0]
prior_pdf = stats.norm.pdf(mu_grid, NN0.mu0, NN0.tau0)
ax.fill_between(mu_grid, prior_pdf, alpha=0.25, color='steelblue')
ax.plot(mu_grid, prior_pdf, 'steelblue', lw=2.5,
        label=rf'$\mu \sim \mathcal{{N}}({NN0.mu0}, {NN0.tau0}^2)$')
ax.axvline(MU_TRUE, color='red', ls='--', lw=1.5, label=rf'true $\mu={MU_TRUE}$')
ax.set_title('① Prior  $p(\\mu)$\n(before any data)', fontsize=10)
ax.set_xlabel('μ');  ax.set_ylabel('density')
ax.legend(fontsize=8);  ax.set_xlim(mu_grid[[0,-1]])

# --- panel 2: likelihood as a function of μ ---
ax = axes[1]
# L(μ) = ∏ N(x_i; μ, σ²)  as function of μ = N(μ; x_bar, σ²/n)
lik_unnorm = stats.norm.pdf(mu_grid, x_bar, SIGMA_TRUE / np.sqrt(n_demo))
ax.fill_between(mu_grid, lik_unnorm, alpha=0.25, color='darkorange')
ax.plot(mu_grid, lik_unnorm, 'darkorange', lw=2.5,
        label=rf'$L(\mu)\propto\mathcal{{N}}(\bar{{x}}, \sigma^2/n)$')
for xi in d_demo:
    ax.axvline(xi, color='gray', lw=0.5, alpha=0.6)
ax.scatter(d_demo, np.zeros(n_demo) - 0.02,
           color='k', s=20, zorder=5, label=f'{n_demo} data points')
ax.axvline(x_bar, color='k', ls=':', lw=1.5, label=rf'$\bar{{x}}={x_bar:.2f}$')
ax.set_title(f'② Likelihood  $L(\\mu)$\n(as function of μ, n={n_demo})', fontsize=10)
ax.set_xlabel('μ');  ax.legend(fontsize=8);  ax.set_xlim(mu_grid[[0,-1]])

# --- panel 3: posterior ---
ax = axes[2]
post_pdf = nn_demo.pdf(mu_grid)
ax.fill_between(mu_grid, post_pdf, alpha=0.25, color='seagreen')
ax.plot(mu_grid, post_pdf, 'seagreen', lw=2.5,
        label=(rf'$\mu|data \sim \mathcal{{N}}({nn_demo.mean:.2f},'
               rf'{nn_demo.std:.2f}^2)$'))
ax.plot(mu_grid, prior_pdf, 'steelblue', lw=1.2, ls='--', alpha=0.5, label='prior')
ax.axvline(MU_TRUE, color='red', ls='--', lw=1.5, label=rf'true $\mu={MU_TRUE}$')
ax.set_title('③ Posterior  $p(\\mu\\,|\\,data)$\n(same Normal family, updated params)',
             fontsize=10)
ax.set_xlabel('μ');  ax.legend(fontsize=8);  ax.set_xlim(mu_grid[[0,-1]])

plt.tight_layout()
fig.savefig(OUT / 'fig1_triptych.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  fig1_triptych.png")


# ════════════════════════════════════════════════════════════════════════
# Figure 2 — Sequential updating: NormalNormal
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
fig.suptitle(
    "Sequential Bayesian updating — Case 1: Normal-Normal\n"
    "Each new datum narrows the posterior (same Normal family throughout)",
    fontsize=12, fontweight='bold'
)

colors = plt.cm.Blues(np.linspace(0.35, 0.95, len(CHECKPOINTS)))

for ax, n, col in zip(axes.flat, CHECKPOINTS, colors):
    state = nn_states[n]
    pdf   = state.pdf(mu_grid)
    ax.fill_between(mu_grid, pdf, alpha=0.30, color=col)
    ax.plot(mu_grid, pdf, color=col, lw=2.5,
            label=rf'$\mathcal{{N}}({state.mean:.2f},\,{state.std:.2f}^2)$')
    if n > 0:
        ax.plot(mu_grid, NN0.pdf(mu_grid), 'gray', lw=1.0, ls='--', alpha=0.6,
                label='prior')
        ax.scatter(data[:n], np.zeros(n) - max(pdf)*0.04,
                   color='k', s=12, alpha=0.5, zorder=4)
    ax.axvline(MU_TRUE, color='red', ls='--', lw=1.5,
               label=rf'true $\mu={MU_TRUE}$' if n == 0 else None)
    ax.set_title(f'n = {n}' + (' (prior)' if n == 0 else ''), fontsize=10)
    ax.set_xlabel('μ');  ax.set_ylabel('density')
    ax.legend(fontsize=7);  ax.set_xlim(-3, 4.5)
    ax.set_ylim(bottom=-max(pdf)*0.07)

plt.tight_layout()
fig.savefig(OUT / 'fig2_sequential_nn.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  fig2_sequential_nn.png")


# ════════════════════════════════════════════════════════════════════════
# Figure 3 — Sequential updating: NormalInvGamma
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
fig.suptitle(
    "Sequential Bayesian updating — Case 2: Normal-InvGamma\n"
    r"Inferring $\sigma^2$ (μ known); same InvGamma family throughout",
    fontsize=12, fontweight='bold'
)

# x-axis range: focus around the truth
s2_lo, s2_hi = S2_TRUE * 0.05, S2_TRUE * 6.0
s2_grid = np.linspace(s2_lo, s2_hi, 600)
colors2 = plt.cm.Oranges(np.linspace(0.35, 0.95, len(CHECKPOINTS)))

for ax, n, col in zip(axes.flat, CHECKPOINTS, colors2):
    state = nig_states[n]
    pdf   = state.pdf(s2_grid)
    ax.fill_between(s2_grid, pdf, alpha=0.30, color=col)
    ax.plot(s2_grid, pdf, color=col, lw=2.5,
            label=(rf'InvGamma({state.alpha_n:.1f}, {state.beta_n:.3f})'
                   f'\nmean={state.mean:.3f}'))
    if n > 0:
        ax.plot(s2_grid, NIG0.pdf(s2_grid), 'gray', lw=1.0, ls='--', alpha=0.5,
                label='prior')
    ax.axvline(S2_TRUE, color='red', ls='--', lw=1.5,
               label=rf'true $\sigma^2={S2_TRUE:.3f}$')
    ax.set_title(f'n = {n}' + (' (prior)' if n == 0 else ''), fontsize=10)
    ax.set_xlabel(r'$\sigma^2$');  ax.set_ylabel('density')
    ax.legend(fontsize=7);  ax.set_xlim(s2_lo, s2_hi)

plt.tight_layout()
fig.savefig(OUT / 'fig3_sequential_nig.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  fig3_sequential_nig.png")


# ════════════════════════════════════════════════════════════════════════
# Figure 4 — Information geometry: why PRECISIONS add
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle(
    "Information geometry of conjugate updates\n"
    "Bayesian updating = adding information (not averaging)",
    fontsize=12, fontweight='bold'
)

ns = np.arange(0, N_DATA + 1)

# --- panel 1: posterior variance tau_n^2 vs n (hyperbolic decrease) ---
ax = axes[0]
variances = [1.0 / (1.0/NN0.tau0**2 + n/SIGMA_TRUE**2) for n in ns]
ax.plot(ns, variances, 'C0-', lw=2.5)
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('n  (number of observations)')
ax.set_ylabel(r'Posterior variance  $\tau_n^2$')
ax.set_title(r'Variance $\tau_n^2 = 1\,/\,(1/\tau_0^2 + n/\sigma^2)$'
             '\nDecreases as  1/n  for large n', fontsize=9)
ax.annotate(r'$\tau_0^2 = 4$', xy=(0, NN0.tau0**2), xytext=(5, 3.2),
            arrowprops=dict(arrowstyle='->', color='k'), fontsize=9)

# --- panel 2: posterior PRECISION 1/tau_n^2 vs n (LINEAR!) ---
ax = axes[1]
precisions = [1.0/NN0.tau0**2 + n/SIGMA_TRUE**2 for n in ns]
ax.plot(ns, precisions, 'C1-', lw=2.5)
ax.set_xlabel('n')
ax.set_ylabel(r'Posterior precision  $1/\tau_n^2$')
ax.set_title(r'Precision $1/\tau_n^2 = 1/\tau_0^2 + n/\sigma^2$'
             '\n(prior precision + data precision — LINEAR in n)', fontsize=9)
slope = 1.0 / SIGMA_TRUE**2
ax.annotate(rf'slope = $1/\sigma^2 = {slope:.1f}$',
            xy=(25, precisions[25]), xytext=(10, precisions[10] + 5),
            arrowprops=dict(arrowstyle='->', color='k'), fontsize=9)

# --- panel 3: same for InvGamma alpha_n vs n ---
ax = axes[2]
alphas = [NIG0.alpha0 + n/2.0 for n in ns]
ax.plot(ns, alphas, 'C2-', lw=2.5, label=r'$\alpha_n = \alpha_0 + n/2$')
ax.set_xlabel('n')
ax.set_ylabel(r'$\alpha_n$  (InvGamma shape)')
ax.set_title(r'InvGamma shape accumulates: $\alpha_n = \alpha_0 + n/2$'
             '\nEquivalent: "half-observation" per data point', fontsize=9)
ax.annotate(rf'slope = 1/2', xy=(30, NIG0.alpha0 + 30/2),
            xytext=(10, NIG0.alpha0 + 15),
            arrowprops=dict(arrowstyle='->', color='k'), fontsize=9)
ax.text(0.05, 0.92,
        r'$\beta_n = \beta_0 + \sum(x_i-\mu)^2/2$' + '\n(also additive)',
        transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
fig.savefig(OUT / 'fig4_precision.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  fig4_precision.png")


# ════════════════════════════════════════════════════════════════════════
# Figure 5 — Joint NIG: 2D posterior contours evolving with n
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Case 3: Normal-InvGamma joint prior — inferring both  μ  and  σ²\n"
    "Posterior contours in  (μ, σ²)  space shrink toward truth",
    fontsize=12, fontweight='bold'
)

mu_g   = np.linspace(-1.0, 4.0, 200)
s2_g   = np.linspace(0.01, 0.9, 200)
MU_2D, S2_2D = np.meshgrid(mu_g, s2_g, indexing='ij')

nig_checkpoints = [0, 8, 50]
nig_j_states    = sequential_posteriors(NIGJ0, nig_checkpoints)
cmaps           = ['Blues', 'Oranges', 'Greens']

for ax, n, cmap in zip(axes, nig_checkpoints, cmaps):
    state = nig_j_states[n]
    log_p = state.joint_log_density(MU_2D, S2_2D)
    log_p -= log_p.max()   # normalise for plotting stability
    p     = np.exp(log_p)
    p    /= p.sum()

    levels = np.quantile(p[p > 0], [0.01, 0.25, 0.55, 0.80, 0.95])
    cs = ax.contourf(mu_g, s2_g, p.T, levels=levels, cmap=cmap, alpha=0.75)
    ax.contour(mu_g, s2_g, p.T,  levels=levels, colors='k', linewidths=0.5,
               alpha=0.4)
    ax.scatter([MU_TRUE], [S2_TRUE], marker='*', s=250, color='red',
               zorder=10, label=rf'truth  $(\mu={MU_TRUE},\sigma^2={S2_TRUE})$')
    ax.set_xlabel('μ');  ax.set_ylabel(r'$\sigma^2$')
    title = 'Prior  (n=0)' if n == 0 else f'Posterior  (n={n})'
    ax.set_title(
        title + f'\nμ_n={state.mean_mu:.2f}  σ²_n={state.mean_sigma2:.3f}',
        fontsize=10
    )
    ax.legend(fontsize=7, loc='upper right')

plt.tight_layout()
fig.savefig(OUT / 'fig5_joint_nig.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  fig5_joint_nig.png")


# ════════════════════════════════════════════════════════════════════════
# Figure 6 — Connection to the Darcy Gibbs step
# ════════════════════════════════════════════════════════════════════════
"""
In the Darcy MCMC (demo_uq.py), the Gibbs step is exactly Case 2:
    xi fixed => residuals r = d_obs - G(xi)
    sigma^2_eps | xi, d ~ InvGamma(alpha0 + n_obs/2,  beta0 + ||r||^2/2)

Here we reconstruct this analytically given a toy residual vector
to show what the conjugate update does concretely.
"""

# Simulate what happens inside one Gibbs step
N_OBS       = 7                        # number of sensor observations
SIGMA2_EPS  = 0.02 ** 2               # true noise variance in Darcy demo
rng2        = np.random.default_rng(99)
residuals   = rng2.normal(0, np.sqrt(SIGMA2_EPS), size=N_OBS)

alpha0_d = 2.0;  beta0_d = 1e-3       # InvGamma hyperprior used in demo_uq.py
nig_darcy_prior = NormalInvGamma(alpha0=alpha0_d, beta0=beta0_d, mu=0.0)
nig_darcy_post  = nig_darcy_prior.update(residuals)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Connection to the Darcy problem: the InvGamma Gibbs step\n"
    r"One conjugate draw replaces one Metropolis accept/reject for $\sigma^2_\varepsilon$",
    fontsize=12, fontweight='bold'
)

s2_lo_d = 1e-6;  s2_hi_d = 4 * SIGMA2_EPS * 6
s2_g_d  = np.linspace(s2_lo_d, s2_hi_d, 800)

# --- panel 1: prior vs posterior ---
ax = axes[0]
pdf_prior = nig_darcy_prior.pdf(s2_g_d)
pdf_post  = nig_darcy_post.pdf(s2_g_d)

ax.fill_between(s2_g_d * 1e4, pdf_prior / (1e4), alpha=0.20, color='steelblue',
                label=(rf'Prior: InvGamma({alpha0_d}, {beta0_d:.0e})'))
ax.fill_between(s2_g_d * 1e4, pdf_post  / (1e4), alpha=0.35, color='darkorange',
                label=(rf'Posterior: InvGamma({nig_darcy_post.alpha_n:.1f},'
                        rf'{nig_darcy_post.beta_n:.2e})'))
ax.plot(s2_g_d * 1e4, pdf_prior / (1e4), 'steelblue', lw=2)
ax.plot(s2_g_d * 1e4, pdf_post  / (1e4), 'darkorange', lw=2.5)
ax.axvline(SIGMA2_EPS * 1e4, color='red', ls='--', lw=2,
           label=rf'true $\sigma^2={SIGMA2_EPS:.2e}$')
ax.set_xlabel(r'$\sigma^2_\varepsilon \times 10^{-4}$')
ax.set_ylabel('density (scaled)')
ax.set_title(f'Prior → Posterior after {N_OBS} Darcy residuals', fontsize=10)
ax.legend(fontsize=8);  ax.set_xlim(0, s2_hi_d * 1e4 * 0.8)

# --- panel 2: annotated update formula ---
ax = axes[1]
ax.axis('off')
alpha_n = nig_darcy_post.alpha_n
beta_n  = nig_darcy_post.beta_n
ss      = np.dot(residuals, residuals)

formula = (
    "  Conjugate Gibbs step — one closed-form draw\n"
    "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "  Model:\n"
    "    d_obs = G(ξ) + ε,   ε ~ N(0, σ²_ε I)\n\n"
    "  Hyperprior:\n"
    f"    σ²_ε ~ InvGamma({alpha0_d}, {beta0_d:.0e})\n\n"
    "  Given residuals  r = d_obs - G(ξ),\n"
    "  the full conditional is EXACT:\n\n"
    "    σ²_ε | ξ, d  ~  InvGamma(α_n, β_n)\n\n"
    f"    α_n = α₀ + n/2  =  {alpha0_d} + {N_OBS}/2  =  {alpha_n:.1f}\n"
    f"    β_n = β₀ + ‖r‖²/2  =  {beta0_d:.1e} + {ss/2:.2e}  =  {beta_n:.2e}\n\n"
    f"    E[σ²_ε | data]  =  β_n/(α_n-1)  =  {nig_darcy_post.mean:.2e}\n"
    f"    true σ²_ε       =  {SIGMA2_EPS:.2e}\n\n"
    "  ➜ No accept/reject needed — just sample from InvGamma."
)
ax.text(0.05, 0.97, formula, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                  edgecolor='darkorange', linewidth=1.5))

plt.tight_layout()
fig.savefig(OUT / 'fig6_darcy_gibbs.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  fig6_darcy_gibbs.png")


# ════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════
print()
print("=" * 62)
print("  CONJUGATE PRIOR SUMMARY")
print("=" * 62)
print()
rows = [
    ("Likelihood",      "Prior on",    "Prior family",  "Posterior family", "Update rule"),
    ("─"*14,            "─"*12,        "─"*14,          "─"*18,             "─"*24),
    ("Normal(μ,σ²)",    "μ  [σ known]","Normal",        "Normal",
     "prec_n = prec_0 + n/σ²"),
    ("Normal(μ,σ²)",    "σ² [μ known]","InvGamma",      "InvGamma",
     "α_n=α₀+n/2, β_n=β₀+SS/2"),
    ("Normal(μ,σ²)",    "(μ,σ²) joint","Normal-InvGamma","Normal-InvGamma",
     "κ_n=κ₀+n, α_n=α₀+n/2, ..."),
]
for r in rows:
    print(f"  {r[0]:<16} {r[1]:<14} {r[2]:<16} {r[3]:<20} {r[4]}")
print()
print("  Figures saved to ./conjugate_priors/figures/")
