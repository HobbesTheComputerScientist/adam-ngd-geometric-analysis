import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 2.2,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.2,
    "grid.linewidth": 0.6,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gamma_indicator import compute_gamma, compute_fisher_vector_product


torch.manual_seed(42)
np.random.seed(42)


# ── Dataset ──────────────────────────────────────────────
N = 100
D = 10


scale = torch.ones(D)
scale[:5] = 10.0
scale[5:] = 0.1
X_raw = torch.randn(N, D)
X = X_raw * scale


true_weights = torch.randn(D, 1)
y = X @ true_weights + 0.1 * torch.randn(N, 1)


# ── Dataset diagnostics ───────────────────────────────────
eigenvalues = torch.linalg.eigvalsh(X.T @ X)
print(f"Largest eigenvalue:  {eigenvalues[-1].item():.2f}")
print(f"Smallest eigenvalue: {eigenvalues[0].item():.2f}")
print(f"Condition number:    {(eigenvalues[-1]/eigenvalues[0]).item():.2f}")
print("High condition number = ill-conditioned landscape\n")


X_raw = torch.randn(N, D)
X = X_raw * scale


true_weights = torch.randn(D, 1)
y = X @ true_weights + 0.1 * torch.randn(N, 1)


print(f"Condition number of X^T X: "
      f"{torch.linalg.cond(X.T @ X).item():.2f}")
print("High condition number = ill-conditioned landscape\n")


# ── Model ─────────────────────────────────────────────────
class LinearModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=False)

    def forward(self, x):
        return self.fc(x)


# ── Training loop ─────────────────────────────────────────
def train_and_measure(optimizer_name, n_steps=2000, lr=0.01):
    model = LinearModel(D)
    params = list(model.parameters())

    m = torch.zeros(D)
    v = torch.zeros(D)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    gammas = []
    losses = []

    for step in range(1, n_steps + 1):
        output = model(X)
        loss = nn.MSELoss()(output, y)

        grads = torch.autograd.grad(
            loss, params,
            create_graph=True,
            retain_graph=True
        )
        grad_flat = torch.cat([g.reshape(-1) for g in grads])

        if optimizer_name == 'SGD':
            update_flat = -lr * grad_flat.detach()

        elif optimizer_name == 'Adam':
            m = beta1 * m + (1 - beta1) * grad_flat.detach()
            v = beta2 * v + (1 - beta2) * grad_flat.detach()**2
            m_hat = m / (1 - beta1**step)
            v_hat = v / (1 - beta2**step)
            update_flat = -lr * m_hat / (
                torch.sqrt(v_hat) + eps
            )

        elif optimizer_name == 'EF':
            ef_diag = grad_flat.detach()**2 + eps
            update_flat = -lr * grad_flat.detach() / ef_diag

        elif optimizer_name == 'iEF':
            # Improved Empirical Fisher (Wu et al., 2024)
            # Vectorized: per-sample gradients for linear model
            # are residual * input, computed in one pass
            preds = model(X).detach()          # (N, 1)
            residuals = preds - y              # (N, 1)
            per_sample_grads = residuals * X   # (N, D)
            norms_sq = (per_sample_grads ** 2).sum(dim=1, keepdim=True) + eps
            ief_diag = (per_sample_grads ** 2 / norms_sq).mean(dim=0) + eps
            update_flat = -lr * grad_flat.detach() / ief_diag

        elif optimizer_name == 'NGD':
            damping = 1e-3
            fisher_diag = (X ** 2).mean(dim=0)
            natural_grad = grad_flat.detach() / (fisher_diag + damping)
            update_flat = -lr * natural_grad

        update_list = []
        idx = 0
        for p in params:
            size = p.numel()
            update_list.append(
                update_flat[idx:idx+size].reshape(p.shape)
            )
            idx += size

        gamma_val = compute_gamma(
            update_list, loss, params, grads
        )
        gammas.append(gamma_val)
        losses.append(loss.item())

        with torch.no_grad():
            for p, u in zip(params, update_list):
                p.add_(u)

        if step % 50 == 0:
            print(f"{optimizer_name} | Step {step} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Gamma: {gamma_val:.4f}")

    return gammas, losses


# ── Run all optimizers ─────────────────────────────────────
print("=" * 50)
print("Experiment 2: Ill-Conditioned Linear Regression")
print("=" * 50)


results = {}
for opt in ['SGD', 'Adam', 'EF', 'iEF', 'NGD']:
    if opt == 'SGD':
        lr = 0.0001
    elif opt in ['Adam', 'EF', 'iEF']:
        lr = 0.005
    else:  # NGD
        lr = 0.1
    print(f"\nRunning {opt}...")
    gammas, losses = train_and_measure(opt, lr=lr)
    results[opt] = {'gammas': gammas, 'losses': losses}


# ── Save ───────────────────────────────────────────────────
os.makedirs('../results', exist_ok=True)
os.makedirs('../figures', exist_ok=True)
np.save('../results/exp2_results.npy', results)
print("\nResults saved.")


# ── Plot ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(12.8, 4.8),
    gridspec_kw={'wspace': 0.22}
)

colors = {
    'SGD':  '#4C78A8',
    'Adam': '#E45756',
    'EF':   '#F2B134',
    'iEF':  '#54A24B',
    'NGD':  '#7A7A7A',
}

steps = np.arange(1, 2001)

# Gamma panel
for opt in ['SGD', 'Adam', 'EF', 'iEF']:
    gammas = np.array(results[opt]['gammas'], dtype=float)
    valid = np.isfinite(gammas)

    if opt == 'Adam':
        lw, alpha, z = 2.6, 0.95, 4
    elif opt == 'iEF':
        lw, alpha, z = 2.2, 0.92, 3
    elif opt == 'EF':
        lw, alpha, z = 1.0, 0.25, 1
    else:
        lw, alpha, z = 2.0, 0.88, 2

    ax1.plot(
        steps[valid],
        gammas[valid],
        label=opt,
        color=colors[opt],
        lw=lw,
        alpha=alpha,
        zorder=z
    )

ax1.set_xlabel('Training step')
ax1.set_ylabel(r'$\gamma(\Delta \theta)$')
ax1.set_yscale('log')
ax1.set_ylim(5e-2, 1e3)
ax1.set_title('Approximation quality to NGD', pad=12, fontsize=11.5, fontweight='semibold')
ax1.grid(True, which='major', alpha=0.16)
ax1.grid(False, which='minor')
ax1.tick_params(direction='out')

# Loss panel
for opt in ['SGD', 'Adam', 'EF', 'iEF', 'NGD']:
    losses = np.array(results[opt]['losses'], dtype=float)

    if opt == 'Adam':
        ax2.plot(steps, losses, label=opt, color=colors[opt], lw=2.6, alpha=0.95, zorder=4)
    elif opt == 'NGD':
        ax2.plot(steps, losses, label=opt, color=colors[opt], lw=2.4, alpha=0.95, zorder=5)
    elif opt == 'iEF':
        ax2.plot(steps, losses, label=opt, color=colors[opt], lw=2.2, alpha=0.95, zorder=6)
    elif opt == 'EF':
        ax2.plot(steps, losses, label=opt, color=colors[opt], lw=1.4, alpha=0.55, zorder=1)
    else:
        ax2.plot(steps, losses, label=opt, color=colors[opt], lw=2.0, alpha=0.88, zorder=2)

ax2.set_xlabel('Training step')
ax2.set_ylabel('MSE loss')
ax2.set_yscale('log')
ax2.set_ylim(7e-3, 2e3)
ax2.set_title('Training loss', pad=12, fontsize=11.5, fontweight='semibold')
ax2.grid(True, which='major', alpha=0.16)
ax2.grid(False, which='minor')
ax2.tick_params(direction='out')

handles, labels = ax2.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc='upper center',
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, 0.985),
    handlelength=2.2,
    columnspacing=1.5
)

fig.suptitle('Ill-conditioned linear regression', y=1.03, fontsize=12, fontweight='semibold')

fig.subplots_adjust(
    top=0.80,
    bottom=0.14,
    left=0.08,
    right=0.98,
    wspace=0.22
)

plt.savefig('../figures/exp2_ill_conditioned.pdf')
plt.savefig('../figures/exp2_ill_conditioned.png', dpi=300)
plt.show()
print("Figure saved.")