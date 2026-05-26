import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
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
from gamma_indicator import compute_gamma


torch.manual_seed(42)
np.random.seed(42)


N = 200
D = 10


X = torch.randn(N, D)
true_weights = torch.randn(D, 1)
logits = X @ true_weights
y = (torch.sigmoid(logits) > 0.5).float()


class LogisticModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=False)
    def forward(self, x):
        return self.fc(x)


def train_and_measure(optimizer_name, n_steps=2000, lr=0.01):
    model = LogisticModel(D)
    params = list(model.parameters())
    m = torch.zeros(D)
    v = torch.zeros(D)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    gammas = []
    losses = []
    loss_fn = nn.BCEWithLogitsLoss()

    for step in range(1, n_steps + 1):
        output = model(X)
        loss = loss_fn(output, y)

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
            update_flat = -lr * m_hat / (torch.sqrt(v_hat) + eps)

        elif optimizer_name == 'EF':
            ef_diag = grad_flat.detach()**2 + eps
            update_flat = -lr * grad_flat.detach() / ef_diag

        elif optimizer_name == 'iEF':
            # Improved Empirical Fisher (Wu et al., 2024)
            with torch.no_grad():
                preds = torch.sigmoid(model(X))        # (N, 1)
                residuals = preds - y                  # (N, 1)
                per_sample_grads = residuals * X       # (N, D)
            norms_sq = (per_sample_grads ** 2).sum(dim=1, keepdim=True) + eps
            ief_diag = (per_sample_grads ** 2 / norms_sq).mean(dim=0) + eps
            update_flat = -lr * grad_flat.detach() / ief_diag

        elif optimizer_name == 'NGD':
            with torch.no_grad():
                p = torch.sigmoid(model(X))
                weights = (p * (1 - p)).squeeze()
                fisher_diag = (weights.unsqueeze(1) * X**2).mean(dim=0)
            damping = 1e-3
            natural_grad = grad_flat.detach() / (fisher_diag + damping)
            update_flat = -lr * natural_grad

        update_list = []
        idx = 0
        for p_param in params:
            size = p_param.numel()
            update_list.append(update_flat[idx:idx+size].reshape(p_param.shape))
            idx += size

        gamma_val = compute_gamma(update_list, loss, params, grads)
        gammas.append(gamma_val)
        losses.append(loss.item())

        with torch.no_grad():
            for p_param, u in zip(params, update_list):
                p_param.add_(u)

        if step % 100 == 0:
            print(f"{optimizer_name} | Step {step} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Gamma: {gamma_val:.4f}")

    return gammas, losses


print("=" * 50)
print("Experiment 3: Logistic Regression")
print("=" * 50)


results = {}
for opt in ['SGD', 'Adam', 'EF', 'iEF', 'NGD']:
    if opt == 'NGD':
        lr = 0.1
    else:
        lr = 0.01
    print(f"\nRunning {opt}...")
    gammas, losses = train_and_measure(opt, lr=lr)
    results[opt] = {'gammas': gammas, 'losses': losses}


os.makedirs('../results', exist_ok=True)
os.makedirs('../figures', exist_ok=True)
np.save('../results/exp3_results.npy', results)


ngd_losses = results['NGD']['losses']
ngd_converge_step = next(
    (i+1 for i, l in enumerate(ngd_losses) if l < 0.3), None
)

# ── Plot ───────────────────────────────────────────────────
from matplotlib.lines import Line2D

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(12.8, 5.1),
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

# Gamma plot
for opt in ['SGD', 'Adam', 'EF', 'iEF']:
    gammas = np.array(results[opt]['gammas'], dtype=float)
    valid = np.isfinite(gammas)

    if opt == 'Adam':
        lw, alpha, z = 2.6, 0.95, 4
    elif opt == 'iEF':
        lw, alpha, z = 2.2, 0.92, 3
    elif opt == 'EF':
        lw, alpha, z = 1.1, 0.35, 1
    else:
        lw, alpha, z = 2.0, 0.88, 2

    ax1.plot(
        steps[valid],
        gammas[valid],
        color=colors[opt],
        lw=lw,
        alpha=alpha,
        zorder=z
    )

if ngd_converge_step:
    ax1.axvline(
        x=ngd_converge_step,
        color=colors['NGD'],
        linestyle='--',
        alpha=0.75,
        linewidth=1.4,
        zorder=2
    )

ax1.set_xlabel('Training step')
ax1.set_ylabel(r'$\gamma(\Delta \theta)$')
ax1.set_yscale('log')
ax1.set_ylim(5e-2, 1.5e3)
ax1.grid(True, which='major', alpha=0.16)
ax1.grid(False, which='minor')
ax1.tick_params(direction='out')
ax1.set_title('Approximation quality to NGD', pad=10, fontsize=11.5, fontweight='semibold')

# Loss plot
for opt in ['SGD', 'Adam', 'EF', 'iEF', 'NGD']:
    losses = np.array(results[opt]['losses'], dtype=float)

    if opt == 'Adam':
        lw, alpha, z = 2.6, 0.95, 4
    elif opt == 'iEF':
        lw, alpha, z = 2.2, 0.92, 3
    elif opt == 'EF':
        lw, alpha, z = 1.4, 0.55, 1
    elif opt == 'NGD':
        lw, alpha, z = 2.4, 0.95, 5
    else:
        lw, alpha, z = 2.0, 0.88, 2

    ax2.plot(
        steps,
        losses,
        color=colors[opt],
        lw=lw,
        alpha=alpha,
        zorder=z
    )

ax2.set_xlabel('Training step')
ax2.set_ylabel('Binary cross-entropy loss')
ax2.set_yscale('log')
ax2.set_ylim(7e-3, 2e3)
ax2.grid(True, which='major', alpha=0.16)
ax2.grid(False, which='minor')
ax2.tick_params(direction='out')
ax2.set_title('Training loss', pad=10, fontsize=11.5, fontweight='semibold')

# Custom legend so NGD is always included
legend_handles = [
    Line2D([0], [0], color=colors['SGD'],  lw=2.0, alpha=0.88, label='SGD'),
    Line2D([0], [0], color=colors['Adam'], lw=2.6, alpha=0.95, label='Adam'),
    Line2D([0], [0], color=colors['EF'],   lw=1.4, alpha=0.55, label='EF'),
    Line2D([0], [0], color=colors['iEF'],  lw=2.2, alpha=0.92, label='iEF'),
    Line2D([0], [0], color=colors['NGD'],  lw=2.4, alpha=0.95, label='NGD'),
]

fig.legend(
    handles=legend_handles,
    loc='upper center',
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, 0.955),
    handlelength=2.2,
    columnspacing=1.5
)

fig.suptitle(
    'Logistic regression',
    y=0.995,
    fontsize=12,
    fontweight='semibold'
)

fig.subplots_adjust(
    top=0.84,
    bottom=0.14,
    left=0.08,
    right=0.98,
    wspace=0.22
)

plt.savefig('../figures/exp3_logistic_regression.pdf')
plt.savefig('../figures/exp3_logistic_regression.png', dpi=300)
plt.show()
print("Figure saved.")