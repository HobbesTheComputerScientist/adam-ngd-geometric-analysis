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
D_in = 20
D_hidden = 32
D_out = 1


X = torch.randn(N, D_in)
true_net = nn.Sequential(
    nn.Linear(D_in, D_hidden, bias=False),
    nn.ReLU(),
    nn.Linear(D_hidden, D_out, bias=False)
)
with torch.no_grad():
    y = true_net(X) + 0.1 * torch.randn(N, D_out)


# ── Model ─────────────────────────────────────────────────
class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_in, D_hidden, bias=False)
        self.fc2 = nn.Linear(D_hidden, D_out, bias=False)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def train_and_measure(optimizer_name, n_steps=500, lr=0.01):
    model = SmallNet()
    params = list(model.parameters())
    D_total = sum(p.numel() for p in params)

    m = torch.zeros(D_total)
    v = torch.zeros(D_total)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    gammas = []
    losses = []
    loss_fn = nn.MSELoss()

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
            # For neural network: per-sample gradients computed explicitly
            # since there is no closed-form residual * input shortcut
            fisher_diag = torch.zeros(D_total)
            for i in range(N):
                out_i = model(X[i:i+1])
                loss_i = loss_fn(out_i, y[i:i+1])
                g_i = torch.autograd.grad(loss_i, params,
                                          retain_graph=False)
                g_i_flat = torch.cat([g.reshape(-1) for g in g_i])
                g_i_flat = g_i_flat.detach()
                fisher_diag += g_i_flat ** 2
            fisher_diag /= N
            # iEF normalisation: divide each per-sample sq-grad by its norm
            # Re-compute to apply normalisation properly
            G = []
            for i in range(N):
                out_i = model(X[i:i+1])
                loss_i = loss_fn(out_i, y[i:i+1])
                g_i = torch.autograd.grad(loss_i, params,
                                          retain_graph=False)
                g_i_flat = torch.cat([g.reshape(-1) for g in g_i]).detach()
                G.append(g_i_flat)
            G = torch.stack(G)                                   # (N, D_total)
            norms_sq = (G ** 2).sum(dim=1, keepdim=True) + eps  # (N, 1)
            ief_diag = (G ** 2 / norms_sq).mean(dim=0) + eps    # (D_total,)
            update_flat = -lr * grad_flat.detach() / ief_diag

        elif optimizer_name == 'NGD':
            damping = 1e-2
            fisher_diag = torch.zeros(D_total)
            for i in range(N):
                out_i = model(X[i:i+1])
                loss_i = loss_fn(out_i, y[i:i+1])
                g_i = torch.autograd.grad(loss_i, params,
                                          retain_graph=False)
                g_i_flat = torch.cat([g.reshape(-1) for g in g_i])
                fisher_diag += g_i_flat.detach()**2
            fisher_diag /= N
            natural_grad = grad_flat.detach() / (fisher_diag + damping)
            update_flat = -lr * natural_grad

        update_list = []
        idx = 0
        for p in params:
            size = p.numel()
            update_list.append(update_flat[idx:idx+size].reshape(p.shape))
            idx += size

        gamma_val = compute_gamma(update_list, loss, params, grads)
        gammas.append(gamma_val)
        losses.append(loss.item())

        with torch.no_grad():
            for p, u in zip(params, update_list):
                p.add_(u)

        if step % 100 == 0:
            print(f"{optimizer_name} | Step {step} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Gamma: {gamma_val:.4f}")

    return gammas, losses


print("=" * 50)
print("Experiment 4: Small Neural Network")
print("=" * 50)


results = {}
for opt in ['SGD', 'Adam', 'EF', 'iEF', 'NGD']:
    if opt == 'NGD':
        lr = 0.05
    elif opt == 'SGD':
        lr = 0.001
    else:
        lr = 0.01
    print(f"\nRunning {opt}...")
    gammas, losses = train_and_measure(opt, lr=lr)
    results[opt] = {'gammas': gammas, 'losses': losses}


os.makedirs('../results', exist_ok=True)
os.makedirs('../figures', exist_ok=True)
np.save('../results/exp4_results.npy', results)

# ── Plot ───────────────────────────────────────────────────
from matplotlib.lines import Line2D

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(13.6, 5.4),
    gridspec_kw={'wspace': 0.24}
)

colors = {
    'SGD':  '#4C78A8',
    'Adam': '#E45756',
    'EF':   '#F2B134',
    'iEF':  '#54A24B',
    'NGD':  '#7A7A7A',
}

steps = np.arange(1, 501)

# Left panel: approximation quality
for opt in ['SGD', 'Adam', 'EF', 'iEF']:
    gammas = np.array(results[opt]['gammas'], dtype=float)
    valid = np.isfinite(gammas)

    if opt == 'Adam':
        lw, alpha, z = 2.6, 0.95, 4
    elif opt == 'iEF':
        lw, alpha, z = 2.2, 0.92, 3
    elif opt == 'EF':
        lw, alpha, z = 1.0, 0.25, 1
    else:  # SGD
        lw, alpha, z = 2.0, 0.88, 2

    ax1.plot(
        steps[valid],
        gammas[valid],
        color=colors[opt],
        lw=lw,
        alpha=alpha,
        zorder=z
    )

ax1.set_xlabel('Training step')
ax1.set_ylabel(r'$\gamma(\Delta \theta)$')
ax1.set_yscale('log')
ax1.set_ylim(1e-3, 1e3)
ax1.grid(True, which='major', alpha=0.16)
ax1.grid(False, which='minor')
ax1.tick_params(direction='out')
ax1.set_title('Approximation quality to NGD', pad=10, fontsize=11.5, fontweight='semibold')

# Right panel: training loss
for opt in ['SGD', 'Adam', 'EF', 'iEF', 'NGD']:
    losses = np.array(results[opt]['losses'], dtype=float)

    if opt == 'Adam':
        lw, alpha, z = 3.0, 1.0, 5
    elif opt == 'iEF':
        lw, alpha, z = 2.2, 0.92, 3
    elif opt == 'EF':
        lw, alpha, z = 2.0, 0.82, 4
    elif opt == 'NGD':
        lw, alpha, z = 1.9, 0.45, 1
    else:  # SGD
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
ax2.set_ylabel('MSE loss')
ax2.set_yscale('log')
ax2.set_ylim(1e-12, 1e12)
ax2.grid(True, which='major', alpha=0.16)
ax2.grid(False, which='minor')
ax2.tick_params(direction='out')
ax2.set_title('Training loss', pad=10, fontsize=11.5, fontweight='semibold')

legend_handles = [
    Line2D([0], [0], color=colors['SGD'],  lw=2.0, alpha=0.88, label='SGD'),
    Line2D([0], [0], color=colors['Adam'], lw=3.0, alpha=1.0,  label='Adam'),
    Line2D([0], [0], color=colors['EF'],   lw=2.0, alpha=0.82, label='EF'),
    Line2D([0], [0], color=colors['iEF'],  lw=2.2, alpha=0.92, label='iEF'),
    Line2D([0], [0], color=colors['NGD'],  lw=1.9, alpha=0.45, label='NGD'),
]

fig.legend(
    handles=legend_handles,
    loc='upper center',
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, 0.955),
    handlelength=2.3,
    columnspacing=1.6
)

fig.suptitle(
    'Small neural network',
    y=1.00,
    fontsize=12,
    fontweight='semibold'
)

fig.subplots_adjust(
    top=0.84,
    bottom=0.14,
    left=0.08,
    right=0.98,
    wspace=0.24
)

plt.savefig('../figures/exp4_neural_network.pdf')
plt.savefig('../figures/exp4_neural_network.png', dpi=300)
plt.show()
print("Figure saved.")