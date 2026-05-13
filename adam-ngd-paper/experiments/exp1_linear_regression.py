import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gamma_indicator import compute_gamma, compute_fisher_vector_product

torch.manual_seed(42)
np.random.seed(42)

N = 100
D = 10

X = torch.randn(N, D)
true_weights = torch.randn(D, 1)
y = X @ true_weights + 0.1 * torch.randn(N, 1)

class LinearModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=False)
    def forward(self, x):
        return self.fc(x)

def train_and_measure(optimizer_name, n_steps=2000, lr=0.01):  # ← 2000 not 2001
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
            update_flat = -lr * m_hat / (torch.sqrt(v_hat) + eps)

        elif optimizer_name == 'EF':
            ef_diag = grad_flat.detach()**2 + eps
            update_flat = -lr * grad_flat.detach() / ef_diag

        elif optimizer_name == 'NGD':
            damping = 1e-3
            fisher_diag = (X ** 2).mean(dim=0)
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

        if step % 50 == 0:
            print(f"{optimizer_name} | Step {step} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Gamma: {gamma_val:.4f}")

    return gammas, losses

print("=" * 50)
print("Experiment 1: Well-Conditioned Linear Regression")
print("=" * 50)

results = {}
for opt in ['SGD', 'Adam', 'EF', 'NGD']:
    lr = 0.1 if opt == 'NGD' else 0.01
    print(f"\nRunning {opt}...")
    gammas, losses = train_and_measure(opt, lr=lr)
    results[opt] = {'gammas': gammas, 'losses': losses}

os.makedirs('../results', exist_ok=True)
os.makedirs('../figures', exist_ok=True)
np.save('../results/exp1_results.npy', results)
print("\nResults saved.")

ngd_losses = results['NGD']['losses']
ngd_converge_step = next(
    (i+1 for i, l in enumerate(ngd_losses) if l < 0.05), None
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
colors = {'SGD': 'blue', 'Adam': 'red', 'EF': 'orange', 'NGD': 'green'}
steps = range(1, 2001)  # ← exactly 2000 values to match n_steps=2000

for opt, data in results.items():
    if opt == 'NGD':
        continue
    gammas = data['gammas']
    valid = [(s, g) for s, g in zip(steps, gammas) if not np.isnan(g)]
    if valid:
        s_plot, g_plot = zip(*valid)
        ax1.plot(s_plot, g_plot, label=opt, color=colors[opt], alpha=0.8)

if ngd_converge_step:
    ax1.axvline(x=ngd_converge_step, color='green', linestyle='--',
                alpha=0.7, linewidth=1.5,
                label=f'NGD converged (step {ngd_converge_step})')

ax1.set_xlabel('Training Step')
ax1.set_ylabel('γ(Δθ)')
ax1.set_title('γ(Δθ): Step Alignment with Natural Gradient\n'
              'Well-Conditioned Linear Regression')
ax1.legend()
ax1.set_yscale('log')

for opt, data in results.items():
    ax2.plot(steps, data['losses'], label=opt, color=colors[opt], alpha=0.8)

ax2.set_xlabel('Training Step')
ax2.set_ylabel('MSE Loss')
ax2.set_title('Training Loss\n'
              'Well-Conditioned Linear Regression')
ax2.legend()
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('../figures/exp1_linear_regression.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved.")