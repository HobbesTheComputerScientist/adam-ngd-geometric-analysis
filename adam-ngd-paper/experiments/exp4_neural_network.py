import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
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


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
colors = {'SGD': 'blue', 'Adam': 'red', 'EF': 'orange', 'iEF': 'purple', 'NGD': 'green'}
steps = range(1, 501)


for opt, data in results.items():
    if opt == 'NGD':
        continue
    gammas = data['gammas']
    valid = [(s, g) for s, g in zip(steps, gammas) if not np.isnan(g)]
    if valid:
        s_plot, g_plot = zip(*valid)
        ax1.plot(s_plot, g_plot, label=opt, color=colors[opt], alpha=0.8)


ax1.set_xlabel('Training Step')
ax1.set_ylabel('γ(Δθ)')
ax1.set_title('γ(Δθ): Step Alignment with Natural Gradient\n'
              'Small Neural Network')
ax1.legend()
ax1.set_yscale('log')


for opt, data in results.items():
    ax2.plot(steps, data['losses'], label=opt, color=colors[opt], alpha=0.8)


ax2.set_xlabel('Training Step')
ax2.set_ylabel('MSE Loss')
ax2.set_title('Training Loss\n'
              'Small Neural Network')
ax2.legend()
ax2.set_yscale('log')


plt.tight_layout()
plt.savefig('../figures/exp4_neural_network.pdf', bbox_inches='tight')
plt.show()
print("Figure saved.")