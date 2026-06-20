import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
import torchvision
import torchvision.transforms as transforms
from matplotlib.lines import Line2D

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


# ── Config ───────────────────────────────────────────────────────────────────
CIFAR10_MEAN      = (0.4914, 0.4822, 0.4465)
CIFAR10_STD       = (0.2023, 0.1994, 0.2010)
DATA_ROOT         = './data'
N_TRAIN_PER_CLASS = 250
N_TEST_PER_CLASS  = 100
BINARY_CLASSES    = [0, 1]   # 0=airplane, 1=automobile

N_STEPS           = 300
SEEDS             = [0, 1, 2, 3, 4]     
MC_SAMPLES        = 5                    
D_PARAM_CEILING   = 2000                 

OPTIMIZERS = ['SGD', 'Adam', 'EF', 'iEF', 'NGD']


# ── Data ──────────────
def load_binary_cifar10(seed=42):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_full = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform)
    test_full = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform)

    rng = np.random.default_rng(seed)

    def extract(dataset, n_per_class):
        xs, ys = [], []
        for new_label, orig in enumerate(BINARY_CLASSES):
            idxs = [i for i, (_, y) in enumerate(dataset) if y == orig]
            chosen = rng.choice(idxs, size=n_per_class, replace=False).tolist()
            for i in chosen:
                x, _ = dataset[i]
                xs.append(x)
                ys.append(new_label)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    X_train, y_train = extract(train_full, N_TRAIN_PER_CLASS)
    X_test,  y_test  = extract(test_full,  N_TEST_PER_CLASS)

    perm = torch.randperm(len(X_train), generator=torch.Generator().manual_seed(seed))
    return X_train[perm], y_train[perm], X_test, y_test


X_train, y_train, X_test, y_test = load_binary_cifar10(seed=42)
N = len(X_train)


# ── Model ──────────────────────────────────────────────────────────────────────
class SmallCNN(nn.Module):
    """
    Total trainable parameters: 380 (no bias terms anywhere).

    conv1: 3->4 channels, 3x3, padding=1   -> 3*4*3*3   = 108
    conv2: 4->4 channels, 3x3, padding=1   -> 4*4*3*3   = 144
    fc:    64 -> 2                          -> 64*2      = 128
    total: 108 + 144 + 128 = 380
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(4, stride=4)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.fc    = nn.Linear(64, 2, bias=False)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        return self.fc(x.reshape(x.size(0), -1))


# ── Clean per-sample gradients ──────────────────────────────────────
def per_sample_gradients(model, X, y, loss_fn, params):
    model.zero_grad(set_to_none=True)
    G = []
    for i in range(X.size(0)):
        out_i  = model(X[i:i+1])
        loss_i = loss_fn(out_i, y[i:i+1])
        g_i    = torch.autograd.grad(loss_i, params, retain_graph=False, create_graph=False)
        G.append(torch.cat([g.reshape(-1) for g in g_i]).detach())
    return torch.stack(G)  # (N, D_total)


# ── Stability pre-sweep: pick largest LR that doesn't diverge ──────────────
#Learning rates are now chosen by an explicit sweep.
LR_GRID = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
DAMPING_GRID = [0.1, 0.05, 0.02, 0.01, 0.005]  # for EF / iEF / NGD denominator stabilizer


def is_stable(losses, blowup_factor=20.0):
    """A run is 'stable' if loss never exceeds blowup_factor x its initial value
    and never becomes NaN/Inf."""
    losses = np.array(losses, dtype=float)
    if not np.all(np.isfinite(losses)):
        return False
    return np.all(losses <= blowup_factor * losses[0] + 1e-8)


def sweep_lr(optimizer_name, damping, n_probe_steps=60):
    """Try LR_GRID from largest to smallest; return the largest stable LR."""
    for lr in LR_GRID:
        torch.manual_seed(0)
        _, losses = run_optimizer(optimizer_name, n_steps=n_probe_steps,
                                   lr=lr, damping=damping, track_gamma=False)
        if is_stable(losses):
            return lr
    return LR_GRID[-1]  # fall back to smallest if nothing is stable


def sweep_damping(optimizer_name, n_probe_steps=60, probe_lr=0.01):
    """For EF/iEF/NGD: pick the largest damping (most stable, least aggressive)
    only as a tie-breaker if needed; primarily we keep damping fixed and sweep LR.
    This returns the damping value used for the main run (documented, not hardcoded
    blindly: smallest damping that remains numerically stable)."""
    for damping in sorted(DAMPING_GRID, reverse=False):  # smallest first: prefer
                                                          # least damping that's stable
        torch.manual_seed(0)
        _, losses = run_optimizer(optimizer_name, n_steps=n_probe_steps,
                                   lr=probe_lr, damping=damping, track_gamma=False)
        if is_stable(losses):
            return damping
    return DAMPING_GRID[0]


# ── Core training/measurement loop ──────────────────────────────────────────
def run_optimizer(optimizer_name, n_steps, lr, damping, track_gamma=True):
    model   = SmallCNN()
    params  = list(model.parameters())
    D_total = sum(p.numel() for p in params)
    loss_fn = nn.CrossEntropyLoss()

    m = torch.zeros(D_total)
    v = torch.zeros(D_total)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    gammas, losses = [], []

    for step in range(1, n_steps + 1):

        # Full-batch forward/backward for the actual update direction.
        model.zero_grad(set_to_none=True)
        output = model(X_train)
        loss   = loss_fn(output, y_train)
        grads  = torch.autograd.grad(loss, params, create_graph=track_gamma,
                                      retain_graph=track_gamma)
        grad_flat = torch.cat([g.reshape(-1) for g in grads])

        # ── Optimizer update rules ─────────────────────────────────────────
        if optimizer_name == 'SGD':
            update_flat = -lr * grad_flat.detach()

        elif optimizer_name == 'Adam':
            m = beta1 * m + (1 - beta1) * grad_flat.detach()
            v = beta2 * v + (1 - beta2) * grad_flat.detach() ** 2
            m_hat = m / (1 - beta1 ** step)
            v_hat = v / (1 - beta2 ** step)
            update_flat = -lr * m_hat / (torch.sqrt(v_hat) + eps)

        elif optimizer_name == 'EF':
            # Standard empirical Fisher diagonal: diag(F_EF) = (1/N) sum g_i^2
            G = per_sample_gradients(model, X_train, y_train, loss_fn, params)
            ef_diag = (G ** 2).mean(dim=0) + damping
            update_flat = -lr * grad_flat.detach() / ef_diag

        elif optimizer_name == 'iEF':
            # Improved empirical Fisher (Wu et al., 2024):
            # diag(F_iEF)_d = (1/N) sum_i  g_{i,d}^2 / ||g_i||^2
            G = per_sample_gradients(model, X_train, y_train, loss_fn, params)
            norms_sq = (G ** 2).sum(dim=1, keepdim=True) + 1e-12
            ief_diag = (G ** 2 / norms_sq).mean(dim=0) + damping
            update_flat = -lr * grad_flat.detach() / ief_diag

        elif optimizer_name == 'NGD':
            # Exact NGD: full (non-diagonal) Fisher matrix F, inverted exactly.
            # F = (1/N) sum_i E_{y~p(y|x_i,theta)}[grad log p * grad log p^T]
            # APPROXIMATION: inner expectation approximated with MC_SAMPLES draws
            # from the model's CURRENT predictive distribution (not observed labels).
            F = torch.zeros(D_total, D_total)

            for i in range(N):
                logits_i = model(X_train[i:i+1])
                with torch.no_grad():
                    probs_i = torch.softmax(logits_i, dim=-1).squeeze(0)

                sampled_ys = torch.multinomial(probs_i, num_samples=MC_SAMPLES,
                                               replacement=True)
                log_probs_i = torch.log_softmax(logits_i, dim=-1)
                for y_s in sampled_ys:
                    g_s = torch.autograd.grad(log_probs_i[0, y_s], params,
                                              retain_graph=True, create_graph=False)
                    g_s_flat = torch.cat([g.reshape(-1) for g in g_s]).detach()
                    F += torch.outer(g_s_flat, g_s_flat)

            F /= N * MC_SAMPLES
            F_damped = F + damping * torch.eye(D_total)

            delta, _, _, _ = torch.linalg.lstsq(F_damped, grad_flat.detach().unsqueeze(1))
            update_flat = -lr * delta.squeeze(1)

        # ── Reshape update into parameter-space list ───────────────────────
        update_list = []
        idx = 0
        for p in params:
            size = p.numel()
            update_list.append(update_flat[idx:idx + size].reshape(p.shape))
            idx += size

        # ── Gamma (exact full Fisher, regardless of which optimizer produced
        #    the update) ──────────────────────────────────────────────────
        if track_gamma:
            gamma_val = compute_gamma(update_list, loss, params, grads)
            gammas.append(gamma_val)
        losses.append(loss.item())

        # ── Apply update ─────────────────────────────────────────────────
        with torch.no_grad():
            for p, u in zip(params, update_list):
                p.add_(u)

        if track_gamma and step % 50 == 0:
            g_str = f"{gammas[-1]:.4f}" if track_gamma else "n/a"
            print(f"  {optimizer_name} | step {step:4d} | loss {loss.item():.4f} | gamma {g_str}")

    return gammas, losses


# ── Multi-seed runner ────────────────────────────────────────────────────────
def run_multi_seed(optimizer_name, lr, damping, n_steps, seeds):
    all_gammas, all_losses = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        gammas, losses = run_optimizer(optimizer_name, n_steps=n_steps,
                                        lr=lr, damping=damping, track_gamma=True)
        all_gammas.append(gammas)
        all_losses.append(losses)

    gammas_arr = np.array(all_gammas, dtype=float)   # (n_seeds, n_steps) or empty for NGD
    losses_arr = np.array(all_losses, dtype=float)   # (n_seeds, n_steps)

    return {
        'gamma_mean': np.nanmean(gammas_arr, axis=0) if gammas_arr.size else np.array([]),
        'gamma_std':  np.nanstd(gammas_arr, axis=0)  if gammas_arr.size else np.array([]),
        'loss_mean':  losses_arr.mean(axis=0),
        'loss_std':   losses_arr.std(axis=0),
        'gamma_all':  gammas_arr,
        'loss_all':   losses_arr,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("Experiment 5: Small CNN on Binary CIFAR-10")
    print("=" * 60)

    D_check = sum(p.numel() for p in SmallCNN().parameters())
    print(f"Model parameter count D = {D_check}  (Fisher matrix: {D_check}x{D_check})")
    assert D_check < D_PARAM_CEILING, (
        f"D={D_check} exceeds {D_PARAM_CEILING}-parameter ceiling for exact Fisher!")

    # ── Stage 1: tune LR (and damping, for EF/iEF/NGD) per optimizer ──────────
    print("\n--- Stability pre-sweep (tuning learning rates / damping) ---")
    tuned_lr = {}
    tuned_damping = {}

    for opt in OPTIMIZERS:
        if opt in ('EF', 'iEF', 'NGD'):
            damping = sweep_damping(opt)
            lr = sweep_lr(opt, damping=damping)
        else:
            damping = None
            lr = sweep_lr(opt, damping=None)
        tuned_lr[opt] = lr
        tuned_damping[opt] = damping
        print(f"  {opt:5s} -> lr = {lr:<8} damping = {damping}")

    print("\nLearning rate / damping summary (Table for paper):")
    print(f"{'Optimizer':<10}{'LR':<10}{'Damping':<10}")
    for opt in OPTIMIZERS:
        d = tuned_damping[opt] if tuned_damping[opt] is not None else '-'
        print(f"{opt:<10}{tuned_lr[opt]:<10}{d}")

    # ── Stage 2: full multi-seed runs with tuned hyperparameters ──────────────
    print(f"\n--- Full runs: {len(SEEDS)} seeds x {N_STEPS} steps per optimizer ---")
    results = {}
    for opt in OPTIMIZERS:
        print(f"\nRunning {opt} (lr={tuned_lr[opt]}, damping={tuned_damping[opt]})...")
        results[opt] = run_multi_seed(opt, lr=tuned_lr[opt], damping=tuned_damping[opt],
                                       n_steps=N_STEPS, seeds=SEEDS)

    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    np.save('results/exp5_results.npy', results, allow_pickle=True)
    np.save('results/exp5_hyperparams.npy',
            {'lr': tuned_lr, 'damping': tuned_damping}, allow_pickle=True)

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.4),
                                    gridspec_kw={'wspace': 0.24})

    colors = {
        'SGD':  '#4C78A8',
        'Adam': '#E45756',
        'EF':   '#F2B134',
        'iEF':  '#54A24B',
        'NGD':  '#7A7A7A',
    }
    style = {
        'SGD':  dict(lw=2.0, alpha=0.88, z=2),
        'Adam': dict(lw=2.8, alpha=1.00, z=5),
        'EF':   dict(lw=1.6, alpha=0.55, z=1),
        'iEF':  dict(lw=2.2, alpha=0.92, z=3),
        'NGD':  dict(lw=1.9, alpha=0.45, z=1),
    }

    steps = np.arange(1, N_STEPS + 1)

    # Left panel: gamma, mean +/- std band, NGD excluded by definition
    for opt in ['SGD', 'Adam', 'EF', 'iEF']:
        mean = results[opt]['gamma_mean']
        std  = results[opt]['gamma_std']
        valid = np.isfinite(mean)
        s = style[opt]
        ax1.plot(steps[valid], mean[valid], color=colors[opt],
                  lw=s['lw'], alpha=s['alpha'], zorder=s['z'])
        ax1.fill_between(steps[valid], (mean - std)[valid], (mean + std)[valid],
                          color=colors[opt], alpha=0.15, zorder=s['z'] - 0.5)

    ax1.set_xlabel('Training step')
    ax1.set_ylabel(r'$\gamma(\Delta \theta)$')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-3, 1e3)
    ax1.grid(True, which='major', alpha=0.16)
    ax1.grid(False, which='minor')
    ax1.tick_params(direction='out')
    ax1.set_title('Approximation quality to NGD', pad=10, fontsize=11.5, fontweight='semibold')

    # Right panel: training loss, mean +/- std band
    for opt in OPTIMIZERS:
        mean = results[opt]['loss_mean']
        std  = results[opt]['loss_std']
        s = style[opt]
        ax2.plot(steps, mean, color=colors[opt], lw=s['lw'], alpha=s['alpha'], zorder=s['z'])
        ax2.fill_between(steps, mean - std, mean + std, color=colors[opt],
                          alpha=0.15, zorder=s['z'] - 0.5)

    ax2.set_xlabel('Training step')
    ax2.set_ylabel('Cross-entropy loss')
    ax2.set_yscale('log')
    ax2.grid(True, which='major', alpha=0.16)
    ax2.grid(False, which='minor')
    ax2.tick_params(direction='out')
    ax2.set_title('Training loss', pad=10, fontsize=11.5, fontweight='semibold')

    legend_handles = [
        Line2D([0], [0], color=colors[o], lw=style[o]['lw'], alpha=style[o]['alpha'], label=o)
        for o in OPTIMIZERS
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.955), handlelength=2.3, columnspacing=1.6)

    fig.suptitle(f'Small CNN on binary CIFAR-10 (airplane vs. automobile), '
                 f'mean \u00b1 std over {len(SEEDS)} seeds',
                 y=1.00, fontsize=12, fontweight='semibold')

    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.08, right=0.98, wspace=0.24)

    plt.savefig('figures/exp5_small_cnn_cifar10.pdf')
    plt.savefig('figures/exp5_small_cnn_cifar10.png', dpi=300)
    plt.show()
    print("Figure saved.")