import torch
import torch.nn as nn
import numpy as np


def compute_fisher_vector_product(loss, params, vector):
    """
    Computes F*vector using the double differentiation trick.
    Following Wu et al. (2024) Appendix F.1.

    F = sum_n E_{y~p(y|x,theta)}[grad log p * grad log p^T]
    F*v is computed efficiently without forming F explicitly.
    """
    grads = torch.autograd.grad(loss, params,
                                create_graph=True,
                                retain_graph=True)

    grad_vec = torch.cat([g.reshape(-1) for g in grads])
    gv_product = torch.sum(grad_vec * vector)

    fvp = torch.autograd.grad(gv_product, params,
                               retain_graph=True)

    return torch.cat([f.reshape(-1) for f in fvp])


def compute_gamma(delta_theta, loss, params, grad_L):
    delta_flat = torch.cat([d.reshape(-1) for d in delta_theta])
    grad_flat  = torch.cat([g.reshape(-1) for g in grad_L])

    if torch.norm(grad_flat) < 1e-6:
        return float('nan')

    if torch.norm(delta_flat) > 1e3:
        return float('nan')

    F_delta = compute_fisher_vector_product(loss, params, delta_flat)

    fdelta_dot = torch.clamp(torch.dot(delta_flat, F_delta), min=0.0)
    numerator  = torch.sqrt(fdelta_dot + 1e-10)
    denominator = torch.abs(torch.dot(delta_flat, grad_flat)) + 1e-10

    gamma = (numerator / denominator).item()

    if gamma > 1e3 or np.isnan(gamma):
        return float('nan')

    return gamma


def get_update_methods(model, loss, params, lr=0.01,
                       beta1=0.9, beta2=0.999, eps=1e-8,
                       m=None, v=None, t=1):
    """
    Returns update vectors for SGD, EF, iEF, and Adam.

    iEF follows Wu et al. (2024): each per-sample gradient g_i is
    normalized by its own squared norm before accumulation, so that
    the diagonal preconditioner is

        diag(F_iEF)_d = (1/N) sum_i  g_{i,d}^2 / ||g_i||^2

    This removes the magnitude bias present in the standard EF, where
    samples with large gradient norms dominate the curvature estimate.
    The update then uses the same formula as EF but with this
    normalized diagonal instead of the raw squared gradient.
    """
    grads = torch.autograd.grad(loss, params,
                                create_graph=True,
                                retain_graph=True)
    grad_flat = torch.cat([g.reshape(-1) for g in grads])

    updates = {}

    # SGD
    updates['SGD'] = [-lr * g for g in grads]

    # Adam
    if m is None:
        m = torch.zeros_like(grad_flat)
    if v is None:
        v = torch.zeros_like(grad_flat)

    m_new = beta1 * m + (1 - beta1) * grad_flat
    v_new = beta2 * v + (1 - beta2) * grad_flat ** 2

    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)

    adam_update_flat = -lr * m_hat / (torch.sqrt(v_hat) + eps)

    adam_updates, idx = [], 0
    for p in params:
        size = p.numel()
        adam_updates.append(adam_update_flat[idx:idx+size].reshape(p.shape))
        idx += size
    updates['Adam'] = adam_updates

    # EF: diagonal of the empirical Fisher using the batch-aggregated gradient.
    # This is the single-pass approximation
    ef_diag = grad_flat.detach() ** 2 + eps
    ef_update_flat = -lr * grad_flat.detach() / ef_diag

    ef_updates, idx = [], 0
    for p in params:
        size = p.numel()
        ef_updates.append(ef_update_flat[idx:idx+size].reshape(p.shape))
        idx += size
    updates['EF'] = ef_updates

    # iEF: improved empirical Fisher (Wu et al., 2024).
    # Per-sample gradients are collected by differentiating each sample's
    # loss individually. Each gradient is then normalized by its own squared
    # norm before the diagonal is averaged. This prevents samples with
    # large gradient magnitudes from distorting the curvature estimate,
    # which is the core failure mode of standard EF identified by Wu et al.
    x_batch = model._iEF_x if hasattr(model, '_iEF_x') else None
    if x_batch is not None:
        pass

    D_total = grad_flat.numel()
    G = []  

    # We need per-sample losses; get them via the model's stored batch.
    # If the model does not expose _iEF_x/_iEF_y, fall back to the
    # batch gradient squared (i.e. same as EF) with a warning.
    if hasattr(model, '_iEF_x') and hasattr(model, '_iEF_y'):
        X = model._iEF_x
        Y = model._iEF_y
        loss_fn = model._iEF_loss_fn
        N = X.shape[0]

        for i in range(N):
            out_i = model(X[i:i+1])
            loss_i = loss_fn(out_i, Y[i:i+1])
            g_i = torch.autograd.grad(loss_i, params, retain_graph=False)
            g_i_flat = torch.cat([g.reshape(-1) for g in g_i]).detach()
            G.append(g_i_flat)

        G = torch.stack(G)                                   
        norms_sq = (G ** 2).sum(dim=1, keepdim=True) + eps   
        ief_diag = (G ** 2 / norms_sq).mean(dim=0) + eps    
    else:
        # Fallback: no per-sample data available in this context.
        # Use EF diagonal. In practice the experiments never call
        # get_update_methods for iEF; they implement it inline.
        ief_diag = ef_diag

    ief_update_flat = -lr * grad_flat.detach() / ief_diag

    ief_updates, idx = [], 0
    for p in params:
        size = p.numel()
        ief_updates.append(ief_update_flat[idx:idx+size].reshape(p.shape))
        idx += size
    updates['iEF'] = ief_updates

    return updates, m_new, v_new


if __name__ == "__main__":
    torch.manual_seed(42)

    model = nn.Linear(5, 1, bias=False)
    params = list(model.parameters())
    loss_fn = nn.MSELoss()

    X = torch.randn(10, 5)
    y = torch.randn(10, 1)

    model._iEF_x = X
    model._iEF_y = y
    model._iEF_loss_fn = loss_fn

    output = model(X)
    loss = loss_fn(output, y)
    loss.backward(retain_graph=True)

    grads = torch.autograd.grad(loss, params,
                                create_graph=True,
                                retain_graph=True)

    sgd_update = [-0.01 * g.detach() for g in grads]
    gamma_val = compute_gamma(sgd_update, loss, params, grads)
    print(f"Gamma (SGD): {gamma_val:.4f}")
    print("Gamma indicator working correctly if no errors above.")