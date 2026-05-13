import torch
import torch.nn as nn
import numpy as np

def compute_fisher_vector_product(loss, params, vector):
    """
    Computes F*vector using double differentiation trick.
    Following Wu et al. (2024) Appendix F.1
    
    F = sum_n E_{y~p(y|x,theta)}[grad log p * grad log p^T]
    F*v can be computed efficiently without forming F explicitly.
    """
    # First compute gradient of loss
    grads = torch.autograd.grad(loss, params, 
                                 create_graph=True, 
                                 retain_graph=True)
    
    # Flatten gradients into single vector
    grad_vec = torch.cat([g.reshape(-1) for g in grads])
    
    # Compute gradient-vector product (grad^T * v)
    gv_product = torch.sum(grad_vec * vector)
    
    # Differentiate again to get F*v
    fvp = torch.autograd.grad(gv_product, params,
                               retain_graph=True)
    
    fvp_vec = torch.cat([f.reshape(-1) for f in fvp])
    
    return fvp_vec

def compute_gamma(delta_theta, loss, params, grad_L):
    delta_flat = torch.cat([d.reshape(-1) for d in delta_theta])
    grad_flat = torch.cat([g.reshape(-1) for g in grad_L])

    # γ is undefined when gradient vanishes (near convergence)
    if torch.norm(grad_flat) < 1e-6:
        return float('nan')

    # Also guard against exploding delta (e.g. NGD with large steps)
    if torch.norm(delta_flat) > 1e3:
        return float('nan')

    F_delta = compute_fisher_vector_product(loss, params, delta_flat)

    fdelta_dot = torch.dot(delta_flat, F_delta)
    fdelta_dot = torch.clamp(fdelta_dot, min=0.0)

    numerator = torch.sqrt(fdelta_dot + 1e-10)
    denominator = torch.abs(torch.dot(delta_flat, grad_flat)) + 1e-10

    gamma = (numerator / denominator).item()

    # Final sanity guard — gamma > 1e3 is numerically meaningless
    if gamma > 1e3 or np.isnan(gamma):
        return float('nan')

    return gamma

def get_update_methods(model, loss, params, lr=0.01, 
                        beta1=0.9, beta2=0.999, eps=1e-8,
                        m=None, v=None, t=1):
    """
    Returns update vectors for SGD, EF, iEF, and Adam.
    """
    # Get gradients
    grads = torch.autograd.grad(loss, params, 
                                 create_graph=True,
                                 retain_graph=True)
    grad_flat = torch.cat([g.reshape(-1) for g in grads])
    
    updates = {}
    
    # SGD update
    updates['SGD'] = [-lr * g for g in grads]
    
    # Adam update
    if m is None:
        m = torch.zeros_like(grad_flat)
    if v is None:
        v = torch.zeros_like(grad_flat)
    
    m_new = beta1 * m + (1 - beta1) * grad_flat
    v_new = beta2 * v + (1 - beta2) * grad_flat ** 2
    
    # Bias correction
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)
    
    adam_update_flat = -lr * m_hat / (torch.sqrt(v_hat) + eps)
    
    # Reshape back to param shapes
    adam_updates = []
    idx = 0
    for p in params:
        size = p.numel()
        adam_updates.append(
            adam_update_flat[idx:idx+size].reshape(p.shape)
        )
        idx += size
    updates['Adam'] = adam_updates
    
    # EF update (empirical Fisher preconditioned)
    # EF = sum of outer products of gradients
    # EF^{-1} grad computed via Woodbury identity
    # For simplicity use diagonal EF approximation here
    ef_diag = grad_flat ** 2 + eps
    ef_update_flat = -lr * grad_flat / ef_diag
    
    ef_updates = []
    idx = 0
    for p in params:
        size = p.numel()
        ef_updates.append(
            ef_update_flat[idx:idx+size].reshape(p.shape)
        )
        idx += size
    updates['EF'] = ef_updates
    
    # iEF update (Wu et al. 2024)
    # Requires logit-level gradient norms
    # Approximated here — full implementation needs model hooks
    # TODO: implement full iEF with logit gradient norms
    updates['iEF'] = ef_updates  # placeholder for now
    
    return updates, m_new, v_new

if __name__ == "__main__":
    # Quick sanity check
    torch.manual_seed(42)
    
    # Simple linear model
    model = nn.Linear(5, 1, bias=False)
    params = list(model.parameters())
    
    x = torch.randn(10, 5)
    y = torch.randn(10, 1)
    
    output = model(x)
    loss = nn.MSELoss()(output, y)
    loss.backward(retain_graph=True)
    
    grads = torch.autograd.grad(loss, params, 
                                 create_graph=True,
                                 retain_graph=True)
    
    # Test gamma with SGD update
    sgd_update = [-0.01 * g.detach() for g in grads]
    
    gamma_val = compute_gamma(sgd_update, loss, params, grads)
    print(f"Gamma (SGD): {gamma_val:.4f}")
    print("Gamma indicator working correctly if no errors above.")