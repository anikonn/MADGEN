import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from typing import Optional, Tuple
import torch
from torch.nn import functional as F
import numpy as np
import math

from src.data.utils import PlaceHolder

# ---------------------------
# Determinism plumbing
# ---------------------------
# You set this once from your main script:
#   from src.frameworks import diffusion_utils
#   gen = torch.Generator(device=torch_device); gen.manual_seed(seed)
#   diffusion_utils.set_sampling_generator(gen)
# All multinomial/randn calls below will use it.
_GEN: Optional[torch.Generator] = None

def set_sampling_generator(gen: torch.Generator):
    global _GEN
    _GEN = gen

def _require_gen(device):
    if _GEN is None:
        # Fall back to default (non-deterministic). Prefer to set via set_sampling_generator().
        return torch.default_generator
    # Torch requires the generator device to match the target device for CUDA ops.
    # If you created it on the correct device in main (recommended), this is fine.
    return _GEN

# ---------------------------
# Small helpers
# ---------------------------

def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)

def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'

def sample_gaussian(size, *, device, generator):
    return torch.randn(size, device=device, generator=generator)

def sample_gaussian_with_mask(size, node_mask, *, device, generator):
    x = torch.randn(size, device=device, generator=generator)
    return (x * node_mask.to(x.dtype))

def clip_noise_schedule(alphas2, clip_value=0.001):
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
    alphas_step = (alphas2[1:] / alphas2[:-1])
    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)
    return alphas2

def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)
    return alphas_cumprod

def cosine_beta_schedule_discrete(timesteps, s=0.008):
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()

def polynomial_beta_schedule_discrete(timesteps, s=0.008):
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas = (1 - 2 * s) * (1 - (x / steps) ** 2)[1:]
    betas = 1 - alphas
    return betas.squeeze()

def linear_beta_schedule_discrete(timesteps, s=0.008):
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas = (1 - 2 * s) * (1 - (x / steps))[1:]
    betas = 1 - alphas
    return betas.squeeze()

def custom_beta_schedule_discrete(timesteps, average_num_nodes=50, s=0.008):
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    assert timesteps >= 100
    p = 4 / 5
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)
    betas[betas < beta_first] = beta_first
    return np.array(betas)

def gaussian_KL(q_mu, q_sigma):
    return sum_except_batch((torch.log(1 / q_sigma) + 0.5 * (q_sigma ** 2 + q_mu ** 2) - 0.5))

def cdf_std_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))

def SNR(gamma):
    return torch.exp(-gamma)

def inflate_batch_array(array, target_shape):
    target_shape = (array.size(0),) + (1,) * (len(target_shape) - 1)
    return array.view(target_shape)

def sigma(gamma, target_shape):
    return inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_shape)

def alpha(gamma, target_shape):
    return inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_shape)

def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)

def check_tensor_same_size(*args):
    for i, arg in enumerate(args):
        if i == 0:
            continue
        assert args[0].size() == arg.size()

def sigma_and_alpha_t_given_s(gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_size: torch.Size):
    sigma2_t_given_s = inflate_batch_array(-torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)), target_size)
    log_alpha2_t = F.logsigmoid(-gamma_t)
    log_alpha2_s = F.logsigmoid(-gamma_s)
    log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s
    alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
    alpha_t_given_s = inflate_batch_array(alpha_t_given_s, target_size)
    sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
    return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]

# ---------------------------
# RNG-using functions (deterministic)
# ---------------------------

def sample_feature_noise(X_size, E_size, y_size, node_mask, *, device, generator=None):
    """Standard normal noise for all features, symmetric for edges."""
    gen = generator or _require_gen(device)
    epsX = sample_gaussian(X_size, device=device, generator=gen)
    epsE = sample_gaussian(E_size, device=device, generator=gen)
    epsy = sample_gaussian(y_size, device=device, generator=gen)

    float_mask = node_mask.to(epsX.dtype)
    epsX = epsX * float_mask
    epsy = epsy * float_mask

    # Mask edges by node presence
    epsE = epsE * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)

    # Upper triangular only (then mirror)
    upper_triangular_mask = torch.zeros_like(epsE, device=device)
    i, j = torch.triu_indices(row=epsE.size(1), col=epsE.size(2), offset=1, device=device)
    upper_triangular_mask[:, i, j, :] = 1
    epsE = epsE * upper_triangular_mask
    epsE = epsE + epsE.transpose(1, 2)

    assert (epsE == epsE.transpose(1, 2)).all()
    return PlaceHolder(X=epsX, E=epsE, y=epsy).mask(node_mask)

def sample_normal(mu_X, mu_E, mu_y, sigma, node_mask, *, device, generator=None):
    gen = generator or _require_gen(device)
    eps = sample_feature_noise(mu_X.size(), mu_E.size(), mu_y.size(), node_mask,
                               device=device, generator=gen).type_as(mu_X)
    X = mu_X + sigma * eps.X
    E = mu_E + sigma.unsqueeze(1) * eps.E
    y = mu_y + sigma.squeeze(1) * eps.y
    return PlaceHolder(X=X, E=E, y=y)

def sample_discrete_features(probX, probE, node_mask, *, generator=None):
    """
    Sample nodes/edges with Multinomial using a fixed generator for determinism.
    probX: (bs, n, dx_out)
    probE: (bs, n, n, de_out)
    """
    device = probX.device
    gen = generator or _require_gen(device)

    bs, n, _ = probX.shape
    probX = probX.clone()

    # masked nodes -> uniform
    probX[~node_mask] = 1.0 / probX.shape[-1]
    probX_flat = probX.reshape(bs * n, -1)

    X_t = probX_flat.multinomial(1, generator=gen).reshape(bs, n)
    X_t_probs = torch.gather(probX_flat, 1, X_t.reshape(-1, 1)).reshape(bs, n)

    inverse_edge_mask = ~(node_mask.unsqueeze(1) & node_mask.unsqueeze(2))
    diag_mask = torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0).expand(bs, -1, -1)

    probE = probE.clone()
    probE[inverse_edge_mask] = 1.0 / probE.shape[-1]
    probE[diag_mask] = 1.0 / probE.shape[-1]

    probE_flat = probE.reshape(bs * n * n, -1)
    E_t = probE_flat.multinomial(1, generator=gen).reshape(bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = E_t + E_t.transpose(1, 2)

    E_t_probs = torch.gather(probE_flat, 1, E_t.reshape(-1, 1)).reshape(bs, n, n)

    total_probs = X_t_probs.mean(dim=1) + E_t_probs.mean(dim=[1, 2])

    return PlaceHolder(
        X=X_t,
        E=E_t,
        y=torch.zeros(bs, 0, device=device, dtype=X_t.dtype)
    ), total_probs

# ---------------------------
# (the rest is unchanged / deterministic math)
# ---------------------------

def compute_posterior_distribution(M, M_t, Qt_M, Qsb_M, Qtb_M):
    M = M.flatten(start_dim=1, end_dim=-2).to(torch.float32)
    M_t = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)
    Qt_M_T = torch.transpose(Qt_M, -2, -1)
    left_term = M_t @ Qt_M_T
    right_term = M @ Qsb_M
    product = left_term * right_term
    denom = M @ Qtb_M
    denom = (denom * M_t).sum(dim=-1)
    prob = product / denom.unsqueeze(-1)
    return prob

def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)
    Qt_T = Qt.transpose(-1, -2)
    left_term = X_t @ Qt_T
    left_term = left_term.unsqueeze(dim=2)
    right_term = Qsb.unsqueeze(1)
    numerator = left_term * right_term
    X_t_transposed = X_t.transpose(-1, -2)
    prod = Qtb @ X_t_transposed
    prod = prod.transpose(-1, -2)
    denominator = prod.unsqueeze(-1)
    denominator[denominator == 0] = 1e-6
    out = numerator / denominator
    return out

def mask_distributions(true_X, true_E, pred_X, pred_E, node_mask):
    pred_X = pred_X + 1e-7
    pred_X = pred_X / torch.sum(pred_X, dim=-1, keepdim=True)
    pred_E = pred_E + 1e-7
    pred_E = pred_E / torch.sum(pred_E, dim=-1, keepdim=True)

    row_X = torch.zeros(true_X.size(-1), dtype=torch.float, device=true_X.device); row_X[0] = 1.
    row_E = torch.zeros(true_E.size(-1), dtype=torch.float, device=true_E.device); row_E[0] = 1.

    diag_mask = ~torch.eye(node_mask.size(1), device=node_mask.device, dtype=torch.bool).unsqueeze(0)
    true_X[~node_mask] = row_X
    true_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E
    pred_X[~node_mask] = row_X
    pred_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E
    return true_X, true_E, pred_X, pred_E

def posterior_distributions(X, E, y, X_t, E_t, y_t, Qt, Qsb, Qtb):
    prob_X = compute_posterior_distribution(M=X, M_t=X_t, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X)
    prob_E = compute_posterior_distribution(M=E, M_t=E_t, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E)
    return PlaceHolder(X=prob_X, E=prob_E, y=y_t)

def sample_discrete_feature_noise(limit_dist, node_mask, *, generator=None):
    device = node_mask.device
    gen = generator or _require_gen(device)

    bs, n_max = node_mask.shape
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1).to(device)
    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1).to(device)
    y_limit = limit_dist.y[None, :].expand(bs, -1).to(device)

    U_X = x_limit.flatten(end_dim=-2).multinomial(1, generator=gen).reshape(bs, n_max)
    U_E = e_limit.flatten(end_dim=-2).multinomial(1, generator=gen).reshape(bs, n_max, n_max)
    U_y = torch.empty((bs, 0), device=device, dtype=U_X.dtype)

    U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).to(x_limit.dtype)
    U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).to(e_limit.dtype)

    upper_triangular_mask = torch.zeros_like(U_E, device=device)
    i, j = torch.triu_indices(U_E.size(1), U_E.size(2), offset=1, device=device)
    upper_triangular_mask[:, i, j, :] = 1
    U_E = U_E * upper_triangular_mask
    U_E = U_E + U_E.transpose(1, 2)

    assert (U_E == U_E.transpose(1, 2)).all()
    return PlaceHolder(X=U_X, E=U_E, y=U_y).mask(node_mask)

def cbo0pdi_X(X_t, Qt, Qsb, Qtb):
    Qt_T = Qt.transpose(-1, -2)
    left_term = X_t.unsqueeze(-2) @ Qt_T
    numerator = left_term * Qsb
    denominator = Qtb @ X_t.unsqueeze(-1)
    denominator[denominator == 0] = 1e-6
    out = numerator / denominator
    return out

def cbo0pdi_E(E_t, Qt, Qsb, Qtb):
    E_t = E_t.flatten(start_dim=1, end_dim=2).to(torch.float32)
    Qt = Qt.flatten(start_dim=1, end_dim=2).to(torch.float32)
    Qsb = Qsb.flatten(start_dim=1, end_dim=2).to(torch.float32)
    Qtb = Qtb.flatten(start_dim=1, end_dim=2).to(torch.float32)
    Qt_T = Qt.transpose(-1, -2)
    left_term = E_t.unsqueeze(-2) @ Qt_T
    numerator = left_term * Qsb
    denominator = Qtb @ E_t.unsqueeze(-1)
    denominator[denominator == 0] = 1e-6
    out = numerator / denominator
    return out