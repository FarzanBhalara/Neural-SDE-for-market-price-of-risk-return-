import torch

def euler_maruyama_step(r_t, t, drift_net, diffusion_net, dt):
    mu = drift_net(t, r_t)
    sigma = diffusion_net(t, r_t)
    eps = torch.randn_like(r_t)
    return r_t + mu * dt + sigma * torch.sqrt(torch.tensor(dt)) * eps



