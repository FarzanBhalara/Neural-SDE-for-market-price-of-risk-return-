import numpy as np


def marginal_sigma_from_components(beta_panel, factor_sigma, idio_sigma, eps=1e-8):
    beta = np.asarray(beta_panel, dtype=np.float32)
    factor_sigma = np.asarray(factor_sigma, dtype=np.float32).reshape(-1, 1)
    idio_sigma = np.asarray(idio_sigma, dtype=np.float32)
    total_var = np.square(beta) * np.square(factor_sigma) + np.square(idio_sigma)
    return np.sqrt(np.clip(total_var, eps, None)).astype(np.float32)


def one_factor_covariance_matrix(beta_vec, factor_var, idio_var_vec, eps=1e-8):
    beta = np.asarray(beta_vec, dtype=float).reshape(-1, 1)
    d = np.diag(np.clip(np.asarray(idio_var_vec, dtype=float), eps, None))
    return d + float(max(factor_var, eps)) * (beta @ beta.T)


def one_factor_gaussian_nll(next_returns, mu_panel, beta_panel, factor_sigma, idio_sigma, valid_mask, eps=1e-8):
    returns = np.asarray(next_returns, dtype=float)
    mu = np.asarray(mu_panel, dtype=float)
    beta = np.asarray(beta_panel, dtype=float)
    factor_var = np.square(np.asarray(factor_sigma, dtype=float).reshape(-1))
    idio_var = np.square(np.asarray(idio_sigma, dtype=float))
    valid = np.asarray(valid_mask, dtype=bool)

    nll_rows = []
    for t in range(returns.shape[0]):
        row_mask = valid[t] & np.isfinite(returns[t]) & np.isfinite(mu[t]) & np.isfinite(beta[t]) & np.isfinite(idio_var[t])
        if row_mask.sum() < 2:
            continue
        x = returns[t, row_mask] - mu[t, row_mask]
        b = beta[t, row_mask]
        d = np.clip(idio_var[t, row_mask], eps, None)
        omega = max(factor_var[t], eps)

        inv_d = 1.0 / d
        b_inv_b = np.sum(b * b * inv_d)
        logdet = np.sum(np.log(d)) + np.log1p(omega * b_inv_b)
        x_inv_x = np.sum(x * x * inv_d)
        b_inv_x = np.sum(b * x * inv_d)
        quad = x_inv_x - (omega * b_inv_x * b_inv_x) / (1.0 + omega * b_inv_b)
        n = row_mask.sum()
        nll = 0.5 * (n * np.log(2.0 * np.pi) + logdet + quad)
        nll_rows.append(float(nll))

    if not nll_rows:
        return np.nan
    return float(np.mean(nll_rows))


def diagonal_gaussian_nll(next_returns, mu_panel, sigma_panel, valid_mask, eps=1e-8):
    returns = np.asarray(next_returns, dtype=float)
    mu = np.asarray(mu_panel, dtype=float)
    sigma = np.asarray(sigma_panel, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)

    nll_rows = []
    for t in range(returns.shape[0]):
        row_mask = valid[t] & np.isfinite(returns[t]) & np.isfinite(mu[t]) & np.isfinite(sigma[t]) & (sigma[t] > 0)
        if row_mask.sum() < 1:
            continue
        x = returns[t, row_mask] - mu[t, row_mask]
        var = np.clip(np.square(sigma[t, row_mask]), eps, None)
        n = row_mask.sum()
        nll = 0.5 * (n * np.log(2.0 * np.pi) + np.sum(np.log(var)) + np.sum(np.square(x) / var))
        nll_rows.append(float(nll))

    if not nll_rows:
        return np.nan
    return float(np.mean(nll_rows))
