# src/alpha_mle.py
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln


def cdm_loglik_alpha(alpha: np.ndarray, counts: np.ndarray) -> float:
    """
    Dirichlet-Multinomial (CDM) log-likelihood for aggregated counts.
    """
    alpha = np.asarray(alpha, dtype=float)
    counts = np.asarray(counts, dtype=float)

    if np.any(alpha <= 0):
        return -np.inf

    A = float(alpha.sum())
    N = float(counts.sum())

    return (
        gammaln(A)
        - gammaln(A + N)
        + float(np.sum(gammaln(alpha + counts) - gammaln(alpha)))
    )


def fit_alpha_mle(
    counts: np.ndarray,
    beta_init: np.ndarray | None = None,
    l2: float = 0.0,
    maxiter: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit alpha via MLE with beta parameterization: alpha = exp(beta).
    Optional L2 penalty on beta.
    Returns (alpha_hat, beta_hat).
    """
    counts = np.asarray(counts, dtype=float)
    k = int(counts.shape[0])

    if beta_init is None:
        beta_init = np.zeros(k, dtype=float)  # alpha = 1.0
    else:
        beta_init = np.asarray(beta_init, dtype=float)
        if beta_init.shape[0] != k:
            beta_init = np.zeros(k, dtype=float)

    def objective(beta: np.ndarray) -> float:
        alpha = np.exp(beta)
        ll = cdm_loglik_alpha(alpha, counts)
        nll = -ll
        if l2 > 0:
            nll = nll + 0.5 * l2 * float(np.dot(beta, beta))
        return float(nll)

    res = minimize(
        objective,
        beta_init,
        method="L-BFGS-B",
        options={"maxiter": maxiter},
    )

    beta_hat = res.x.astype(float)
    alpha_hat = np.exp(beta_hat)
    return alpha_hat, beta_hat
