"""
Modul model Black-Litterman untuk optimasi portofolio.

Implementasi mengikuti formulasi standar (He & Litterman, 1999):

    Pi = delta * Sigma * w_mkt                                (prior/implied return)
    M  = [(tau * Sigma)^-1 + P' Omega^-1 P]^-1
    mu_BL = M [ (tau * Sigma)^-1 Pi + P' Omega^-1 Q ]          (posterior return)
    Sigma_BL = Sigma + M                                       (posterior covariance)

dengan:
    delta  : koefisien risk-aversion pasar
    Sigma  : matriks kovarians return (historis, tahunan)
    w_mkt  : bobot pasar awal (di sini didekati equal-weight karena
             tidak tersedia data kapitalisasi pasar pada dataset)
    tau    : skalar ketidakpastian pada prior (umumnya kecil, 0.01-0.05)
    P      : matriks "picking" yang menunjukkan saham mana yang terkait
             tiap view
    Q      : vektor besar return yang "dipandang" (di sini diisi oleh
             expected return hasil prediksi XGBoost)
    Omega  : matriks kovarians ketidakpastian view (diagonal,
             proporsional terhadap variansnya sendiri)

Setelah mu_BL dan Sigma_BL diperoleh, portofolio optimal dicari lewat
optimasi mean-variance standar (maksimasi Sharpe ratio) dengan
constraint bobot >= 0 dan total bobot = 1 (long-only, fully invested).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class BLResult:
    prior_return: pd.Series
    posterior_return: pd.Series
    posterior_cov: pd.DataFrame
    optimal_weights: pd.Series
    expected_portfolio_return: float
    expected_portfolio_vol: float
    sharpe_ratio: float


def compute_implied_prior_return(
    cov: pd.DataFrame, market_weights: pd.Series, delta: float = 2.5
) -> pd.Series:
    """Menghitung return prior tersirat (reverse optimization) dari bobot pasar."""
    pi = delta * cov.values @ market_weights.values
    return pd.Series(pi, index=cov.index)


def build_views_from_predictions(
    tickers: list[str], expected_returns: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Menyusun matriks P dan vektor Q untuk "view absolut" sederhana:
    setiap saham yang dipilih user mendapat satu view langsung berupa
    expected return hasil prediksi XGBoost-nya sendiri.

    P berbentuk identitas (K view = K saham, tiap view hanya menyentuh
    1 saham) dan Q adalah vektor expected return tersebut.
    """
    k = len(tickers)
    P = np.eye(k)
    Q = np.array([expected_returns[t] for t in tickers])
    return P, Q


def black_litterman_posterior(
    cov: pd.DataFrame,
    prior_return: pd.Series,
    P: np.ndarray,
    Q: np.ndarray,
    tau: float = 0.025,
    view_confidence: np.ndarray | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Menghitung posterior return dan posterior covariance Black-Litterman.

    view_confidence: array sepanjang jumlah view berisi nilai 0-1
    (1 = sangat percaya pada view, mendekati 0 = ragu). Dipakai untuk
    menyusun Omega proporsional terhadap (1/confidence) * variansnya
    sendiri, mengikuti pendekatan Idzorek (2005) yang lebih intuitif
    daripada menebak Omega secara langsung.
    """
    sigma = cov.values
    pi = prior_return.values.reshape(-1, 1)
    Q = Q.reshape(-1, 1)
    k = P.shape[0]

    if view_confidence is None:
        view_confidence = np.full(k, 0.5)

    view_confidence = np.clip(view_confidence, 1e-3, 1 - 1e-3)

    raw_view_var = np.diag(P @ (tau * sigma) @ P.T)
    omega_diag = raw_view_var * (1.0 / view_confidence - 1.0)
    omega_diag = np.where(omega_diag <= 0, 1e-8, omega_diag)
    omega = np.diag(omega_diag)

    tau_sigma_inv = np.linalg.inv(tau * sigma)
    omega_inv = np.linalg.inv(omega)

    M_inv = tau_sigma_inv + P.T @ omega_inv @ P
    M = np.linalg.inv(M_inv)

    mu_bl = M @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
    sigma_bl = sigma + M

    mu_bl_series = pd.Series(mu_bl.flatten(), index=cov.index)
    sigma_bl_df = pd.DataFrame(sigma_bl, index=cov.index, columns=cov.columns)
    return mu_bl_series, sigma_bl_df


def optimize_max_sharpe(
    expected_return: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.06,
    weight_bounds: tuple[float, float] = (0.0, 0.40),
) -> pd.Series:
    """
    Mencari bobot portofolio yang memaksimalkan Sharpe ratio
    (long-only, fully invested) menggunakan SLSQP.
    risk_free_rate default 6% mengacu pada level suku bunga acuan
    domestik jangka pendek dan dapat disesuaikan pengguna.
    """
    n = len(expected_return)
    mu = expected_return.values
    sigma = cov.values

    def neg_sharpe(w):
        port_return = w @ mu
        port_vol = np.sqrt(w @ sigma @ w)
        if port_vol < 1e-10:
            return 1e6
        return -(port_return - risk_free_rate) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [weight_bounds] * n
    w0 = np.full(n, 1.0 / n)

    result = minimize(
        neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    weights = result.x
    weights = np.clip(weights, 0, None)
    weights = weights / weights.sum()
    return pd.Series(weights, index=expected_return.index)


def optimize_min_variance(
    cov: pd.DataFrame, weight_bounds: tuple[float, float] = (0.0, 1.0)
) -> pd.Series:
    """Mencari bobot portofolio dengan variansi minimum (long-only, fully invested)."""
    n = cov.shape[0]
    sigma = cov.values

    def variance(w):
        return w @ sigma @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [weight_bounds] * n
    w0 = np.full(n, 1.0 / n)

    result = minimize(
        variance, w0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    weights = np.clip(result.x, 0, None)
    weights = weights / weights.sum()
    return pd.Series(weights, index=cov.index)


def run_black_litterman(
    cov: pd.DataFrame,
    expected_returns_xgb: dict[str, float],
    market_weights: pd.Series | None = None,
    delta: float = 2.5,
    tau: float = 0.025,
    view_confidence: np.ndarray | None = None,
    risk_free_rate: float = 0.06,
    max_weight: float = 0.40,
) -> BLResult:
    """Menjalankan seluruh pipeline Black-Litterman dari prior sampai bobot optimal."""
    tickers = list(cov.index)

    if market_weights is None:
        market_weights = pd.Series(np.full(len(tickers), 1.0 / len(tickers)), index=tickers)

    prior = compute_implied_prior_return(cov, market_weights, delta=delta)
    P, Q = build_views_from_predictions(tickers, expected_returns_xgb)
    posterior_return, posterior_cov = black_litterman_posterior(
        cov, prior, P, Q, tau=tau, view_confidence=view_confidence
    )
    n = len(tickers)
    effective_max = max(max_weight, 1.0 / n)
    weights = optimize_max_sharpe(
        posterior_return, posterior_cov, risk_free_rate=risk_free_rate,
        weight_bounds=(0.0, effective_max),
    )

    port_return = float(weights.values @ posterior_return.values)
    port_vol = float(np.sqrt(weights.values @ posterior_cov.values @ weights.values))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0.0

    return BLResult(
        prior_return=prior,
        posterior_return=posterior_return,
        posterior_cov=posterior_cov,
        optimal_weights=weights,
        expected_portfolio_return=port_return,
        expected_portfolio_vol=port_vol,
        sharpe_ratio=sharpe,
    )
