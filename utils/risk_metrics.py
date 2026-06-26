"""
Modul metrik risiko portofolio: Value at Risk (VaR), Conditional VaR
(CVaR / Expected Shortfall), kontribusi risiko marjinal per saham, dan
simulasi stress test berbasis skenario shock pasar.
"""

import numpy as np
import pandas as pd


def historical_var_cvar(
    portfolio_returns: pd.Series, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Menghitung VaR dan CVaR historis pada tingkat kepercayaan tertentu.
    VaR dilaporkan sebagai angka positif yang merepresentasikan besar
    kerugian (bukan return negatif), demikian pula CVaR.
    """
    alpha = 1 - confidence
    sorted_returns = portfolio_returns.sort_values()
    var_index = int(np.ceil(alpha * len(sorted_returns))) - 1
    var_index = max(var_index, 0)
    var = -sorted_returns.iloc[var_index]
    cvar = -sorted_returns.iloc[: var_index + 1].mean()
    return float(var), float(cvar)


def parametric_var(
    port_mean: float, port_vol: float, confidence: float = 0.95
) -> float:
    """VaR parametrik dengan asumsi distribusi return normal."""
    from scipy.stats import norm

    z = norm.ppf(1 - confidence)
    var = -(port_mean + z * port_vol)
    return float(var)


def monte_carlo_var_cvar(
    expected_return: pd.Series,
    cov: pd.DataFrame,
    weights: pd.Series,
    confidence: float = 0.95,
    n_sims: int = 20000,
    horizon_days: int = 1,
    random_seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """
    Mensimulasikan distribusi return portofolio dengan Monte Carlo
    (asumsi return multivariate normal mengikuti mu dan Sigma posterior
    Black-Litterman), lalu menghitung VaR dan CVaR dari hasil simulasi.
    Mengembalikan juga array hasil simulasi untuk divisualisasikan.
    """
    rng = np.random.default_rng(random_seed)
    mu_daily = expected_return.values / 252 * horizon_days
    cov_daily = cov.values / 252 * horizon_days

    sims = rng.multivariate_normal(mu_daily, cov_daily, size=n_sims)
    port_sims = sims @ weights.values

    var, cvar = historical_var_cvar(pd.Series(port_sims), confidence=confidence)
    return var, cvar, port_sims


def risk_contribution(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung kontribusi risiko marjinal (Marginal Contribution to Risk)
    dan kontribusi risiko absolut/persentase tiap saham terhadap total
    volatilitas portofolio:

        sigma_p = sqrt(w' Sigma w)
        MCR_i   = (Sigma w)_i / sigma_p
        CR_i    = w_i * MCR_i
        %CR_i   = CR_i / sigma_p
    """
    w = weights.values
    sigma = cov.values
    port_vol = np.sqrt(w @ sigma @ w)

    marginal_contrib = (sigma @ w) / port_vol
    contrib = w * marginal_contrib
    pct_contrib = contrib / port_vol

    return pd.DataFrame(
        {
            "Bobot": weights.values,
            "KontribusiMarjinal": marginal_contrib,
            "KontribusiRisiko": contrib,
            "PersenKontribusiRisiko": pct_contrib,
        },
        index=weights.index,
    ).sort_values("PersenKontribusiRisiko", ascending=False)


def apply_shock_scenario(
    expected_return: pd.Series,
    shock_pct: float,
    affected_tickers: list[str] | None = None,
) -> pd.Series:
    """
    Menerapkan skenario shock berupa penurunan/kenaikan persentase
    tertentu pada expected return saham yang dipilih (atau seluruh
    saham bila affected_tickers None), digunakan pada simulasi what-if.
    """
    shocked = expected_return.copy()
    targets = affected_tickers if affected_tickers else list(expected_return.index)
    for t in targets:
        if t in shocked.index:
            shocked[t] = shocked[t] + shock_pct
    return shocked


def simulate_portfolio_value_path(
    initial_value: float,
    daily_mean: float,
    daily_vol: float,
    n_days: int,
    n_paths: int = 200,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Mensimulasikan lintasan nilai portofolio ke depan menggunakan
    Geometric Brownian Motion sederhana, dipakai untuk visualisasi
    fan-chart pada simulasi what-if.
    Mengembalikan array berdimensi (n_paths, n_days+1).
    """
    rng = np.random.default_rng(random_seed)
    dt = 1
    shocks = rng.normal(
        loc=daily_mean, scale=daily_vol, size=(n_paths, n_days)
    )
    log_paths = np.cumsum(shocks, axis=1)
    paths = initial_value * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))
    return paths
