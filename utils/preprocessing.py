"""
Modul preprocessing data harga saham.
Menangani hari-hari dengan volume nol (indikasi saham disuspensi atau
tidak likuid pada hari tersebut) serta menyediakan fungsi perhitungan
return harian dan log-return yang dipakai pada modul XGBoost dan
Black-Litterman.
"""

import numpy as np
import pandas as pd


def flag_illiquid_days(volume_series: pd.Series, threshold: int = 0) -> pd.Series:
    """Mengembalikan mask boolean hari dengan volume <= threshold."""
    return volume_series <= threshold


def clean_price_series(price: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Pada hari dengan volume nol, harga penutupan biasanya hanya mengulang
    harga hari sebelumnya (tidak ada transaksi). Nilai ini valid sebagai
    harga referensi sehingga TIDAK dihapus, namun ditandai agar return
    pada hari tersebut tidak dianggap sebagai sinyal pergerakan pasar
    yang nyata saat pelatihan model. Di sini kita hanya forward-fill
    seandainya ada gap, harga itu sendiri dipertahankan.
    """
    cleaned = price.copy()
    cleaned = cleaned.ffill()
    return cleaned


def compute_returns(price: pd.Series, log_return: bool = True) -> pd.Series:
    """Menghitung return harian (default: log-return) dari deret harga."""
    if log_return:
        return np.log(price / price.shift(1)).dropna()
    return price.pct_change().dropna()


def compute_return_matrix(price_df: pd.DataFrame, log_return: bool = True) -> pd.DataFrame:
    """Menghitung matriks return harian untuk seluruh kolom saham."""
    if log_return:
        ret = np.log(price_df / price_df.shift(1))
    else:
        ret = price_df.pct_change()
    return ret.dropna()


def annualize_return(daily_mean_return: float, periods_per_year: int = 252) -> float:
    return daily_mean_return * periods_per_year


def annualize_volatility(daily_std: float, periods_per_year: int = 252) -> float:
    return daily_std * np.sqrt(periods_per_year)


def annualize_covariance(daily_cov: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    return daily_cov * periods_per_year
