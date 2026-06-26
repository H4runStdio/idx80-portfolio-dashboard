"""
Modul pemuatan data untuk dashboard IDX80.
Membaca seluruh sheet pada file Excel (1 sheet = 1 saham) dan
menyediakan akses terstruktur ke harga penutupan dan volume historis.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "IDX80_5yr.xlsx"


@st.cache_data(show_spinner=False)
def load_all_tickers() -> list[str]:
    """Mengembalikan daftar kode saham (nama sheet) yang tersedia."""
    xls = pd.ExcelFile(DATA_PATH)
    return sorted(xls.sheet_names)


@st.cache_data(show_spinner=False)
def load_raw_data() -> dict[str, pd.DataFrame]:
    """
    Memuat seluruh sheet sekaligus ke dalam dictionary {ticker: DataFrame}.
    Dipanggil sekali di awal sesi lalu di-cache oleh Streamlit agar tidak
    membaca ulang file Excel pada setiap interaksi pengguna.
    """
    sheets = pd.read_excel(DATA_PATH, sheet_name=None)
    cleaned = {}
    for ticker, df in sheets.items():
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        cleaned[ticker] = df
    return cleaned


@st.cache_data(show_spinner=False)
def get_price_matrix(tickers: tuple[str, ...]) -> pd.DataFrame:
    """
    Menyusun matriks harga penutupan (Date x Ticker) untuk sekumpulan saham.
    Menggunakan inner join pada tanggal agar seluruh kolom memiliki
    panjang historis yang sama (penting untuk perhitungan kovariansi BL).
    """
    raw = load_raw_data()
    series_list = []
    for t in tickers:
        s = raw[t].set_index("Date")["Close"].rename(t)
        series_list.append(s)
    price_df = pd.concat(series_list, axis=1, join="inner")
    price_df = price_df.sort_index()
    return price_df


@st.cache_data(show_spinner=False)
def get_volume_matrix(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Menyusun matriks volume perdagangan (Date x Ticker)."""
    raw = load_raw_data()
    series_list = []
    for t in tickers:
        s = raw[t].set_index("Date")["Volume"].rename(t)
        series_list.append(s)
    vol_df = pd.concat(series_list, axis=1, join="inner")
    return vol_df.sort_index()


def get_ticker_metadata() -> pd.DataFrame:
    """
    Menghitung metadata ringkas tiap saham: rentang tanggal, jumlah hari
    trading, harga terakhir, return historis, dan persentase hari dengan
    volume nol (indikasi suspensi/ilikuiditas) — dipakai pada panel
    pemilihan saham di Section 1.
    """
    raw = load_raw_data()
    rows = []
    for ticker, df in raw.items():
        last_close = df["Close"].iloc[-1]
        first_close = df["Close"].iloc[0]
        n_days = len(df)
        total_return = (last_close / first_close) - 1
        daily_ret = df["Close"].pct_change().dropna()
        ann_vol = daily_ret.std() * (252 ** 0.5)
        pct_zero_volume = (df["Volume"] == 0).sum() / n_days * 100
        rows.append(
            {
                "Ticker": ticker,
                "HargaTerakhir": last_close,
                "TanggalAwal": df["Date"].iloc[0],
                "TanggalAkhir": df["Date"].iloc[-1],
                "JumlahHari": n_days,
                "ReturnTotal": total_return,
                "VolatilitasTahunan": ann_vol,
                "PersenHariVolumeNol": pct_zero_volume,
            }
        )
    return pd.DataFrame(rows).set_index("Ticker")
