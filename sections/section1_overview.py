"""
Section 1 — Pemilihan Saham & Ringkasan Portofolio.

Pengguna memilih maksimal 10 saham dari IDX80, lalu sistem menjalankan
pipeline lengkap (XGBoost untuk prediksi, Black-Litterman untuk
optimasi) dan menampilkan ringkasan hasilnya pada halaman yang sama.
Hasil pipeline disimpan di st.session_state agar dapat dipakai ulang
oleh Section 2, 3, dan 4 tanpa menghitung ulang.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import black_litterman as bl
from utils import data_loader as dl
from utils import preprocessing as pp
from utils import xgboost_model as xgb_mod
from utils.styling import (
    PLOTLY_TEMPLATE,
    panel_end,
    panel_start,
    render_callout,
    render_metric_card,
    render_section_heading,
    render_ticker_chips,
)

MAX_TICKERS = 10
DEFAULT_SELECTION = ["BBCA", "BBRI", "TLKM", "ASII", "ICBP"]
FORECAST_HORIZON = 10


def _run_full_pipeline(tickers: list[str], horizon: int, max_weight: float, risk_free_rate: float) -> dict:
    """Menjalankan XGBoost untuk tiap saham lalu Black-Litterman untuk seluruh portofolio."""
    price_df = dl.get_price_matrix(tuple(tickers))
    return_matrix = pp.compute_return_matrix(price_df, log_return=True)
    cov_annual = pp.annualize_covariance(return_matrix.cov(), periods_per_year=252)

    train_results = {}
    forecasts = {}
    expected_returns_xgb = {}

    progress = st.progress(0.0, text="Melatih model XGBoost per saham...")
    for i, t in enumerate(tickers):
        price_series = price_df[t]
        train_res = xgb_mod.train_xgboost_model(price_series)
        forecast_df = xgb_mod.forecast_future(
            price_series, train_res.model, train_res.feature_cols, horizon=horizon
        )
        exp_ret = xgb_mod.get_expected_returns_from_predictions(forecast_df, price_series.iloc[-1])

        train_results[t] = train_res
        forecasts[t] = forecast_df
        expected_returns_xgb[t] = exp_ret

        progress.progress((i + 1) / len(tickers), text=f"Memproses {t} ({i+1}/{len(tickers)})")
    progress.empty()

    bl_result = bl.run_black_litterman(
        cov_annual,
        expected_returns_xgb,
        delta=2.5,
        tau=0.025,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
    )

    return {
        "tickers": tickers,
        "price_df": price_df,
        "return_matrix": return_matrix,
        "cov_annual": cov_annual,
        "train_results": train_results,
        "forecasts": forecasts,
        "expected_returns_xgb": expected_returns_xgb,
        "bl_result": bl_result,
        "horizon": horizon,
        "max_weight": max_weight,
        "risk_free_rate": risk_free_rate,
    }


def render() -> None:
    render_section_heading(
        eyebrow="Section 01",
        icon="bi-collection",
        title="Pemilihan Saham & Ringkasan Portofolio",
        description=(
            "Pilih hingga 10 saham dari indeks IDX80 sebagai kandidat portofolio. "
            "Sistem akan melatih model XGBoost untuk memprediksi pergerakan harga "
            "tiap saham, kemudian menggunakan hasil prediksi tersebut sebagai "
            "pandangan (views) pada model Black-Litterman untuk menentukan alokasi optimal."
        ),
    )

    metadata = dl.get_ticker_metadata()
    all_tickers = list(metadata.index)

    panel_start("Kriteria Pemilihan Saham", "bi-sliders")
    col_select, col_param = st.columns([2, 1])

    with col_select:
        selected = st.multiselect(
            "Pilih saham (maksimum 10)",
            options=all_tickers,
            default=[t for t in DEFAULT_SELECTION if t in all_tickers],
            max_selections=MAX_TICKERS,
            help="Kode saham mengikuti penamaan resmi Bursa Efek Indonesia.",
        )

    with col_param:
        horizon = st.slider("Horizon prediksi (hari trading)", 5, 30, FORECAST_HORIZON)
        max_weight_pct = st.slider("Bobot maksimum per saham", 10, 100, 40, step=5)
        risk_free_pct = st.slider("Suku bunga acuan tahunan", 0.0, 10.0, 6.0, step=0.25)

    panel_end()

    with st.expander("Lihat tabel referensi seluruh 80 saham IDX80"):
        ref_df = metadata.copy()
        ref_df["HargaTerakhir"] = ref_df["HargaTerakhir"].map(lambda x: f"Rp {x:,.0f}")
        ref_df["ReturnTotal"] = (ref_df["ReturnTotal"] * 100).map(lambda x: f"{x:+.1f}%")
        ref_df["VolatilitasTahunan"] = (ref_df["VolatilitasTahunan"] * 100).map(lambda x: f"{x:.1f}%")
        ref_df["PersenHariVolumeNol"] = ref_df["PersenHariVolumeNol"].map(lambda x: f"{x:.1f}%")
        ref_df = ref_df[
            ["HargaTerakhir", "ReturnTotal", "VolatilitasTahunan", "PersenHariVolumeNol", "TanggalAwal"]
        ]
        ref_df.columns = ["Harga Terakhir", "Return Total", "Volatilitas Tahunan", "Hari Volume Nol", "Tanggal Listing"]
        st.dataframe(ref_df, use_container_width=True, height=320)

    if len(selected) == 0:
        render_callout(
            "Belum ada saham yang dipilih. Pilih minimal 2 saham untuk menjalankan analisis.",
            icon="bi-info-circle",
        )
        return

    if len(selected) < 2:
        render_callout(
            "Pilih setidaknya 2 saham agar matriks kovarians dapat dihitung untuk optimasi portofolio.",
            icon="bi-exclamation-triangle",
            variant="warning",
        )
        return

    render_ticker_chips(selected)
    st.write("")

    metadata_selected = metadata.loc[selected]
    earliest_start = metadata_selected["TanggalAwal"].max()
    if metadata_selected["TanggalAwal"].nunique() > 1:
        render_callout(
            f"Saham terpilih memiliki tanggal listing yang berbeda-beda. Untuk menjaga "
            f"konsistensi periode antar model, seluruh analisis akan menggunakan data "
            f"mulai {earliest_start.strftime('%d %B %Y')} (tanggal listing saham termuda "
            f"dalam pilihan Anda).",
            icon="bi-calendar-range",
            variant="warning",
        )

    illiquid = metadata_selected[metadata_selected["PersenHariVolumeNol"] > 10]
    if len(illiquid) > 0:
        nama_saham = ", ".join(
            f"{t} ({illiquid.loc[t, 'PersenHariVolumeNol']:.0f}% hari tanpa transaksi)"
            for t in illiquid.index
        )
        render_callout(
            f"Saham berikut memiliki proporsi hari tanpa transaksi (volume nol) yang cukup "
            f"tinggi dalam periode historis: {nama_saham}. Volatilitas dan kovarians yang "
            f"dihitung dari saham-saham ini cenderung under-estimate karena harga yang "
            f"tidak bergerak pada hari-hari tersebut, sehingga estimasi risiko portofolio "
            f"perlu diinterpretasikan dengan hati-hati.",
            icon="bi-exclamation-diamond",
            variant="warning",
        )

    run_clicked = st.button("Jalankan Analisis Portofolio", type="primary", use_container_width=False)

    cache_key = (tuple(sorted(selected)), horizon, max_weight_pct, risk_free_pct)
    state_key = "pipeline_cache_key"
    result_key = "pipeline_result"

    if run_clicked:
        with st.spinner("Menjalankan pipeline XGBoost dan Black-Litterman..."):
            result = _run_full_pipeline(
                selected, horizon, max_weight_pct / 100, risk_free_pct / 100
            )
        st.session_state[result_key] = result
        st.session_state[state_key] = cache_key
    elif result_key not in st.session_state:
        render_callout(
            "Atur kriteria di atas lalu klik \"Jalankan Analisis Portofolio\" untuk memulai.",
            icon="bi-play-circle",
        )
        return
    elif st.session_state.get(state_key) != cache_key:
        render_callout(
            "Parameter telah berubah. Klik \"Jalankan Analisis Portofolio\" untuk memperbarui hasil.",
            icon="bi-arrow-repeat",
            variant="warning",
        )

    result = st.session_state.get(result_key)
    if result is None:
        return

    _render_overview(result)


def _render_overview(result: dict) -> None:
    bl_result = result["bl_result"]
    tickers = result["tickers"]
    weights = bl_result.optimal_weights.sort_values(ascending=False)
    active = weights[weights > 0.001]

    st.markdown("<br>", unsafe_allow_html=True)
    render_callout(
        f"Analisis berhasil dijalankan untuk {len(tickers)} saham dengan horizon prediksi "
        f"{result['horizon']} hari trading. Berikut ringkasan hasil prediksi dan alokasi optimal.",
        icon="bi-check-circle",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            "Expected Return Tahunan", f"{bl_result.expected_portfolio_return*100:.2f}%", "bi-graph-up-arrow"
        )
    with c2:
        render_metric_card(
            "Volatilitas Tahunan", f"{bl_result.expected_portfolio_vol*100:.2f}%", "bi-activity"
        )
    with c3:
        sharpe = bl_result.sharpe_ratio
        delta_type = "positive" if sharpe > 0 else "negative"
        render_metric_card("Sharpe Ratio", f"{sharpe:.2f}", "bi-bullseye", f"{'Baik' if sharpe>1 else 'Moderat' if sharpe>0 else 'Negatif'}", delta_type)
    with c4:
        render_metric_card("Saham Aktif di Portofolio", f"{len(active)} dari {len(tickers)}", "bi-pie-chart")

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        panel_start("Alokasi Bobot Portofolio Optimal", "bi-pie-chart-fill")
        pie_weights = active if len(active) > 0 else weights
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=pie_weights.index,
                    values=pie_weights.values,
                    hole=0.55,
                    textinfo="label+percent",
                    textposition="outside",
                    marker=dict(colors=PLOTLY_TEMPLATE["layout"]["colorway"]),
                    pull=[0.03] * len(pie_weights),
                )
            ]
        )
        fig.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            height=360,
            margin=dict(t=30, b=30, l=30, r=30),
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
        )
        st.plotly_chart(fig, use_container_width=True, key="pie_allocation_chart")
        panel_end()

    with col_right:
        panel_start("Ringkasan Prediksi per Saham", "bi-list-check")
        rows = []
        for t in tickers:
            exp_ret = result["expected_returns_xgb"][t]
            last_price = result["price_df"][t].iloc[-1]
            rows.append(
                {
                    "Saham": t,
                    "Harga Terakhir": f"Rp {last_price:,.0f}",
                    "Prediksi Return": f"{exp_ret*100:+.2f}%",
                    "Bobot Portofolio": f"{weights.get(t, 0)*100:.1f}%",
                }
            )
        df_summary = pd.DataFrame(rows)
        st.dataframe(df_summary, use_container_width=True, hide_index=True, height=360)
        panel_end()

    render_callout(
        "Lihat detail proses prediksi pada Section 2 dan rincian optimasi portofolio pada Section 3.",
        icon="bi-arrow-right-circle",
    )
