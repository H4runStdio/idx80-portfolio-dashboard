"""
Section 2 — Detail Prediksi Harga (XGBoost).

Menampilkan, untuk setiap saham yang dipilih pada Section 1:
- Grafik harga aktual vs hasil prediksi pada periode pengujian (test set)
- Metrik akurasi model (RMSE pada return, MAPE pada harga)
- Grafik forecast ke depan sejauh horizon yang dipilih
- Tingkat kepentingan fitur (feature importance) dari model
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.styling import (
    PLOTLY_TEMPLATE,
    panel_end,
    panel_start,
    render_callout,
    render_metric_card,
    render_section_heading,
)

FEATURE_LABELS = {
    "rsi": "RSI (14 hari)",
    "momentum_10": "Momentum 10 Hari",
}


def _humanize_feature(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.startswith("lag_return_"):
        return f"Lag Return t-{name.split('_')[-1]}"
    if name.startswith("roll_mean_"):
        return f"Rata-rata Return {name.split('_')[-1]} Hari"
    if name.startswith("roll_std_"):
        return f"Volatilitas Return {name.split('_')[-1]} Hari"
    if name.startswith("price_to_ma_"):
        return f"Rasio Harga/MA {name.split('_')[-1]} Hari"
    return name


def render() -> None:
    render_section_heading(
        eyebrow="Section 02",
        icon="bi-graph-up",
        title="Detail Prediksi Harga Saham",
        description=(
            "Evaluasi performa model XGBoost untuk setiap saham terpilih, mencakup "
            "akurasi pada data historis (backtesting) dan proyeksi harga ke depan "
            "yang menjadi dasar pandangan (views) pada model Black-Litterman."
        ),
    )

    result = st.session_state.get("pipeline_result")
    if result is None:
        render_callout(
            "Belum ada hasil analisis. Jalankan pipeline pada Section 1 terlebih dahulu.",
            icon="bi-exclamation-triangle",
            variant="warning",
        )
        return

    tickers = result["tickers"]
    tab_labels = tickers
    tabs = st.tabs(tab_labels)

    for tab, ticker in zip(tabs, tickers):
        with tab:
            _render_ticker_detail(result, ticker)


def _render_ticker_detail(result: dict, ticker: str) -> None:
    train_res = result["train_results"][ticker]
    forecast_df = result["forecasts"][ticker]
    price_series = result["price_df"][ticker]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("RMSE Return (Test)", f"{train_res.test_rmse:.5f}", "bi-rulers")
    with c2:
        render_metric_card("MAPE Harga (Test)", f"{train_res.test_mape:.2f}%", "bi-percent")
    with c3:
        exp_ret = result["expected_returns_xgb"][ticker]
        delta_type = "positive" if exp_ret > 0 else "negative"
        render_metric_card(
            f"Prediksi Return {result['horizon']} Hari", f"{exp_ret*100:+.2f}%", "bi-graph-up-arrow",
            delta_type=delta_type,
        )
    with c4:
        final_price = forecast_df["PredictedClose"].iloc[-1]
        render_metric_card("Proyeksi Harga Akhir", f"Rp {final_price:,.0f}", "bi-currency-exchange")

    st.markdown("<br>", unsafe_allow_html=True)

    panel_start(f"Harga Aktual vs Prediksi — {ticker}", "bi-bar-chart-line")

    test_dates = train_res.y_test.index
    test_positions = price_series.index.get_indexer(test_dates)
    next_positions = test_positions + 1
    valid_mask = next_positions < len(price_series)

    pred_dates = price_series.index[next_positions[valid_mask]]
    base_price_test = price_series.iloc[test_positions[valid_mask]].values
    actual_price_test = pd.Series(
        price_series.iloc[next_positions[valid_mask]].values, index=pred_dates
    )
    pred_price_test = pd.Series(
        base_price_test * np.exp(train_res.y_pred_test.values[valid_mask]), index=pred_dates
    )

    history_window = price_series.loc[: test_dates[0]].iloc[-60:]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_window.index, y=history_window.values,
            name="Harga Historis (Train)", line=dict(color="#5A6776", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=actual_price_test.index, y=actual_price_test.values,
            name="Harga Aktual (Test)", line=dict(color="#2DD4A7", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pred_price_test.index, y=pred_price_test.values,
            name="Harga Prediksi (Test)", line=dict(color="#E8A33D", width=2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df.index, y=forecast_df["PredictedClose"].values,
            name=f"Forecast {result['horizon']} Hari ke Depan",
            line=dict(color="#4D9FE5", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"], height=420,
        margin=dict(t=20, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        yaxis_title="Harga (Rp)",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"price_chart_{ticker}")
    panel_end()

    col_left, col_right = st.columns(2)

    with col_left:
        panel_start("Tingkat Kepentingan Fitur", "bi-bar-chart-steps")
        importances = pd.Series(
            train_res.model.feature_importances_, index=train_res.feature_cols
        ).sort_values(ascending=True).tail(8)
        importances.index = [_humanize_feature(i) for i in importances.index]

        fig_imp = go.Figure(
            go.Bar(
                x=importances.values, y=importances.index, orientation="h",
                marker_color="#2DD4A7",
            )
        )
        fig_imp.update_layout(
            **PLOTLY_TEMPLATE["layout"], height=320,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_imp, use_container_width=True, key=f"importance_chart_{ticker}")
        panel_end()

    with col_right:
        panel_start("Tabel Proyeksi Harga", "bi-table")
        display_df = forecast_df.copy()
        display_df.index = display_df.index.strftime("%d %b %Y")
        display_df["PredictedClose"] = display_df["PredictedClose"].map(lambda x: f"Rp {x:,.0f}")
        display_df["PredictedReturn"] = display_df["PredictedReturn"].map(lambda x: f"{x*100:+.2f}%")
        display_df.columns = ["Harga Prediksi", "Return Harian"]
        st.dataframe(display_df, use_container_width=True, height=320)
        panel_end()

    render_callout(
        "RMSE dihitung pada skala log-return harian, sedangkan MAPE direkonstruksi pada skala "
        "harga untuk interpretasi yang lebih intuitif. Forecast multi-hari dilakukan secara "
        "recursive: prediksi satu hari digunakan kembali sebagai input penyusunan fitur untuk hari berikutnya.",
        icon="bi-info-circle",
    )
