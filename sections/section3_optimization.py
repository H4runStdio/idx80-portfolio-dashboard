"""
Section 3 — Detail Optimasi Portofolio (Black-Litterman).

Menampilkan komponen inti model Black-Litterman: return prior (implied
dari pasar), return posterior (setelah dipadukan dengan views XGBoost),
bobot alokasi akhir, serta efficient frontier untuk membandingkan
posisi portofolio terpilih relatif terhadap kombinasi bobot lain yang
mungkin dibentuk dari saham yang sama.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import black_litterman as bl
from utils.styling import (
    PLOTLY_TEMPLATE,
    panel_end,
    panel_start,
    render_callout,
    render_metric_card,
    render_section_heading,
)


def render() -> None:
    render_section_heading(
        eyebrow="Section 03",
        icon="bi-diagram-3",
        title="Detail Optimasi Portofolio Black-Litterman",
        description=(
            "Telusuri bagaimana model Black-Litterman memadukan return tersirat pasar "
            "(prior) dengan pandangan dari hasil prediksi XGBoost (views) untuk membentuk "
            "estimasi return posterior, yang kemudian menjadi dasar optimasi bobot portofolio."
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

    bl_result = result["bl_result"]
    tickers = result["tickers"]

    _render_prior_vs_posterior(bl_result, tickers)
    st.markdown("<br>", unsafe_allow_html=True)
    _render_weights_and_frontier(result)
    st.markdown("<br>", unsafe_allow_html=True)
    _render_covariance_heatmap(result)


def _render_prior_vs_posterior(bl_result, tickers: list[str]) -> None:
    panel_start("Return Prior vs Posterior", "bi-arrow-left-right")

    comparison = pd.DataFrame(
        {
            "Prior (Implied Pasar)": bl_result.prior_return,
            "Posterior (Setelah Views XGBoost)": bl_result.posterior_return,
        }
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=comparison.index, y=comparison["Prior (Implied Pasar)"] * 100,
            name="Prior (Implied Pasar)", marker_color="#5A6776",
        )
    )
    fig.add_trace(
        go.Bar(
            x=comparison.index, y=comparison["Posterior (Setelah Views XGBoost)"] * 100,
            name="Posterior (Setelah Views XGBoost)", marker_color="#2DD4A7",
        )
    )
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"], height=380, barmode="group",
        margin=dict(t=20, b=10, l=10, r=10),
        yaxis_title="Expected Return Tahunan (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True, key="prior_posterior_chart")
    panel_end()

    render_callout(
        "Return prior diperoleh melalui reverse optimization dari bobot pasar (di sini "
        "didekati equal-weight karena dataset tidak menyertakan kapitalisasi pasar). "
        "Selisih antara prior dan posterior menunjukkan seberapa besar pandangan dari "
        "XGBoost menggeser ekspektasi return dibandingkan asumsi pasar netral.",
        icon="bi-info-circle",
    )


def _render_weights_and_frontier(result: dict) -> None:
    bl_result = result["bl_result"]
    cov = bl_result.posterior_cov
    mu = bl_result.posterior_return
    tickers = result["tickers"]

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        panel_start("Bobot Alokasi Optimal", "bi-bar-chart-fill")
        weights_sorted = bl_result.optimal_weights.sort_values(ascending=True)
        fig = go.Figure(
            go.Bar(
                x=weights_sorted.values * 100, y=weights_sorted.index, orientation="h",
                marker_color="#2DD4A7",
                text=[f"{v*100:.1f}%" for v in weights_sorted.values],
                textposition="outside",
            )
        )
        fig.update_layout(
            **PLOTLY_TEMPLATE["layout"], height=max(320, 36 * len(tickers)),
            margin=dict(t=10, b=10, l=10, r=40),
            xaxis_title="Bobot Portofolio (%)",
        )
        st.plotly_chart(fig, use_container_width=True, key="weights_bar_chart")
        panel_end()

    with col_right:
        panel_start("Efficient Frontier", "bi-graph-up")
        frontier_df, current_point = _build_efficient_frontier(mu, cov, bl_result, n_portfolios=2500)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=frontier_df["vol"] * 100, y=frontier_df["ret"] * 100,
                mode="markers",
                marker=dict(
                    size=5, color=frontier_df["sharpe"],
                    colorscale=[[0, "#5A6776"], [0.5, "#4D9FE5"], [1, "#2DD4A7"]],
                    showscale=True, colorbar=dict(title="Sharpe"),
                ),
                name="Kombinasi Bobot Acak",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[current_point["vol"] * 100], y=[current_point["ret"] * 100],
                mode="markers", marker=dict(size=16, color="#E8A33D", symbol="star"),
                name="Portofolio Terpilih",
            )
        )
        fig.update_layout(
            **PLOTLY_TEMPLATE["layout"], height=400,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title="Volatilitas Tahunan (%)", yaxis_title="Expected Return Tahunan (%)",
        )
        st.plotly_chart(fig, use_container_width=True, key="efficient_frontier_chart")
        panel_end()


@st.cache_data(show_spinner=False)
def _build_efficient_frontier(mu: pd.Series, cov: pd.DataFrame, _bl_result, n_portfolios: int = 2500):
    rng = np.random.default_rng(7)
    n = len(mu)
    mu_arr = mu.values
    cov_arr = cov.values

    weights_random = rng.dirichlet(np.ones(n), size=n_portfolios)
    rets = weights_random @ mu_arr
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights_random, cov_arr, weights_random))
    sharpe = (rets - 0.06) / vols

    frontier_df = pd.DataFrame({"ret": rets, "vol": vols, "sharpe": sharpe})

    w_opt = _bl_result.optimal_weights.values
    current_point = {
        "ret": float(w_opt @ mu_arr),
        "vol": float(np.sqrt(w_opt @ cov_arr @ w_opt)),
    }
    return frontier_df, current_point


def _render_covariance_heatmap(result: dict) -> None:
    panel_start("Matriks Korelasi Antar Saham", "bi-grid-3x3")
    return_matrix = result["return_matrix"]
    corr = return_matrix.corr()

    fig = go.Figure(
        go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale=[[0, "#E5484D"], [0.5, "#1A222C"], [1, "#2DD4A7"]],
            zmin=-1, zmax=1,
            text=corr.round(2).values, texttemplate="%{text}",
            textfont=dict(size=10),
        )
    )
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"], height=max(360, 40 * len(corr)),
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True, key="correlation_heatmap")
    panel_end()

    render_callout(
        "Korelasi antar saham memengaruhi besarnya manfaat diversifikasi: saham dengan "
        "korelasi rendah atau negatif terhadap saham lain dalam portofolio cenderung "
        "menurunkan volatilitas total tanpa mengorbankan banyak expected return.",
        icon="bi-info-circle",
    )
