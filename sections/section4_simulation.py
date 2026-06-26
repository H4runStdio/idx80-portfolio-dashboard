"""
Section 4 — Simulasi What-If & Manajemen Risiko.

Tiga kelompok analisis:
1. Value at Risk (VaR) & Conditional VaR (CVaR) — historis, parametrik,
   dan Monte Carlo.
2. Kontribusi risiko per saham terhadap volatilitas portofolio.
3. Simulasi skenario shock pasar (stress test) dan sensitivitas
   parameter Black-Litterman (risk aversion delta, tau).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import black_litterman as bl
from utils import risk_metrics as rm
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
        eyebrow="Section 04",
        icon="bi-shield-exclamation",
        title="Simulasi What-If & Manajemen Risiko",
        description=(
            "Uji ketahanan portofolio terhadap skenario pasar yang tidak diharapkan, "
            "pahami sumber risiko pada level saham individual, dan amati bagaimana "
            "perubahan asumsi model memengaruhi alokasi yang disarankan."
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

    tab_var, tab_risk_contrib, tab_stress, tab_sensitivity = st.tabs(
        [
            "Value at Risk",
            "Kontribusi Risiko",
            "Skenario Shock",
            "Sensitivitas Parameter",
        ]
    )

    with tab_var:
        _render_var_section(result)
    with tab_risk_contrib:
        _render_risk_contribution(result)
    with tab_stress:
        _render_stress_test(result)
    with tab_sensitivity:
        _render_sensitivity(result)


def _render_var_section(result: dict) -> None:
    bl_result = result["bl_result"]
    weights = bl_result.optimal_weights
    return_matrix = result["return_matrix"]

    confidence = st.select_slider(
        "Tingkat kepercayaan", options=[0.90, 0.95, 0.975, 0.99], value=0.95,
        format_func=lambda x: f"{x*100:.1f}%",
    )

    port_return_hist = return_matrix @ weights
    var_hist, cvar_hist = rm.historical_var_cvar(port_return_hist, confidence=confidence)

    port_mean_daily = float(bl_result.posterior_return.values @ weights.values) / 252
    port_vol_daily = bl_result.expected_portfolio_vol / np.sqrt(252)
    var_param = rm.parametric_var(port_mean_daily, port_vol_daily, confidence=confidence)

    var_mc, cvar_mc, sims = rm.monte_carlo_var_cvar(
        bl_result.posterior_return, bl_result.posterior_cov, weights, confidence=confidence
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("VaR Historis (1 Hari)", f"{var_hist*100:.2f}%", "bi-clock-history")
    with c2:
        render_metric_card("CVaR Historis (1 Hari)", f"{cvar_hist*100:.2f}%", "bi-exclamation-diamond")
    with c3:
        render_metric_card("VaR Parametrik (1 Hari)", f"{var_param*100:.2f}%", "bi-calculator")
    with c4:
        render_metric_card("VaR Monte Carlo (1 Hari)", f"{var_mc*100:.2f}%", "bi-dice-5")

    st.markdown("<br>", unsafe_allow_html=True)
    panel_start("Distribusi Simulasi Return Portofolio (Monte Carlo)", "bi-bar-chart")

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=sims * 100, nbinsx=80, marker_color="#2DD4A7", opacity=0.85, name="Simulasi Return",
        )
    )
    fig.add_vline(
        x=-var_mc * 100, line_dash="dash", line_color="#E8A33D",
        annotation_text=f"VaR {confidence*100:.1f}%", annotation_position="top",
    )
    fig.add_vline(
        x=-cvar_mc * 100, line_dash="dash", line_color="#E5484D",
        annotation_text=f"CVaR {confidence*100:.1f}%", annotation_position="top",
    )
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"], height=380,
        margin=dict(t=20, b=10, l=10, r=10),
        xaxis_title="Return Harian Portofolio (%)", yaxis_title="Frekuensi",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="mc_distribution_chart")
    panel_end()

    render_callout(
        f"Dengan tingkat kepercayaan {confidence*100:.1f}%, kerugian harian portofolio "
        f"secara historis tidak diperkirakan melebihi {var_hist*100:.2f}% dalam kondisi pasar normal. "
        f"CVaR mengukur rata-rata kerugian pada skenario terburuk di luar batas VaR tersebut, "
        f"sehingga memberikan gambaran risiko ekor (tail risk) yang lebih konservatif.",
        icon="bi-info-circle",
    )


def _render_risk_contribution(result: dict) -> None:
    bl_result = result["bl_result"]
    rc = rm.risk_contribution(bl_result.optimal_weights, bl_result.posterior_cov)

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        panel_start("Kontribusi Risiko per Saham", "bi-pie-chart")
        rc_sorted = rc.sort_values("PersenKontribusiRisiko", ascending=True)
        fig = go.Figure(
            go.Bar(
                x=rc_sorted["PersenKontribusiRisiko"] * 100, y=rc_sorted.index, orientation="h",
                marker_color="#4D9FE5",
                text=[f"{v*100:.1f}%" for v in rc_sorted["PersenKontribusiRisiko"]],
                textposition="outside",
            )
        )
        fig.update_layout(
            **PLOTLY_TEMPLATE["layout"], height=max(320, 40 * len(rc)),
            margin=dict(t=10, b=10, l=10, r=40),
            xaxis_title="Kontribusi terhadap Volatilitas Portofolio (%)",
        )
        st.plotly_chart(fig, use_container_width=True, key="risk_contrib_chart")
        panel_end()

    with col_right:
        panel_start("Bobot vs Kontribusi Risiko", "bi-table")
        display_df = rc.copy()
        display_df["Bobot"] = (display_df["Bobot"] * 100).map(lambda x: f"{x:.1f}%")
        display_df["PersenKontribusiRisiko"] = (display_df["PersenKontribusiRisiko"] * 100).map(lambda x: f"{x:.1f}%")
        display_df = display_df[["Bobot", "PersenKontribusiRisiko"]]
        display_df.columns = ["Bobot Portofolio", "Kontribusi Risiko"]
        st.dataframe(display_df, use_container_width=True, height=max(320, 40 * len(rc)))
        panel_end()

    render_callout(
        "Saham dengan bobot kecil dapat menyumbang kontribusi risiko yang besar apabila "
        "volatilitas atau korelasinya terhadap saham lain dalam portofolio tinggi. "
        "Perbandingan kolom bobot dan kontribusi risiko membantu mengidentifikasi saham "
        "yang secara tidak proporsional meningkatkan risiko portofolio.",
        icon="bi-info-circle",
    )


SHOCK_PRESETS = {
    "Koreksi Pasar Ringan (-5%)": -0.05,
    "Koreksi Pasar Sedang (-10%)": -0.10,
    "Koreksi Pasar Tajam (-20%)": -0.20,
    "Rally Pasar (+10%)": 0.10,
}


def _render_stress_test(result: dict) -> None:
    bl_result = result["bl_result"]
    tickers = result["tickers"]
    weights = bl_result.optimal_weights

    col_a, col_b = st.columns([1, 1])
    with col_a:
        preset_label = st.selectbox("Skenario Shock", options=list(SHOCK_PRESETS.keys()))
        shock_pct = SHOCK_PRESETS[preset_label]
    with col_b:
        affected = st.multiselect(
            "Saham Terdampak (kosongkan untuk seluruh portofolio)",
            options=tickers, default=[],
        )

    affected_list = affected if affected else None
    shocked_return = rm.apply_shock_scenario(bl_result.posterior_return, shock_pct, affected_list)

    base_port_return = float(weights.values @ bl_result.posterior_return.values)
    shocked_port_return = float(weights.values @ shocked_return.values)
    impact = shocked_port_return - base_port_return

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Expected Return Sebelum Shock", f"{base_port_return*100:.2f}%", "bi-graph-up")
    with c2:
        render_metric_card("Expected Return Setelah Shock", f"{shocked_port_return*100:.2f}%", "bi-graph-down")
    with c3:
        delta_type = "negative" if impact < 0 else "positive"
        render_metric_card("Dampak terhadap Portofolio", f"{impact*100:+.2f}%", "bi-arrow-down-up", delta_type=delta_type)

    st.markdown("<br>", unsafe_allow_html=True)
    panel_start("Simulasi Lintasan Nilai Portofolio", "bi-graph-down-arrow")

    initial_value = 100_000_000.0
    n_days = 60
    port_vol_daily = bl_result.expected_portfolio_vol / np.sqrt(252)

    base_mean_daily = base_port_return / 252
    shocked_mean_daily = shocked_port_return / 252

    paths_base = rm.simulate_portfolio_value_path(
        initial_value, base_mean_daily, port_vol_daily, n_days, n_paths=150, random_seed=1
    )
    paths_shocked = rm.simulate_portfolio_value_path(
        initial_value, shocked_mean_daily, port_vol_daily, n_days, n_paths=150, random_seed=1
    )

    days_axis = list(range(n_days + 1))
    base_median = np.median(paths_base, axis=0)
    base_p10 = np.percentile(paths_base, 10, axis=0)
    base_p90 = np.percentile(paths_base, 90, axis=0)
    shocked_median = np.median(paths_shocked, axis=0)
    shocked_p10 = np.percentile(paths_shocked, 10, axis=0)
    shocked_p90 = np.percentile(paths_shocked, 90, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days_axis, y=base_p90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=days_axis, y=base_p10, fill="tonexty", fillcolor="rgba(45,212,167,0.12)",
            line=dict(width=0), name="Rentang P10-P90 (Sebelum Shock)",
        )
    )
    fig.add_trace(go.Scatter(x=days_axis, y=base_median, line=dict(color="#2DD4A7", width=2), name="Median (Sebelum Shock)"))

    fig.add_trace(go.Scatter(x=days_axis, y=shocked_p90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=days_axis, y=shocked_p10, fill="tonexty", fillcolor="rgba(229,72,77,0.12)",
            line=dict(width=0), name="Rentang P10-P90 (Setelah Shock)",
        )
    )
    fig.add_trace(go.Scatter(x=days_axis, y=shocked_median, line=dict(color="#E5484D", width=2, dash="dot"), name="Median (Setelah Shock)"))

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"], height=420,
        margin=dict(t=20, b=10, l=10, r=10),
        xaxis_title="Hari ke Depan", yaxis_title="Nilai Portofolio (Rp, asumsi modal awal Rp 100 juta)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True, key="stress_path_chart")
    panel_end()

    render_callout(
        "Simulasi lintasan menggunakan Geometric Brownian Motion dengan rata-rata return "
        "yang disesuaikan skenario shock dan volatilitas historis portofolio. Pita berwarna "
        "menunjukkan rentang persentil ke-10 hingga ke-90 dari seluruh lintasan yang disimulasikan, "
        "bukan prediksi pasti.",
        icon="bi-info-circle",
    )


def _render_sensitivity(result: dict) -> None:
    cov = result["cov_annual"]
    expected_returns_xgb = result["expected_returns_xgb"]
    base_max_weight = result["max_weight"]
    base_rf = result["risk_free_rate"]

    col_a, col_b = st.columns(2)
    with col_a:
        delta_test = st.slider("Risk Aversion (delta)", 1.0, 5.0, 2.5, step=0.25)
    with col_b:
        tau_test = st.slider("Ketidakpastian Prior (tau)", 0.01, 0.10, 0.025, step=0.005)

    bl_result_sens = bl.run_black_litterman(
        cov, expected_returns_xgb, delta=delta_test, tau=tau_test,
        risk_free_rate=base_rf, max_weight=base_max_weight,
    )

    panel_start("Perbandingan Bobot: Parameter Awal vs Parameter Uji", "bi-sliders2")

    base_result = st.session_state["pipeline_result"]["bl_result"]
    comparison = pd.DataFrame(
        {
            "Parameter Awal (delta=2.5, tau=0.025)": base_result.optimal_weights,
            f"Parameter Uji (delta={delta_test}, tau={tau_test})": bl_result_sens.optimal_weights,
        }
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=comparison.index, y=comparison.iloc[:, 0] * 100,
            name="Parameter Awal", marker_color="#5A6776",
        )
    )
    fig.add_trace(
        go.Bar(
            x=comparison.index, y=comparison.iloc[:, 1] * 100,
            name="Parameter Uji", marker_color="#2DD4A7",
        )
    )
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"], height=380, barmode="group",
        margin=dict(t=20, b=10, l=10, r=10),
        yaxis_title="Bobot Portofolio (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True, key="sensitivity_chart")
    panel_end()

    c1, c2 = st.columns(2)
    with c1:
        render_metric_card(
            "Expected Return (Parameter Uji)", f"{bl_result_sens.expected_portfolio_return*100:.2f}%", "bi-graph-up-arrow"
        )
    with c2:
        render_metric_card(
            "Volatilitas (Parameter Uji)", f"{bl_result_sens.expected_portfolio_vol*100:.2f}%", "bi-activity"
        )

    render_callout(
        "Parameter delta (risk aversion) menentukan seberapa besar return prior diturunkan "
        "dari volatilitas pasar; nilai yang lebih tinggi mencerminkan investor yang lebih "
        "menghindari risiko. Parameter tau mengatur seberapa besar keyakinan terhadap "
        "estimasi prior dibandingkan terhadap views dari XGBoost — nilai tau yang lebih kecil "
        "membuat model lebih percaya pada asumsi pasar awal, sedangkan nilai yang lebih besar "
        "memberi bobot lebih pada hasil prediksi.",
        icon="bi-info-circle",
    )
