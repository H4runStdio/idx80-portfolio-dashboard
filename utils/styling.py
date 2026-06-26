"""
Modul utilitas tampilan: menyuntikkan stylesheet kustom, memuat font,
ikon Bootstrap Icons, dan menyediakan komponen HTML siap pakai
(metric card, panel, badge, callout) agar tampilan dashboard konsisten
secara profesional tanpa bergantung pada emoji.
"""

from pathlib import Path

import streamlit as st

CSS_PATH = Path(__file__).resolve().parent.parent / "assets" / "style.css"

BOOTSTRAP_ICONS_CDN = (
    "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css"
)
GOOGLE_FONTS_CDN = (
    "https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600;700&"
    "family=IBM+Plex+Mono:wght@400;500;600&display=swap"
)


def inject_global_styles() -> None:
    """Menyuntikkan font eksternal, Bootstrap Icons, dan CSS kustom ke halaman."""
    css_content = CSS_PATH.read_text()
    html_block = (
        f'<link href="{GOOGLE_FONTS_CDN}" rel="stylesheet">'
        f'<link rel="stylesheet" href="{BOOTSTRAP_ICONS_CDN}">'
        f"<style>{css_content}</style>"
    )
    st.markdown(html_block, unsafe_allow_html=True)

def render_app_header() -> None:
    """Header brand utama di puncak halaman."""
    st.markdown(
        """
        <div class="app-header">
            <div class="app-header-left">
                <div class="app-header-icon"><i class="bi bi-graph-up-arrow"></i></div>
                <div>
                    <p class="app-header-title">IDX80 Portfolio Intelligence</p>
                    <p class="app-header-subtitle">XGBoost Forecasting &amp; Black-Litterman Allocation</p>
                </div>
            </div>
            <div class="app-header-badge">
                <i class="bi bi-circle-fill" style="font-size:0.5rem;"></i>
                <span>DATA SAMPAI 5 JUN 2026</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(eyebrow: str, icon: str, title: str, description: str) -> None:
    """Heading standar tiap section: label kecil + judul + deskripsi singkat."""
    st.markdown(
        f"""
        <div class="section-eyebrow"><i class="bi {icon}"></i><span>{eyebrow}</span></div>
        <p class="section-title">{title}</p>
        <p class="section-desc">{description}</p>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, icon: str, delta: str | None = None, delta_type: str = "neutral") -> str:
    """Mengembalikan HTML untuk satu kartu metrik (dipakai dalam st.columns)."""
    delta_html = ""
    if delta is not None:
        delta_html = f'<div class="metric-card-delta delta-{delta_type}">{delta}</div>'
    return f"""
        <div class="metric-card">
            <div class="metric-card-label"><i class="bi {icon}"></i><span>{label}</span></div>
            <div class="metric-card-value">{value}</div>
            {delta_html}
        </div>
    """


def render_metric_card(label: str, value: str, icon: str, delta: str | None = None, delta_type: str = "neutral") -> None:
    st.markdown(metric_card(label, value, icon, delta, delta_type), unsafe_allow_html=True)


def panel_start(title: str, icon: str) -> None:
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title"><i class="bi {icon}"></i><span>{title}</span></div>
        """,
        unsafe_allow_html=True,
    )


def panel_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_callout(text: str, icon: str = "bi-info-circle", variant: str = "info") -> None:
    variant_class = "" if variant == "info" else variant
    st.markdown(
        f"""
        <div class="callout {variant_class}">
            <i class="bi {icon}"></i>
            <span>{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ticker_chips(tickers: list[str]) -> None:
    chips = "".join(
        f'<span class="ticker-chip"><i class="bi bi-check-circle-fill" style="font-size:0.7rem;"></i>{t}</span>'
        for t in tickers
    )
    st.markdown(chips, unsafe_allow_html=True)


def render_hairline() -> None:
    st.markdown('<hr class="hairline">', unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown(
        """
        <div class="app-footer">
            IDX80 Portfolio Intelligence &middot; Dibangun dengan XGBoost &amp; Black-Litterman Model
        </div>
        """,
        unsafe_allow_html=True,
    )


# Plotly chart theme konsisten dengan palet dashboard
PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, sans-serif", "color": "#E8ECEF"},
        "xaxis": {
            "gridcolor": "#2A3441",
            "linecolor": "#2A3441",
            "zerolinecolor": "#2A3441",
        },
        "yaxis": {
            "gridcolor": "#2A3441",
            "linecolor": "#2A3441",
            "zerolinecolor": "#2A3441",
        },
        "legend": {"bgcolor": "rgba(0,0,0,0)"},
        "colorway": [
            "#2DD4A7", "#4D9FE5", "#E8A33D", "#E5484D",
            "#9D7BE5", "#5AC8E8", "#E87BB3", "#8FE82D",
        ],
    }
}
