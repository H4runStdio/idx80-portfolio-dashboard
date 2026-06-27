"""
IDX80 Portfolio Intelligence Dashboard
Implementasi XGBoost dan Black-Litterman Model untuk Optimasi
Portofolio Saham pada Indeks IDX80.

Entry point aplikasi Streamlit. Mengatur navigasi sidebar antar lima
section dan menyuntikkan stylesheet global di awal sesi.
"""

import streamlit as st

from sections import (
    section1_overview,
    section2_prediction,
    section3_optimization,
    section4_simulation,
    section5_about,
)
from utils.styling import inject_global_styles, render_app_header, render_footer

st.set_page_config(
    page_title="IDX80 Portfolio Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={} 
)

inject_global_styles()

NAV_ITEMS = [
    {"key": "section1", "label": "Pemilihan Saham & Overview", "icon": "bi-collection", "module": section1_overview},
    {"key": "section2", "label": "Prediksi Harga (XGBoost)", "icon": "bi-graph-up", "module": section2_prediction},
    {"key": "section3", "label": "Optimasi Portofolio (BL)", "icon": "bi-diagram-3", "module": section3_optimization},
    {"key": "section4", "label": "Simulasi & Manajemen Risiko", "icon": "bi-shield-exclamation", "module": section4_simulation},
    {"key": "section5", "label": "Informasi Penyusun", "icon": "bi-person-badge", "module": section5_about},
]

if "active_section" not in st.session_state:
    st.session_state["active_section"] = "section1"

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.6rem; padding:0.4rem 0 1.1rem 0;">
            <i class="bi bi-bar-chart-line-fill" style="font-size:1.4rem; color:#2DD4A7;"></i>
            <div>
                <div style="font-weight:600; font-size:0.95rem; color:#E8ECEF;">IDX80 Intelligence</div>
                <div style="font-size:0.72rem; color:#5A6776;">Portfolio Optimization Suite</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_labels = [item["label"] for item in NAV_ITEMS]
    current_index = next(i for i, item in enumerate(NAV_ITEMS) if item["key"] == st.session_state["active_section"])

    selected_label = st.radio(
        "Navigasi",
        options=nav_labels,
        index=current_index,
        label_visibility="collapsed",
        key="sidebar_nav_radio",
    )
    selected_key = next(item["key"] for item in NAV_ITEMS if item["label"] == selected_label)
    if selected_key != st.session_state["active_section"]:
        st.session_state["active_section"] = selected_key
        st.rerun()

    st.markdown('<hr style="border-color:#2A3441; margin:1.2rem 0;">', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size:0.74rem; color:#5A6776; line-height:1.6;">
            <i class="bi bi-database" style="margin-right:0.3rem;"></i>
            80 saham konstituen IDX80<br>
            <i class="bi bi-calendar3" style="margin-right:0.3rem;"></i>
            Periode 2021&ndash;2026
        </div>
        """,
        unsafe_allow_html=True,
    )

render_app_header()

active = st.session_state["active_section"]
active_module = next(item["module"] for item in NAV_ITEMS if item["key"] == active)
active_module.render()

render_footer()
