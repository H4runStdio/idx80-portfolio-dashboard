"""
Section 5 — Informasi Penyusun.

Berisi placeholder profil penyusun dashboard serta ringkasan metodologi
dan tech stack yang digunakan. Seluruh teks bertanda [ISI DI SINI]
dimaksudkan untuk diganti langsung oleh penyusun.
"""

import streamlit as st

from utils.styling import (
    panel_end,
    panel_start,
    render_section_heading,
)


def render() -> None:
    render_section_heading(
        eyebrow="Section 05",
        icon="bi-person-badge",
        title="Informasi Penyusun",
        description="Profil penyusun dan ringkasan metodologi yang digunakan dalam dashboard ini.",
    )

    col_photo, col_info = st.columns([1, 2.4])

    with col_photo:
        panel_start("Foto Profil", "bi-image")
        st.markdown(
            """
            <div style="width:100%; aspect-ratio:1; border-radius:10px;
                        background:rgba(45,212,167,0.08); border:1px dashed #2A3441;
                        display:flex; align-items:center; justify-content:center;">
                <i class="bi bi-person" style="font-size:3rem; color:#5A6776;"></i>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("[Ganti dengan foto profil]")
        panel_end()

    with col_info:
        panel_start("Identitas", "bi-card-text")
        info_rows = [
            ("bi-person-fill", "Nama", "[ISI DI SINI]"),
            ("bi-mortarboard", "Institusi / Program Studi", "[ISI DI SINI]"),
            ("bi-hash", "NIM / Identitas", "[ISI DI SINI]"),
            ("bi-envelope", "Email", "[ISI DI SINI]"),
            ("bi-linkedin", "LinkedIn", "[ISI DI SINI]"),
            ("bi-github", "GitHub", "[ISI DI SINI]"),
        ]
        for icon, label, value in info_rows:
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:0.7rem; padding:0.55rem 0;
                            border-bottom:1px solid #2A3441;">
                    <i class="bi {icon}" style="color:#2DD4A7; width:20px;"></i>
                    <span style="color:#8B98A5; min-width:170px; font-size:0.88rem;">{label}</span>
                    <span style="color:#E8ECEF; font-size:0.9rem;">{value}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        panel_end()

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        panel_start("Ringkasan Proyek", "bi-journal-text")
        st.markdown(
            """
            <p style="color:#8B98A5; font-size:0.88rem; line-height:1.7;">
            [ISI DI SINI — uraikan secara singkat latar belakang dan tujuan proyek,
            misalnya: dashboard ini dikembangkan sebagai studi penerapan machine
            learning pada peramalan harga saham yang diintegrasikan dengan model
            optimasi portofolio Black-Litterman, menggunakan data historis saham
            anggota indeks IDX80.]
            </p>
            """,
            unsafe_allow_html=True,
        )
        panel_end()

    with col_right:
        panel_start("Metodologi & Tech Stack", "bi-cpu")
        stack_items = [
            ("bi-diagram-2", "XGBoost", "Prediksi harga / return saham"),
            ("bi-calculator", "Black-Litterman Model", "Optimasi alokasi portofolio"),
            ("bi-window", "Streamlit", "Kerangka antarmuka dashboard"),
            ("bi-bar-chart", "Plotly", "Visualisasi data interaktif"),
            ("bi-table", "Pandas / NumPy", "Pengolahan data"),
            ("bi-broadcast", "Ngrok", "Publikasi dashboard"),
        ]
        for icon, name, desc in stack_items:
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:0.7rem; padding:0.5rem 0;">
                    <i class="bi {icon}" style="color:#4D9FE5; font-size:1.05rem; width:20px;"></i>
                    <div>
                        <div style="color:#E8ECEF; font-size:0.88rem; font-weight:500;">{name}</div>
                        <div style="color:#5A6776; font-size:0.78rem;">{desc}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        panel_end()

    st.markdown("<br>", unsafe_allow_html=True)
    panel_start("Sumber Data", "bi-database")
    st.markdown(
        """
        <p style="color:#8B98A5; font-size:0.85rem; line-height:1.6;">
        Data harga penutupan dan volume harian 80 saham konstituen indeks IDX80
        periode 8 Juni 2021 sampai 5 Juni 2026. [ISI DI SINI — sebutkan sumber
        data asli, misalnya nama penyedia data atau platform yang digunakan
        untuk mengunduh data historis.]
        </p>
        """,
        unsafe_allow_html=True,
    )
    panel_end()
