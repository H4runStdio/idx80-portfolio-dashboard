"""
Section 5 — Informasi Penyusun.
"""

import streamlit as st

from utils.styling import render_section_heading, panel_start, panel_end, render_hairline


def render() -> None:
    render_section_heading(
        eyebrow="Section 05",
        icon="bi-person-badge",
        title="Informasi Penyusun",
        description="Dashboard ini disusun sebagai bagian dari proyek akademik di Institut Teknologi Sepuluh Nopember.",
    )

    PENYUSUN = [
        {
            "nama": "Haruna Lufni",
            "nim": "2043231065",
            "institusi": "Institut Teknologi Sepuluh Nopember",
            "email": "harunalufni@gmail.com",
        },
        {
            "nama": "Jadd Radyn Surya Trivisia",
            "nim": "2043231085",
            "institusi": "Institut Teknologi Sepuluh Nopember",
            "email": "jaddradynsurya@gmail.com",
        },
    ]

    for person in PENYUSUN:
        col_foto, col_info = st.columns([1, 2])

        with col_foto:
            panel_start("Foto Profil", "bi-image")
            st.markdown(
                """
                <div style="
                    height: 220px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: rgba(255,255,255,0.03);
                    border-radius: 10px;
                    margin-bottom: 0.5rem;
                ">
                    <i class="bi bi-person-circle" style="font-size: 5rem; color: #2A3441;"></i>
                </div>
                """,
                unsafe_allow_html=True,
            )
            panel_end()

        with col_info:
            panel_start("Identitas", "bi-card-text")
            st.markdown(
                f"""
                <table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
                    <tr style="border-bottom:1px solid #1E3A5C;">
                        <td style="padding:0.65rem 0.5rem; color:#5C7393; width:40%;">
                            <i class="bi bi-person" style="margin-right:0.5rem; color:#60A5FA;"></i>Nama
                        </td>
                        <td style="padding:0.65rem 0.5rem; color:#F1F5F9; font-weight:500;">{person["nama"]}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #1E3A5C;">
                        <td style="padding:0.65rem 0.5rem; color:#5C7393;">
                            <i class="bi bi-mortarboard" style="margin-right:0.5rem; color:#60A5FA;"></i>Institusi
                        </td>
                        <td style="padding:0.65rem 0.5rem; color:#F1F5F9;">{person["institusi"]}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #1E3A5C;">
                        <td style="padding:0.65rem 0.5rem; color:#5C7393;">
                            <i class="bi bi-hash" style="margin-right:0.5rem; color:#60A5FA;"></i>NIM
                        </td>
                        <td style="padding:0.65rem 0.5rem; color:#F1F5F9; font-family:'IBM Plex Mono', monospace;">{person["nim"]}</td>
                    </tr>
                    <tr>
                        <td style="padding:0.65rem 0.5rem; color:#5C7393;">
                            <i class="bi bi-envelope" style="margin-right:0.5rem; color:#60A5FA;"></i>Email
                        </td>
                        <td style="padding:0.65rem 0.5rem;">
                            <a href="mailto:{person["email"]}" style="color:#60A5FA; text-decoration:none; font-size:0.88rem;">
                                {person["email"]}
                            </a>
                        </td>
                    </tr>
                </table>
                """,
                unsafe_allow_html=True,
            )
            panel_end()

        render_hairline()

    st.markdown(
        """
        <div style="text-align:center; margin-top:1rem; color:#5C7393; font-size:0.8rem;">
            <i class="bi bi-mortarboard" style="margin-right:0.4rem;"></i>
            Institut Teknologi Sepuluh Nopember &mdash; Surabaya
        </div>
        """,
        unsafe_allow_html=True,
    )
