"""
Section 5 — Informasi Penyusun.
"""

import streamlit as st

from utils.styling import render_section_heading, panel_start, panel_end


def render() -> None:
    render_section_heading(
        eyebrow="Section 05",
        icon="bi-person-badge",
        title="Informasi Penyusun",
        description="Dashboard ini disusun sebagai bagian dari proyek akademik di Institut Teknologi Sepuluh Nopember.",
    )

    for person in [
        {
            "nama": "Haruna Lufni",
            "nim": "2043231065",
            "email": "harunalufni@gmail.com",
        },
        {
            "nama": "Jadd Radyn Surya Trivisia",
            "nim": "2043231085",
            "email": "jaddradynsurya@gmail.com",
        },
    ]:
        panel_start(person["nama"], "bi-person-circle")
        st.markdown(
            f"""
            <div style="display:flex; flex-direction:column; gap:0.6rem; padding:0.2rem 0 0.4rem 0;">
                <div style="display:flex; align-items:center; gap:0.75rem;">
                    <i class="bi bi-building" style="color:#60A5FA; font-size:1rem; width:18px;"></i>
                    <span style="color:#93A8C3; font-size:0.85rem;">Institut Teknologi Sepuluh Nopember</span>
                </div>
                <div style="display:flex; align-items:center; gap:0.75rem;">
                    <i class="bi bi-credit-card-2-front" style="color:#60A5FA; font-size:1rem; width:18px;"></i>
                    <span style="color:#93A8C3; font-size:0.85rem;">{person["nim"]}</span>
                </div>
                <div style="display:flex; align-items:center; gap:0.75rem;">
                    <i class="bi bi-envelope" style="color:#60A5FA; font-size:1rem; width:18px;"></i>
                    <a href="mailto:{person["email"]}" style="color:#60A5FA; font-size:0.85rem; text-decoration:none;">
                        {person["email"]}
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        panel_end()

    st.markdown(
        """
        <div style="text-align:center; margin-top:2rem; color:#5C7393; font-size:0.8rem; line-height:1.8;">
            <i class="bi bi-mortarboard" style="margin-right:0.4rem;"></i>
            Institut Teknologi Sepuluh Nopember &mdash; Surabaya
        </div>
        """,
        unsafe_allow_html=True,
    )
