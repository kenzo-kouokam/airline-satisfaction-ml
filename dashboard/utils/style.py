# -*- coding: utf-8 -*-
"""Shared corporate visual identity for the dashboard: colors, CSS, header/footer."""
import streamlit as st

NAVY = "#1E2761"
NAVY_DARK = "#141B47"
ICE = "#CADCFC"
ACCENT = "#FF8552"
GREEN = "#1F9D55"
RED = "#DC2626"
AMBER = "#F59E0B"
BG_CARD = "#F3F6FC"
TEXT_MUTED = "#6B7280"
TEXT_DARK = "#1F2937"

PLOTLY_TEMPLATE_COLORS = [NAVY, ACCENT, "#5B7BD5", "#8FA8E8", NAVY_DARK]


def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, sans-serif;
        }}

        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header[data-testid="stHeader"] {{background: transparent;}}

        section[data-testid="stSidebar"] {{
            background-color: {NAVY};
        }}
        section[data-testid="stSidebar"] * {{
            color: #E7EDFB !important;
        }}
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stMultiSelect label,
        section[data-testid="stSidebar"] .stSlider label {{
            color: #CADCFC !important;
            font-weight: 600;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: #3B4A8C;
        }}

        h1, h2, h3 {{
            color: {NAVY};
            font-weight: 700 !important;
        }}

        div[data-testid="stMetric"] {{
            background-color: {BG_CARD};
            border-radius: 10px;
            padding: 14px 16px 10px 16px;
        }}
        div[data-testid="stMetricLabel"] {{
            color: {TEXT_MUTED};
            font-weight: 600;
        }}
        div[data-testid="stMetricValue"] {{
            color: {NAVY};
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border-radius: 12px !important;
        }}

        .corp-header {{
            background: linear-gradient(135deg, {NAVY} 0%, {NAVY_DARK} 100%);
            padding: 28px 32px;
            border-radius: 14px;
            margin-bottom: 22px;
        }}
        .corp-header h1 {{
            color: white !important;
            margin: 0;
            font-size: 1.7rem;
        }}
        .corp-header p {{
            color: {ICE};
            margin: 6px 0 0 0;
            font-size: 0.95rem;
        }}
        .corp-badge {{
            display: inline-block;
            background-color: rgba(255,255,255,0.12);
            color: {ICE};
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            margin-right: 6px;
        }}
        .risk-pill {{
            display: inline-block;
            padding: 4px 14px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.85rem;
        }}
        .footer-note {{
            color: {TEXT_MUTED};
            font-size: 0.78rem;
            text-align: center;
            margin-top: 40px;
            padding-top: 14px;
            border-top: 1px solid #E5E7EB;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, badges: list[str] | None = None):
    badges_html = "".join(f'<span class="corp-badge">{b}</span>' for b in (badges or []))
    st.markdown(
        f"""
        <div class="corp-header">
            <div>{badges_html}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(
        """
        <div class="footer-note">
            Airline Satisfaction Intelligence — Outil d'aide à la décision construit à partir du projet Rush 3
            (régression logistique &amp; KNN) · Cedric Enzo KOUOKAM KAMHOUA
        </div>
        """,
        unsafe_allow_html=True,
    )
