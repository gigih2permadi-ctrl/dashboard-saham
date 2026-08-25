import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import statsmodels.api as sm
from datetime import date, datetime, timedelta
import textwrap
from io import BytesIO
import requests
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup


def render_html(html):
    """Render custom HTML safely as Streamlit markdown."""
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)


def render_table(df, max_height=None):
    """Render a pandas DataFrame as a dark, readable HTML table."""
    if df is None or df.empty:
        st.info("Tidak ada data untuk ditampilkan.")
        return

    table = df.copy()
    html_table = table.to_html(
        index=False,
        escape=True,
        classes="dark-table",
        border=0
    )

    height_style = ""
    if max_height:
        height_style = f"max-height:{int(max_height)}px;overflow:auto;"

    render_html(
        f"""
        <div class=\"table-wrap\" style=\"{height_style}\">
            {html_table}
        </div>
        """
    )



# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Stock Analysis — Global Market Terminal v14",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# DARK AI DASHBOARD STYLE
# =========================================================

st.markdown("""
<style>

/* =====================================================
   GLOBAL
   ===================================================== */

html, body, [class*="css"] {
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 15% 0%,
            rgba(0, 230, 118, 0.035),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(0, 184, 255, 0.035),
            transparent 30%
        ),
        #080b10;
    color: #e6edf3;
}


/* =====================================================
   MAIN CONTAINER
   ===================================================== */

.block-container {
    padding-top: 1.0rem;
    padding-bottom: 1.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1700px;
}


/* =====================================================
   SIDEBAR
   ===================================================== */

section[data-testid="stSidebar"] {
    background: #0b0f14;
    border-right: 1px solid #202833;
}

section[data-testid="stSidebar"] * {
    color: #d8dee6;
}

section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stDateInput input {
    background: #11161d;
    color: #ffffff;
    border: 1px solid #28313d;
    border-radius: 8px;
}

section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
    background: #11161d;
    border-radius: 8px;
}


/* =====================================================
   HEADINGS
   ===================================================== */

h1 {
    color: #f5f7fa !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

h2 {
    color: #f5f7fa !important;
    font-size: 1.35rem !important;
    margin-top: 0.7rem !important;
    margin-bottom: 0.45rem !important;
}

h3 {
    color: #dce3ea !important;
    font-size: 1.05rem !important;
    margin-top: 0.45rem !important;
    margin-bottom: 0.35rem !important;
}


/* =====================================================
   COMPACT VERTICAL SPACING
   ===================================================== */

div[data-testid="stVerticalBlock"] {
    gap: 0.35rem;
}

div[data-testid="stHorizontalBlock"] {
    gap: 0.65rem;
}


/* =====================================================
   METRIC CARDS
   ===================================================== */

div[data-testid="metric-container"] {
    background:
        linear-gradient(
            145deg,
            #131921,
            #0e1319
        );

    border: 1px solid #232d38;
    border-radius: 12px;

    padding:
        0.75rem
        0.9rem;

    min-height: 82px;

    box-shadow:
        0 4px 18px rgba(0,0,0,0.18);

    transition:
        transform 0.15s ease,
        border-color 0.15s ease;
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-1px);
    border-color: #344252;
}

div[data-testid="stMetricLabel"] {
    color: #7f8b99 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

div[data-testid="stMetricValue"] {
    color: #f5f7fa !important;
    font-size: 1.35rem !important;
    font-weight: 700;
}


/* =====================================================
   DATAFRAME
   ===================================================== */

[data-testid="stDataFrame"] {
    border: 1px solid #252e39;
    border-radius: 10px;
    overflow: hidden;
}


/* =====================================================
   DARK HTML TABLES
   ===================================================== */

.table-wrap {
    width: 100%;
    overflow-x: auto;
    overflow-y: auto;
    background: #0d131a;
    border: 1px solid #26313d;
    border-radius: 10px;
    margin: 4px 0 10px 0;
    box-shadow: 0 5px 18px rgba(0,0,0,0.16);
}

.dark-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    color: #dce3ea;
    font-size: 0.78rem;
    background: #0d131a;
}

.dark-table thead th {
    position: sticky;
    top: 0;
    z-index: 2;
    background: #151d26;
    color: #9eabb8;
    font-weight: 700;
    text-align: left;
    text-transform: none;
    letter-spacing: 0.15px;
    padding: 8px 10px;
    border-bottom: 1px solid #2b3744;
    white-space: nowrap;
}

.dark-table tbody td {
    background: #0d131a;
    color: #dce3ea;
    padding: 7px 10px;
    border-bottom: 1px solid #1b2530;
    white-space: nowrap;
}

.dark-table tbody tr:nth-child(even) td {
    background: #10171f;
}

.dark-table tbody tr:hover td {
    background: #17212b;
}

.dark-table tbody tr:last-child td {
    border-bottom: none;
}

/* =====================================================
   ALERT / INFO
   ===================================================== */

.stAlert {
    background: #10161d;
    border: 1px solid #26313d;
    border-radius: 10px;
}


/* =====================================================
   BUTTON
   ===================================================== */

.stButton > button {
    background: #00c853;
    color: #061009;
    border: none;
    border-radius: 8px;
    font-weight: 700;
}

.stButton > button:hover {
    background: #00e676;
}


/* =====================================================
   EXPANDER
   ===================================================== */

[data-testid="stExpander"] {
    background: #0e1319;
    border: 1px solid #232d38;
    border-radius: 10px;
}


/* =====================================================
   CAPTION
   ===================================================== */

.stCaption {
    color: #697685 !important;
}


/* =====================================================
   CUSTOM DASHBOARD CARDS
   ===================================================== */

.ai-card {
    background:
        linear-gradient(
            145deg,
            #121820,
            #0c1117
        );

    border:
        1px solid #222c37;

    border-radius: 14px;

    padding:
        16px 18px;

    margin:
        2px 0 8px 0;

    box-shadow:
        0 8px 28px rgba(0,0,0,0.18);
}

.ai-card-title {
    color: #7f8b99;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 5px;
}

.ai-card-value {
    color: #f4f7fa;
    font-size: 1.45rem;
    font-weight: 700;
}

.ai-card-sub {
    color: #697685;
    font-size: 0.75rem;
    margin-top: 4px;
}


/* =====================================================
   AI SIGNAL
   ===================================================== */

.signal-buy {
    background:
        linear-gradient(
            135deg,
            rgba(0,230,118,0.16),
            rgba(0,230,118,0.035)
        );

    border: 1px solid rgba(0,230,118,0.35);

    border-radius: 14px;

    padding: 18px;

    box-shadow:
        0 0 25px rgba(0,230,118,0.06);
}

.signal-wait {
    background:
        linear-gradient(
            135deg,
            rgba(255,193,7,0.15),
            rgba(255,193,7,0.03)
        );

    border: 1px solid rgba(255,193,7,0.3);

    border-radius: 14px;

    padding: 18px;
}

.signal-avoid {
    background:
        linear-gradient(
            135deg,
            rgba(255,69,96,0.15),
            rgba(255,69,96,0.03)
        );

    border: 1px solid rgba(255,69,96,0.3);

    border-radius: 14px;

    padding: 18px;
}

.signal-title {
    color: #778391;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.signal-value {
    color: #ffffff;
    font-size: 1.7rem;
    font-weight: 800;
    margin-top: 5px;
}


/* =====================================================
   TRADING PLAN
   ===================================================== */

.trade-card {
    background: linear-gradient(145deg, #121a22, #0d1319);
    border: 1px solid #2a3542;
    border-radius: 12px;
    padding: 14px 12px;
    text-align: center;
    min-height: 72px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    overflow: hidden;
    box-shadow: 0 5px 16px rgba(0,0,0,0.16);
}

.trade-label {
    color: #8b98a6;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.45px;
}

.trade-value {
    color: #f5f7fa;
    font-size: 1.15rem;
    font-weight: 750;
    margin-top: 5px;
    white-space: nowrap;
    line-height: 1.2;
}


/* =====================================================
   SECTION HEADER
   ===================================================== */

.section-header {
    display: flex;
    align-items: center;
    gap: 9px;
    margin-top: 13px;
    margin-bottom: 5px;
}

.section-line {
    height: 1px;
    flex: 1;
    background: #202a34;
}


/* =====================================================
   TOP HEADER
   ===================================================== */

.dashboard-header {
    background:
        linear-gradient(
            135deg,
            #121a22,
            #0c1117
        );

    border:
        1px solid #26313d;

    border-radius:
        15px;

    padding:
        16px 20px;

    margin-bottom:
        10px;

    box-shadow:
        0 8px 30px rgba(0,0,0,0.18);
}

.dashboard-title {
    font-size: 1.65rem;
    font-weight: 800;
    color: #f5f7fa;
}

.dashboard-subtitle {
    color: #6f7c89;
    font-size: 0.78rem;
    margin-top: 3px;
}

.live-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    background: #00e676;
    border-radius: 50%;
    box-shadow: 0 0 10px #00e676;
    margin-right: 6px;
}


/* =====================================================
   SECTION / RESPONSIVE POLISH
   ===================================================== */

.section-header b {
    color: #e8eef4;
    font-size: 0.82rem;
    letter-spacing: 0.45px;
}

[data-testid="stHorizontalBlock"] > div {
    min-width: 0;
}

@media (max-width: 900px) {
    .block-container {
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }

    .dashboard-title {
        font-size: 1.35rem;
    }

    .trade-value {
        font-size: 0.95rem;
    }
}

/* =====================================================
   FOOTER
   ===================================================== */

/* =====================================================
   GLOBAL vs IDX SECTOR COMPARISON
   ===================================================== */
.sector-panel {background:linear-gradient(180deg,#0d151e,#0a1017);border:1px solid #263544;border-radius:10px;overflow:hidden;box-shadow:0 8px 24px rgba(0,0,0,.18);margin-bottom:8px;}
.sector-panel-head {display:flex;justify-content:space-between;align-items:center;gap:12px;padding:10px 12px 8px;border-bottom:1px solid #1d2a37;}
.sector-panel-title {font-size:14px;font-weight:800;color:#e7edf3;letter-spacing:.2px;}
.sector-panel-sub {font-size:9px;color:#748394;margin-top:3px;}
.sector-summary {font-size:9px;color:#8d9bab;white-space:nowrap;}
.sector-summary b {color:#dbe5ed;}
.sector-header-row,.sector-row {display:grid;grid-template-columns:42px minmax(170px,1fr) 58px 58px 58px 78px;align-items:center;column-gap:5px;}
.sector-header-row {padding:7px 9px;background:#111b25;color:#7f8e9d;font-size:8px;font-weight:800;letter-spacing:.45px;}
.sector-row {min-height:43px;padding:5px 9px;border-top:1px solid #1b2733;}
.sector-row:hover {background:#111b24;}
.sector-rank {color:#93a1ae;font-size:9px;font-weight:700;}
.sector-name {min-width:0;line-height:1.1;}
.sector-name b {display:block;color:#e6edf4;font-size:9px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.sector-name small {display:block;color:#667687;font-size:7px;margin-top:3px;}
.sector-ret {font-size:9px;font-weight:800;text-align:right;color:#cdd8e1;}
.sector-status {justify-self:end;border:1px solid;border-radius:999px;padding:3px 6px;font-size:7px;font-weight:800;letter-spacing:.25px;}
.sector-empty {padding:22px 14px;color:#7d8b99;font-size:10px;}
@media (max-width:1000px) {.sector-header-row,.sector-row {grid-template-columns:34px minmax(120px,1fr) 50px 50px 50px 70px;}.sector-panel-head {align-items:flex-start;}.sector-summary {display:none;}}

.footer {
    color: #4e5a67;
    text-align: center;
    font-size: 0.7rem;
    padding: 12px;
    border-top: 1px solid #18212b;
    margin-top: 20px;
}



/* =====================================================
   PREMIUM ANALYTICS CARDS
   ===================================================== */
.analytics-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; margin:2px 0 8px; }
.analytics-card { position:relative; overflow:hidden; background:linear-gradient(145deg,#121a23 0%,#0b1016 100%); border:1px solid #263342; border-radius:14px; padding:14px 15px; min-height:112px; box-shadow:0 8px 24px rgba(0,0,0,.20); }
.analytics-card:before { content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background:#2b3643; }
.analytics-card.green:before { background:#00e676; box-shadow:0 0 16px rgba(0,230,118,.35); }
.analytics-card.blue:before { background:#29b6f6; box-shadow:0 0 16px rgba(41,182,246,.25); }
.analytics-card.purple:before { background:#ab47bc; box-shadow:0 0 16px rgba(171,71,188,.25); }
.analytics-kicker { color:#7f8c9a; font-size:.66rem; text-transform:uppercase; letter-spacing:.8px; font-weight:700; }
.analytics-number { color:#f5f7fa; font-size:1.65rem; line-height:1.05; font-weight:800; margin-top:7px; }
.analytics-number.green { color:#00e676; }
.analytics-number.blue { color:#29b6f6; }
.analytics-number.yellow { color:#ffd54f; }
.analytics-note { color:#687584; font-size:.68rem; margin-top:7px; }
.analytics-icon { position:absolute; right:13px; top:11px; font-size:1.25rem; opacity:.85; }
.ai-engine { background:linear-gradient(145deg,#111a22,#0a1016); border:1px solid #263442; border-radius:14px; padding:14px; box-shadow:0 8px 26px rgba(0,0,0,.20); }
.ai-engine-top { display:grid; grid-template-columns:100px 1fr; gap:10px; margin-bottom:11px; }
.score-orb { min-height:104px; border-radius:13px; display:flex; flex-direction:column; align-items:center; justify-content:center; background:radial-gradient(circle at 50% 25%,rgba(0,230,118,.12),transparent 60%),#0c131a; border:1px solid #263542; }
.score-orb .score { font-size:1.8rem; font-weight:850; color:#00e676; line-height:1; }
.score-orb .label { font-size:.62rem; color:#7f8c99; margin-top:6px; text-transform:uppercase; letter-spacing:.6px; }
.signal-panel { border:1px solid #263542; border-radius:13px; padding:13px 14px; background:#0d141b; }
.signal-panel .small { color:#788694; font-size:.65rem; text-transform:uppercase; letter-spacing:.7px; }
.signal-pill { display:inline-flex; align-items:center; gap:7px; margin-top:10px; padding:7px 11px; border-radius:999px; background:rgba(0,230,118,.10); border:1px solid rgba(0,230,118,.28); color:#63f0a0; font-size:.74rem; font-weight:800; }
.signal-pill.wait { background:rgba(255,193,7,.10); border-color:rgba(255,193,7,.28); color:#ffd54f; }
.signal-pill.bear { background:rgba(255,69,96,.10); border-color:rgba(255,69,96,.28); color:#ff7185; }
.reason-list { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
.reason-item { display:flex; align-items:center; gap:7px; padding:7px 8px; border:1px solid #202c37; border-radius:9px; background:#0d141b; color:#b5c0ca; font-size:.68rem; }
.reason-item.ok { color:#d9fbe8; border-color:#254636; }
.reason-dot { width:8px; height:8px; border-radius:50%; flex:none; background:#596572; }
.reason-item.ok .reason-dot { background:#00e676; box-shadow:0 0 7px rgba(0,230,118,.55); }
.threshold-card { position:relative; overflow:hidden; background:linear-gradient(145deg,#17150e,#0d1115); border:1px solid #5a4b1b; border-radius:14px; padding:15px; min-height:100%; box-shadow:0 8px 26px rgba(0,0,0,.20); }
.threshold-card:after { content:""; position:absolute; width:120px;height:120px; right:-40px;top:-50px; border-radius:50%; background:rgba(255,193,7,.05); }
.threshold-badge { display:inline-flex; padding:4px 8px; border-radius:999px; background:rgba(255,193,7,.11); color:#ffd54f; border:1px solid rgba(255,193,7,.25); font-size:.62rem; font-weight:800; text-transform:uppercase; letter-spacing:.5px; }
.threshold-title { color:#f0e8c7; font-size:.78rem; font-weight:750; margin-top:9px; }
.threshold-number { color:#ffd54f; font-size:2.2rem; font-weight:900; line-height:1; margin:4px 0 12px; }
.threshold-sub { color:#7f8a95; font-size:.65rem; }
.threshold-stats { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:12px; }
.threshold-stat { border-top:1px solid #2b2a20; padding-top:7px; }
.threshold-stat .lbl { color:#737f8b; font-size:.6rem; text-transform:uppercase; }
.threshold-stat .val { color:#f5f7fa; font-size:.88rem; font-weight:800; margin-top:2px; }
.threshold-stat .positive { color:#00e676; }
.threshold-stat .negative { color:#ff5b70; }
.stat-signal-wrap { display:grid; grid-template-columns:1fr 1fr 1fr; gap:9px; }
.stat-card { background:linear-gradient(145deg,#111a22,#0b1117); border:1px solid #253340; border-radius:12px; padding:12px 13px; }
.stat-card .lbl { color:#7c8997; font-size:.62rem; text-transform:uppercase; letter-spacing:.65px; }
.stat-card .val { color:#00e676; font-size:1.45rem; font-weight:850; margin-top:5px; }
.stat-card .sub { color:#687582; font-size:.62rem; margin-top:3px; }
.stat-card.blue .val { color:#29b6f6; }
.stat-card.purple .val { color:#c77dff; }
@media(max-width:900px){ .stat-signal-wrap { grid-template-columns:1fr; } .reason-list { grid-template-columns:1fr; } .ai-engine-top { grid-template-columns:85px 1fr; } }

/* V3 GRID POLISH */
.v3-small { color:#8a96a3; font-size:.72rem; }
.v3-table table { width:100%; border-collapse:collapse; background:#0d131a; color:#dce4eb; font-size:12px; }
.v3-table th { background:#121a22; color:#8f9cab; padding:7px 8px; border-bottom:1px solid #27323e; text-align:left; }
.v3-table td { padding:7px 8px; border-bottom:1px solid #1d2731; }
.v3-table-wrap { overflow-x:auto; border:1px solid #202b37; border-radius:8px; }
.ai-card, .signal-buy, .signal-wait, .signal-avoid { min-height:92px; }
.trade-card { min-height:68px; }
[data-testid="stMetricValue"] { white-space:nowrap; }
div[data-testid="stPlotlyChart"] { margin-top:-2px; }


/* =====================================================
   SECTOR MARKET CONDITION
   ===================================================== */
.sector-focus-grid {display:grid;grid-template-columns:1.35fr 0.85fr 0.95fr 1.35fr;gap:10px;margin:8px 0 10px 0;}
.sector-focus-card {background:linear-gradient(145deg,#0f1720,#0b1118);border:1px solid #273443;border-radius:10px;padding:12px 13px;min-height:92px;box-shadow:0 4px 18px rgba(0,0,0,.12);}
.sector-focus-name {color:#f5f7fa;font-size:1.05rem;font-weight:750;margin-top:7px;}
.sector-focus-value {color:#f5f7fa;font-size:1.25rem;font-weight:800;margin-top:7px;}
.sector-progress {height:7px;background:#202a34;border-radius:7px;margin-top:9px;overflow:hidden;}
.sector-progress span {display:block;height:100%;border-radius:7px;}
.sector-row {display:grid;grid-template-columns:2.1fr .55fr .8fr .85fr .75fr .8fr 1.25fr;gap:8px;align-items:center;padding:9px 10px;border-bottom:1px solid #202a34;font-size:.78rem;}
.sector-row:last-child {border-bottom:0;}
.sector-row:hover {background:#101923;}
@media (max-width:900px){.sector-focus-grid{grid-template-columns:1fr 1fr;}.sector-row{grid-template-columns:1.7fr .55fr .75fr .8fr .95fr 1.1fr;}.sector-row .sector-vol{display:none;}}


/* =====================================================
   GLOBAL MARKET TERMINAL / FOCUSED SECTOR ROTATION
   ===================================================== */
.global-regime-grid{display:grid;grid-template-columns:1.7fr repeat(7,1fr);gap:7px;margin:8px 0 10px}
.global-terminal-card{background:linear-gradient(145deg,#10161d,#0b1016);border:1px solid #26313d;border-radius:9px;padding:11px 10px;min-height:86px;box-shadow:0 3px 14px rgba(0,0,0,.18)}
.global-terminal-card.hero{border-color:#344252;background:linear-gradient(145deg,#121a22,#0b1016)}
.terminal-kicker{font-size:.60rem;letter-spacing:.8px;color:#718091;font-weight:800;text-transform:uppercase}
.terminal-big{font-size:1.20rem;font-weight:900;letter-spacing:.2px;margin-top:5px}
.terminal-value{font-size:.95rem;font-weight:800;color:#e8edf2;margin-top:5px}.terminal-change{font-size:.67rem;color:#8b98a7;margin-top:4px}.terminal-sub{font-size:.66rem;color:#7f8b99;line-height:1.4;margin-top:3px}
.rotation-chart-card{background:linear-gradient(145deg,#0d141b,#0a0f14);border:1px solid #263442;border-radius:11px;padding:6px 6px 2px;overflow:hidden}
.rotation-rank-card{background:linear-gradient(145deg,#101820,#0b1016);border:1px solid #263442;border-radius:11px;padding:12px;height:100%;box-sizing:border-box}
.rotation-rank-title{color:#f1f5f9;font-size:.78rem;font-weight:900;letter-spacing:.8px;text-transform:uppercase;margin-bottom:8px}
.rotation-rank-row{display:grid;grid-template-columns:30px minmax(0,1fr) auto;gap:7px;align-items:center;padding:8px 4px;border-bottom:1px solid #202b35}.rotation-rank-row:last-child{border-bottom:0}
.rotation-rank-num{color:#94a3b8;font-weight:900;font-size:.70rem}.rotation-rank-name{color:#f1f5f9;font-weight:850;font-size:.73rem;line-height:1.2}.rotation-rank-meta{color:#82909e;font-size:.60rem;margin-top:3px;line-height:1.2}
.rotation-status-pill{padding:4px 7px;border-radius:999px;font-size:.55rem;font-weight:900;letter-spacing:.25px;white-space:nowrap}
.rotation-caption{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin:8px 0 8px;padding:9px 12px;border:1px solid #202c38;border-radius:8px;background:#0c1218;color:#9aa8b6;font-size:.67rem;letter-spacing:.15px}.rotation-caption b{color:#f1f5f9}
.rotation-note{margin-top:8px;padding:8px 10px;border:1px solid #202c38;border-radius:8px;background:#0c1218;color:#8997a5;font-size:.63rem;line-height:1.45}
.global-focus-card{display:grid;grid-template-columns:2.4fr 1fr 1fr 1.2fr;gap:12px;align-items:center;margin-top:9px;padding:12px 14px;border:1px solid #2a3744;border-radius:10px;background:linear-gradient(100deg,#101820,#0c1117)}
.focus-title{font-size:1rem;font-weight:850;color:#f1f4f7;margin-top:4px}.focus-number{font-size:1.1rem;font-weight:900;color:#eef2f5;margin-top:4px}.focus-regime{font-size:.85rem;font-weight:900;margin-top:5px}
@media(max-width:1200px){.global-regime-grid{grid-template-columns:repeat(4,1fr)}.global-regime-grid .hero{grid-column:span 2}.global-focus-card{grid-template-columns:2fr 1fr 1fr 1fr}}
@media(max-width:900px){.global-regime-grid{grid-template-columns:repeat(2,1fr)}.global-focus-card{grid-template-columns:1fr 1fr}}
</style>
""", unsafe_allow_html=True)


# =========================================================
# TIMEFRAME CONFIG
# =========================================================

TIMEFRAME_CONFIG = {

    "5 Menit": {
        "interval": "5m",
        "max_days": 59,
        "default_days": 10,
        "label": "5 menit"
    },

    "15 Menit": {
        "interval": "15m",
        "max_days": 59,
        "default_days": 20,
        "label": "15 menit"
    },

    "1 Jam": {
        "interval": "1h",
        "max_days": 700,
        "default_days": 90,
        "label": "1 jam"
    },

    "1 Hari": {
        "interval": "1d",
        "max_days": None,
        "default_days": 365,
        "label": "1 hari"
    },

    "1 Minggu": {
        "interval": "1wk",
        "max_days": None,
        "default_days": 365 * 5,
        "label": "1 minggu"
    },

    "1 Bulan": {
        "interval": "1mo",
        "max_days": None,
        "default_days": 365 * 10,
        "label": "1 bulan"
    }
}


# =========================================================
# INDICATORS
# =========================================================

def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    return 100 - (
        100 / (1 + rs)
    )


def atr(data, period=14):

    hl = (
        data["High"] -
        data["Low"]
    )

    hc = (
        data["High"] -
        data["Close"].shift(1)
    ).abs()

    lc = (
        data["Low"] -
        data["Close"].shift(1)
    ).abs()

    tr = pd.concat(
        [hl, hc, lc],
        axis=1
    ).max(axis=1)

    return tr.rolling(
        period
    ).mean()


def cmo(series, period=14):

    delta = series.diff()

    up = delta.clip(
        lower=0
    )

    down = -delta.clip(
        upper=0
    )

    sum_up = (
        up.rolling(period)
        .sum()
    )

    sum_down = (
        down.rolling(period)
        .sum()
    )

    denominator = (
        sum_up + sum_down
    ).replace(
        0,
        np.nan
    )

    return (
        100 *
        (sum_up - sum_down) /
        denominator
    )


# =========================================================
# ADD INDICATORS
# =========================================================

def add_indicators(df):

    data = df.copy()

    if isinstance(
        data.columns,
        pd.MultiIndex
    ):

        data.columns = (
            data.columns
            .get_level_values(0)
        )

    data.columns = [
        str(x).title()
        for x in data.columns
    ]

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for col in required:

        if col not in data.columns:

            raise ValueError(
                f"Kolom {col} tidak tersedia."
            )

    for col in required:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    # MA

    for n in [
        5,
        10,
        20,
        50,
        100,
        200
    ]:

        data[f"MA{n}"] = (
            data["Close"]
            .rolling(n)
            .mean()
        )

    # EMA

    for n in [
        20,
        50,
        100
    ]:

        data[f"EMA{n}"] = (
            data["Close"]
            .ewm(
                span=n,
                adjust=False
            )
            .mean()
        )

    # RSI

    data["RSI14"] = rsi(
        data["Close"],
        14
    )

    # CMO

    data["CMO14"] = cmo(
        data["Close"],
        14
    )

    # MACD

    ema12 = (
        data["Close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        data["Close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    data["MACD"] = (
        ema12 - ema26
    )

    data["MACD_SIGNAL"] = (
        data["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    data["MACD_HIST"] = (
        data["MACD"] -
        data["MACD_SIGNAL"]
    )

    # Bollinger

    std20 = (
        data["Close"]
        .rolling(20)
        .std()
    )

    data["BB_UPPER"] = (
        data["MA20"] +
        2 * std20
    )

    data["BB_LOWER"] = (
        data["MA20"] -
        2 * std20
    )

    data["BB_POSITION"] = (
        (
            data["Close"] -
            data["BB_LOWER"]
        )
        /
        (
            data["BB_UPPER"] -
            data["BB_LOWER"]
        ).replace(
            0,
            np.nan
        )
    )

    # CMF

    hl = (
        data["High"] -
        data["Low"]
    ).replace(
        0,
        np.nan
    )

    mfm = (
        (
            data["Close"] -
            data["Low"]
        )
        -
        (
            data["High"] -
            data["Close"]
        )
    ) / hl

    mfv = (
        mfm *
        data["Volume"]
    )

    data["CMF20"] = (
        mfv.rolling(20).sum()
        /
        data["Volume"]
        .rolling(20).sum()
    )

    # Volume

    data["VOL_MA20"] = (
        data["Volume"]
        .rolling(20)
        .mean()
    )

    data["VOL_MEDIAN20"] = (
        data["Volume"]
        .rolling(20)
        .median()
    )

    data["VOL_RATIO"] = (
        data["Volume"] /
        data["VOL_MA20"].replace(
            0,
            np.nan
        )
    )

    # ATR

    data["ATR14"] = atr(
        data,
        14
    )

    data["ATR_PERCENT"] = (
        data["ATR14"] /
        data["Close"] *
        100
    )

    # Returns

    data["RETURN_1"] = (
        data["Close"].pct_change()
    )

    data["RETURN_5"] = (
        data["Close"].pct_change(5)
    )

    data["RETURN_20"] = (
        data["Close"].pct_change(20)
    )

    data["RETURN_BESOK"] = (
        data["Close"].shift(-1) /
        data["Close"] -
        1
    )

    # Score

    data["SCORE_RSI"] = np.where(
        data["RSI14"] > 50,
        1,
        0
    )

    data["SCORE_MACD"] = np.where(
        data["MACD"] >
        data["MACD_SIGNAL"],
        1,
        0
    )

    data["SCORE_CMF"] = np.where(
        data["CMF20"] > 0,
        1,
        0
    )

    data["SCORE_CMO"] = np.where(
        data["CMO14"] > 0,
        1,
        0
    )

    data["SCORE_TREND"] = np.where(
        data["Close"] >
        data["MA20"],
        1,
        0
    )

    data["SCORE_VOLUME"] = np.where(
        data["Volume"] >
        data["VOL_MEDIAN20"],
        1,
        0
    )

    score_cols = [
        "SCORE_RSI",
        "SCORE_MACD",
        "SCORE_CMF",
        "SCORE_CMO",
        "SCORE_TREND",
        "SCORE_VOLUME"
    ]

    data["SCORE_HISTORIS"] = (
        data[score_cols].sum(axis=1)
    )

    return data


# =========================================================
# SUPPORT RESISTANCE
# =========================================================

def cari_support_resistance(
    data,
    lookback=60
):

    d = data.tail(
        lookback
    ).copy()

    if d.empty:

        return {
            "support": np.nan,
            "support_low": np.nan,
            "support_high": np.nan,
            "resistance": np.nan,
            "resistance_low": np.nan,
            "resistance_high": np.nan,
            "atr": np.nan
        }

    harga = float(
        d["Close"].iloc[-1]
    )

    swing_low = (
        (d["Low"] < d["Low"].shift(1))
        &
        (d["Low"] < d["Low"].shift(-1))
    )

    swing_high = (
        (d["High"] > d["High"].shift(1))
        &
        (d["High"] > d["High"].shift(-1))
    )

    lows = (
        d.loc[
            swing_low,
            "Low"
        ].dropna()
    )

    highs = (
        d.loc[
            swing_high,
            "High"
        ].dropna()
    )

    support_candidates = (
        lows[lows < harga]
    )

    if not support_candidates.empty:

        support = float(
            support_candidates.iloc[
                np.argmin(
                    np.abs(
                        support_candidates -
                        harga
                    )
                )
            ]
        )

    else:

        support = float(
            d["Low"].min()
        )

    resistance_candidates = (
        highs[highs > harga]
    )

    if not resistance_candidates.empty:

        resistance = float(
            resistance_candidates.iloc[
                np.argmin(
                    np.abs(
                        resistance_candidates -
                        harga
                    )
                )
            ]
        )

    else:

        resistance = float(
            d["High"].max()
        )

    atr_now = d["ATR14"].iloc[-1]

    if pd.isna(atr_now):

        atr_now = harga * 0.02

    else:

        atr_now = float(
            atr_now
        )

    area = atr_now * 0.30

    return {

        "support": support,
        "support_low": support - area,
        "support_high": support + area,

        "resistance": resistance,
        "resistance_low": resistance - area,
        "resistance_high": resistance + area,

        "atr": atr_now
    }


# =========================================================
# TP SL
# =========================================================

def hitung_tp_sl(
    harga,
    support,
    resistance,
    atr_now
):

    if pd.isna(resistance):

        resistance = (
            harga + atr_now
        )

    if resistance <= harga:

        resistance = (
            harga + atr_now
        )

    tp1 = resistance

    jarak = (
        resistance - harga
    )

    tp2 = (
        resistance + jarak
    )

    sl1 = (
        support -
        0.30 * atr_now
    )

    sl2 = (
        support -
        0.70 * atr_now
    )

    return tp1, tp2, sl1, sl2


# =========================================================
# BACKTEST
# =========================================================

def jalankan_backtest(
    df,
    threshold
):

    data = df.copy()

    data["SIGNAL_BACKTEST"] = np.where(
        data["SCORE_HISTORIS"] >= threshold,
        "BUY",
        "WAIT"
    )

    data["RETURN_STRATEGI"] = np.where(

        (
            data["SIGNAL_BACKTEST"] ==
            "BUY"
        )
        &
        data["RETURN_BESOK"].notna(),

        data["RETURN_BESOK"],

        0.0
    )

    data["RETURN_STRATEGI"] = (
        data["RETURN_STRATEGI"]
        .fillna(0)
    )

    data["EQUITY_CURVE"] = (
        1 + data["RETURN_STRATEGI"]
    ).cumprod()

    if data.empty:

        return {
            "THRESHOLD": threshold,
            "JUMLAH_TRADE": 0,
            "WIN_RATE": 0,
            "AVG_RETURN": 0,
            "TOTAL_RETURN": 0,
            "MAX_DRAWDOWN": 0,
            "PROFIT_FACTOR": 0,
            "DATA": data
        }

    total_return = (
        data["EQUITY_CURVE"].iloc[-1] - 1
    )

    trades = data[
        (
            data["SIGNAL_BACKTEST"] ==
            "BUY"
        )
        &
        data["RETURN_BESOK"].notna()
    ]

    jumlah_trade = len(
        trades
    )

    if jumlah_trade:

        win_rate = (
            trades["RETURN_BESOK"] > 0
        ).mean()

        avg_return = (
            trades["RETURN_BESOK"].mean()
        )

    else:

        win_rate = 0
        avg_return = 0

    equity = data[
        "EQUITY_CURVE"
    ]

    running_max = equity.cummax()

    drawdown = (
        equity / running_max - 1
    )

    max_drawdown = drawdown.min()

    profit = trades.loc[
        trades["RETURN_BESOK"] > 0,
        "RETURN_BESOK"
    ].sum()

    loss = abs(
        trades.loc[
            trades["RETURN_BESOK"] < 0,
            "RETURN_BESOK"
        ].sum()
    )

    if loss > 0:

        profit_factor = (
            profit / loss
        )

    elif profit > 0:

        profit_factor = np.inf

    else:

        profit_factor = 0

    return {

        "THRESHOLD": threshold,
        "JUMLAH_TRADE": jumlah_trade,
        "WIN_RATE": win_rate,
        "AVG_RETURN": avg_return,
        "TOTAL_RETURN": total_return,
        "MAX_DRAWDOWN": max_drawdown,
        "PROFIT_FACTOR": profit_factor,
        "DATA": data
    }


# =========================================================
# REGRESSION
# =========================================================

def regression_pvalues(data):

    features = [
        "RSI14",
        "MACD",
        "CMF20",
        "CMO14",
        "BB_POSITION",
        "VOL_RATIO"
    ]

    reg = (
        data[
            features +
            ["RETURN_BESOK"]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    if len(reg) < 30:

        return pd.DataFrame()

    X = sm.add_constant(
        reg[features]
    )

    y = reg["RETURN_BESOK"]

    try:

        model = sm.OLS(
            y,
            X
        ).fit()

    except Exception:

        return pd.DataFrame()

    rows = []

    for feature in features:

        p = model.pvalues.get(
            feature,
            np.nan
        )

        coef = model.params.get(
            feature,
            np.nan
        )

        rows.append({

            "Indikator": feature,
            "Coefficient": coef,
            "P-value": p,

            "Signifikan":
                "YA"
                if p < 0.05
                else "TIDAK"
        })

    return pd.DataFrame(rows)


# =========================================================
# EMPIRICAL SCORE
# =========================================================

def empirical_score_stats(data):

    d = data[
        [
            "SCORE_HISTORIS",
            "RETURN_BESOK"
        ]
    ].dropna()

    if d.empty:

        return pd.DataFrame()

    return (
        d.groupby(
            "SCORE_HISTORIS"
        )
        .agg(

            Jumlah=(
                "RETURN_BESOK",
                "count"
            ),

            Probabilitas_Naik=(
                "RETURN_BESOK",
                lambda x:
                (x > 0).mean()
            ),

            Avg_Return=(
                "RETURN_BESOK",
                "mean"
            ),

            Median_Return=(
                "RETURN_BESOK",
                "median"
            )
        )
        .reset_index()
    )


# =========================================================
# DOWNLOAD
# =========================================================

@st.cache_data(ttl=900)
def ambil_data(
    symbol,
    start_date,
    end_date,
    interval
):

    raw = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if raw.empty:

        return pd.DataFrame()

    if isinstance(
        raw.columns,
        pd.MultiIndex
    ):

        raw.columns = (
            raw.columns
            .get_level_values(0)
        )

    return raw


# =========================================================
# CONSTANT
# =========================================================

MONTH_NAMES = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec"
}


MONTH_NAMES_FULL = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}


# =========================================================
# ZODIAC HISTORICAL HEATMAP
# =========================================================

ZODIAC_PERIODS = [
    ("Capricorn", 12, 22, 1, 19),
    ("Aquarius", 1, 20, 2, 18),
    ("Pisces", 2, 19, 3, 20),
    ("Aries", 3, 21, 4, 19),
    ("Taurus", 4, 20, 5, 20),
    ("Gemini", 5, 21, 6, 20),
    ("Cancer", 6, 21, 7, 22),
    ("Leo", 7, 23, 8, 22),
    ("Virgo", 8, 23, 9, 22),
    ("Libra", 9, 23, 10, 22),
    ("Scorpio", 10, 23, 11, 21),
    ("Sagittarius", 11, 22, 12, 21),
]

ZODIAC_ORDER = [
    "Capricorn",
    "Aquarius",
    "Pisces",
    "Aries",
    "Taurus",
    "Gemini",
    "Cancer",
    "Leo",
    "Virgo",
    "Libra",
    "Scorpio",
    "Sagittarius",
]


def get_zodiac(date_value):
    """Return western zodiac sign for a date."""
    if pd.isna(date_value):
        return None

    month = int(date_value.month)
    day = int(date_value.day)

    for name, start_m, start_d, end_m, end_d in ZODIAC_PERIODS:
        if start_m == 12 and end_m == 1:
            if (month == 12 and day >= start_d) or (
                month == 1 and day <= end_d
            ):
                return name
        elif (
            (month == start_m and day >= start_d)
            or
            (month == end_m and day <= end_d)
        ):
            return name

    return None


def calculate_zodiac_historical_returns(data):
    """
    Calculate compounded historical return by Year x Zodiac.
    The current, incomplete zodiac period is retained only if it has
    completed trading observations; no artificial return is created.
    """
    if data is None or data.empty:
        return pd.DataFrame()

    df = data.copy()

    if "Date" not in df.columns or "Return" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Return"] = pd.to_numeric(
        df["Return"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date", "Return"]
    ).copy()

    if df.empty:
        return pd.DataFrame()

    df["Zodiac"] = df["Date"].apply(
        get_zodiac
    )

    df["Year"] = df["Date"].dt.year

    df = df.dropna(
        subset=["Zodiac"]
    )

    if df.empty:
        return pd.DataFrame()

    # Compound daily returns within each year/zodiac period.
    result = (
        df.groupby(
            ["Year", "Zodiac"]
        )["Return"]
        .apply(
            lambda x: (1 + x).prod() - 1
        )
        .reset_index()
    )

    heatmap = (
        result
        .pivot(
            index="Year",
            columns="Zodiac",
            values="Return"
        )
        .reindex(
            columns=ZODIAC_ORDER
        )
        .sort_index()
    )

    return heatmap


def format_zodiac_heatmap_text(heatmap):
    if heatmap.empty:
        return heatmap

    # Pandas versi baru (>= 2.1) sudah menghapus DataFrame.applymap().
    # Gunakan DataFrame.map() jika tersedia, dengan fallback yang kompatibel
    # untuk versi pandas yang lebih lama.
    try:
        return heatmap.map(
            lambda x:
            ""
            if pd.isna(x)
            else f"{x:.1%}"
        )
    except AttributeError:
        return heatmap.apply(
            lambda column:
            column.map(
                lambda x:
                ""
                if pd.isna(x)
                else f"{x:.1%}"
            )
        )


PERIOD_OPTIONS = {
    "1 Tahun": "1y",
    "2 Tahun": "2y",
    "3 Tahun": "3y",
    "5 Tahun": "5y",
    "10 Tahun": "10y",
    "All Time": "max"
}


# =========================================================
# HEADER
# =========================================================

st.title(
    "📊 STOCK ANALYSIS TERMINAL"
)

st.caption(
    "Seasonality • Technical Analysis • News & Sentiment • Yahoo Finance"
)


# =========================================================
# LOAD MASTER
# =========================================================

@st.cache_data
def load_master():

    try:

        master = pd.read_excel(
            "saham_master.xlsx"
        )

    except Exception as e:

        st.error(
            f"Gagal membaca saham_master.xlsx: {e}"
        )

        st.stop()

    master.columns = (
        master.columns
        .astype(str)
        .str.strip()
    )

    if "Ticker" not in master.columns:

        st.error(
            "Kolom 'Ticker' tidak ditemukan "
            "di saham_master.xlsx"
        )

        st.stop()

    master["Ticker"] = (
        master["Ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    master = master[
        master["Ticker"] != ""
    ]

    master = (
        master
        .drop_duplicates(
            subset=["Ticker"]
        )
        .reset_index(drop=True)
    )

    return master


master = load_master()


# =========================================================
# REMOVE CURRENT INCOMPLETE MONTH
# =========================================================

def remove_current_month(data):

    if data.empty:
        return data

    now = pd.Timestamp.now()

    current_month_start = pd.Timestamp(
        year=now.year,
        month=now.month,
        day=1
    )

    return data[
        data["Date"] < current_month_start
    ].copy()


# =========================================================
# CLEAN YFINANCE DATA
# =========================================================

def clean_yfinance_columns(data):

    if data.empty:
        return data

    data = data.reset_index()

    if isinstance(
        data.columns,
        pd.MultiIndex
    ):

        data.columns = [
            col[0]
            if isinstance(col, tuple)
            else col
            for col in data.columns
        ]

    data = data.loc[
        :,
        ~data.columns.duplicated()
    ]

    return data


# =========================================================
# DOWNLOAD STOCK — SEASONALITY
# =========================================================

@st.cache_data(ttl=3600)
def download_stock(
    ticker,
    period
):

    try:

        data = yf.download(
            ticker + ".JK",
            period=period,
            auto_adjust=True,
            progress=False,
            group_by="column"
        )

    except Exception:

        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    data = clean_yfinance_columns(
        data
    )

    if (
        "Date" not in data.columns
        or
        "Close" not in data.columns
    ):

        return pd.DataFrame()

    data = data[
        [
            "Date",
            "Close"
        ]
    ].copy()

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["Close"]
    )

    data["Return"] = (
        data["Close"]
        .pct_change()
    )

    data["Year"] = (
        data["Date"]
        .dt.year
    )

    data["Month"] = (
        data["Date"]
        .dt.month
    )

    return data


# =========================================================
# DOWNLOAD STOCK — TECHNICAL
# =========================================================

@st.cache_data(ttl=3600)
def download_technical_stock(
    ticker,
    period="1y"
):

    try:

        data = yf.download(
            ticker + ".JK",
            period=period,
            auto_adjust=True,
            progress=False,
            group_by="column"
        )

    except Exception:

        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    data = clean_yfinance_columns(
        data
    )

    required = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    if not all(
        col in data.columns
        for col in required
    ):

        return pd.DataFrame()

    data = data[
        required
    ].copy()

    for col in required[1:]:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = data.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close"
        ]
    )

    return data


# =========================================================
# DOWNLOAD IHSG
# =========================================================

@st.cache_data(ttl=3600)
def download_ihsg(period):

    try:

        data = yf.download(
            "^JKSE",
            period=period,
            auto_adjust=True,
            progress=False,
            group_by="column"
        )

    except Exception:

        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    data = clean_yfinance_columns(
        data
    )

    if (
        "Date" not in data.columns
        or
        "Close" not in data.columns
    ):

        return pd.DataFrame()

    data = data[
        [
            "Date",
            "Close"
        ]
    ].copy()

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["Close"]
    )

    data["Return"] = (
        data["Close"]
        .pct_change()
    )

    data["Year"] = (
        data["Date"]
        .dt.year
    )

    data["Month"] = (
        data["Date"]
        .dt.month
    )

    return data


# =========================================================
# MONTHLY RETURN
# =========================================================

def calculate_monthly_returns(data):
    """
    Calculate true compounded monthly return.

    This function is intentionally kept for the broader seasonality
    statistics and weekly/monthly analysis. The Historical Year x Month
    heatmap below uses a separate Net Daily Movement calculation so that
    large bullish/bearish moves inside a month are not hidden.
    """
    if data.empty:
        return pd.DataFrame()

    complete_data = remove_current_month(data)

    if complete_data.empty:
        return pd.DataFrame()

    monthly = (
        complete_data
        .groupby(["Year", "Month"])["Return"]
        .apply(
            lambda x: (1 + x.dropna()).prod() - 1
        )
        .reset_index()
    )

    return monthly


# =========================================================
# NET DAILY MOVEMENT — HISTORICAL YEAR × MONTH HEATMAP
# =========================================================

def calculate_monthly_net_daily_movement(data):
    """
    Calculate the monthly net sum of daily returns.

    For every completed month:
      Bullish Movement = sum of all positive daily returns
      Bearish Movement = sum of all negative daily returns
      Net Daily Movement = Bullish Movement + Bearish Movement

    IMPORTANT:
    - No division by trading days.
    - No monthly close-to-close return is used for the heatmap.
    - Large bullish/bearish days inside a month remain visible in the
      monthly score.
    """
    if data is None or data.empty:
        return pd.DataFrame()

    complete_data = remove_current_month(data).copy()

    if complete_data.empty:
        return pd.DataFrame()

    # Ensure daily return is calculated from consecutive trading-day closes.
    complete_data = complete_data.sort_values("Date").copy()
    complete_data["Daily_Return_Pct"] = (
        pd.to_numeric(complete_data["Close"], errors="coerce")
        .pct_change()
        * 100.0
    )

    complete_data = complete_data.dropna(
        subset=["Daily_Return_Pct", "Date", "Close"]
    ).copy()

    if complete_data.empty:
        return pd.DataFrame()

    complete_data["Year"] = complete_data["Date"].dt.year
    complete_data["Month"] = complete_data["Date"].dt.month

    def summarize_month(group):
        returns = group["Daily_Return_Pct"]
        bullish = returns[returns > 0]
        bearish = returns[returns < 0]

        bullish_movement = float(bullish.sum()) if not bullish.empty else 0.0
        bearish_movement = float(bearish.sum()) if not bearish.empty else 0.0
        net_movement = bullish_movement + bearish_movement

        return pd.Series({
            "Bullish_Movement": bullish_movement,
            "Bearish_Movement": bearish_movement,
            "Net_Daily_Movement": net_movement,
            "Bullish_Days": int((returns > 0).sum()),
            "Bearish_Days": int((returns < 0).sum()),
            "Flat_Days": int((returns == 0).sum()),
            "Trading_Days": int(returns.notna().sum()),
            "Bullish_Rate": float((returns > 0).mean()),
        })

    result = (
        complete_data
        .groupby(["Year", "Month"], sort=True)
        .apply(summarize_month, include_groups=False)
        .reset_index()
    )

    result["Month_Name"] = result["Month"].map(MONTH_NAMES)

    return result


# =========================================================
# MONTHLY SEASONALITY
# =========================================================

def calculate_seasonality(data):

    monthly = calculate_monthly_returns(
        data
    )

    if monthly.empty:
        return pd.DataFrame()

    result = (
        monthly
        .groupby("Month")["Return"]
        .agg(
            Average_Return="mean",
            Median_Return="median",
            Win_Rate=lambda x:
            (x > 0).mean(),
            Observations="count",
            Bullish=lambda x:
            (x > 0).sum(),
            Bearish=lambda x:
            (x < 0).sum(),
            Flat=lambda x:
            (x == 0).sum()
        )
        .reset_index()
    )

    result["Month_Name"] = (
        result["Month"]
        .map(MONTH_NAMES)
    )

    return result


# =========================================================
# ASSIGN WEEK BUCKET
# =========================================================

def assign_week_bucket(group):

    group = (
        group
        .sort_values("Date")
        .copy()
    )

    n = len(group)

    if n == 0:
        return group

    week_values = np.ones(
        n,
        dtype=int
    )

    split_positions = np.array_split(
        np.arange(n),
        4
    )

    for week_number, positions in enumerate(
        split_positions,
        start=1
    ):

        if len(positions) > 0:

            week_values[
                positions
            ] = week_number

    group["Week"] = week_values

    return group


# =========================================================
# WEEKLY PHASE RETURNS
# =========================================================

def calculate_weekly_phase_returns(data):

    if data.empty:
        return pd.DataFrame()

    complete_data = remove_current_month(
        data
    )

    if complete_data.empty:
        return pd.DataFrame()

    groups = []

    for (
        (year, month),
        group
    ) in complete_data.groupby(
        [
            "Year",
            "Month"
        ],
        sort=True
    ):

        group = assign_week_bucket(
            group
        )

        groups.append(group)

    if not groups:
        return pd.DataFrame()

    grouped = pd.concat(
        groups,
        ignore_index=True
    )

    weekly = (
        grouped
        .groupby(
            [
                "Year",
                "Month",
                "Week"
            ],
            as_index=False
        )
        .agg(
            Start_Close=(
                "Close",
                "first"
            ),
            End_Close=(
                "Close",
                "last"
            ),
            Trading_Days=(
                "Date",
                "count"
            )
        )
    )

    weekly["Return"] = (
        weekly["End_Close"]
        /
        weekly["Start_Close"]
        - 1
    )

    weekly["Month_Name"] = (
        weekly["Month"]
        .map(MONTH_NAMES)
    )

    weekly["Week_Name"] = (
        "Week "
        +
        weekly["Week"].astype(str)
    )

    return weekly


# =========================================================
# WEEKLY SEASONAL SUMMARY
# =========================================================

def calculate_weekly_summary(data):

    weekly = calculate_weekly_phase_returns(
        data
    )

    if weekly.empty:
        return pd.DataFrame()

    summary = (
        weekly
        .groupby(
            [
                "Month",
                "Week"
            ]
        )["Return"]
        .agg(
            Average_Return="mean",
            Median_Return="median",
            Win_Rate=lambda x:
            (x > 0).mean(),
            Observations="count",
            Bullish=lambda x:
            (x > 0).sum(),
            Bearish=lambda x:
            (x < 0).sum(),
            Flat=lambda x:
            (x == 0).sum()
        )
        .reset_index()
    )

    summary["Month_Name"] = (
        summary["Month"]
        .map(MONTH_NAMES)
    )

    summary["Week_Name"] = (
        "Week "
        +
        summary["Week"].astype(str)
    )

    return summary


# =========================================================
# SEASONALITY DECISION
# =========================================================

def classify_seasonality(
    avg_return,
    median_return,
    win_rate
):

    if (
        avg_return > 0
        and
        median_return > 0
        and
        win_rate >= 0.60
    ):
        return "FAVORABLE"

    if (
        avg_return < 0
        and
        median_return < 0
        and
        win_rate < 0.50
    ):
        return "UNFAVORABLE"

    return "NEUTRAL / MIXED"


# =========================================================
# EVIDENCE STRENGTH
# =========================================================

def classify_evidence(
    observations
):

    if observations >= 8:
        return "STRONG EVIDENCE"

    if observations >= 5:
        return "MODERATE EVIDENCE"

    if observations >= 3:
        return "LOW EVIDENCE"

    return "VERY LOW EVIDENCE"


# =========================================================
# CURRENT MONTH ANALYSIS
# =========================================================

def create_current_month_analysis(
    seasonal,
    period_label,
    first_date,
    last_date
):

    if seasonal.empty:
        return None

    current_month = (
        datetime.now().month
    )

    current_name = (
        MONTH_NAMES_FULL[
            current_month
        ]
    )

    current = seasonal[
        seasonal["Month"]
        == current_month
    ]

    if current.empty:

        return {
            "available": False,
            "month": current_name
        }

    row = current.iloc[0]

    avg_return = float(
        row["Average_Return"]
    )

    median_return = float(
        row["Median_Return"]
    )

    win_rate = float(
        row["Win_Rate"]
    )

    observations = int(
        row["Observations"]
    )

    bullish = int(
        row["Bullish"]
    )

    bearish = int(
        row["Bearish"]
    )

    flat = int(
        row["Flat"]
    )

    decision = classify_seasonality(
        avg_return,
        median_return,
        win_rate
    )

    evidence = classify_evidence(
        observations
    )

    return {
        "available": True,
        "month": current_name,
        "month_number": current_month,
        "avg_return": avg_return,
        "median_return": median_return,
        "win_rate": win_rate,
        "observations": observations,
        "bullish": bullish,
        "bearish": bearish,
        "flat": flat,
        "decision": decision,
        "evidence": evidence,
        "period_label": period_label,
        "first_date": first_date,
        "last_date": last_date
    }


# =========================================================
# CURRENT MONTH WEEK DECISION
# =========================================================

def get_current_month_week_decision(
    weekly_summary
):

    if weekly_summary.empty:
        return None

    current_month = (
        datetime.now().month
    )

    current = weekly_summary[
        weekly_summary["Month"]
        == current_month
    ].copy()

    if current.empty:
        return None

    current = (
        current
        .sort_values("Week")
        .reset_index(drop=True)
    )

    entry_candidates = current[
        (
            current["Average_Return"] > 0
        )
        &
        (
            current["Median_Return"] > 0
        )
        &
        (
            current["Win_Rate"] >= 0.60
        )
    ].copy()

    if not entry_candidates.empty:

        best_entry = (
            entry_candidates
            .sort_values(
                [
                    "Average_Return",
                    "Win_Rate"
                ],
                ascending=False
            )
            .iloc[0]
        )

    else:

        best_entry = None

    weak_candidates = current[
        (
            current["Average_Return"] < 0
        )
        &
        (
            current["Median_Return"] < 0
        )
        &
        (
            current["Win_Rate"] < 0.50
        )
    ].copy()

    if not weak_candidates.empty:

        weakest = (
            weak_candidates
            .sort_values(
                [
                    "Average_Return",
                    "Win_Rate"
                ],
                ascending=True
            )
            .iloc[0]
        )

    else:

        weakest = None

    return {
        "current": current,
        "best_entry": best_entry,
        "weakest": weakest
    }


# =========================================================
# SEASONAL VECTOR
# =========================================================

def seasonal_vector(data):

    result = calculate_seasonality(
        data
    )

    if result.empty:
        return None

    vector = (
        result
        .set_index("Month")
        ["Average_Return"]
        .reindex(range(1, 13))
    )

    return vector


# =========================================================
# SIMILARITY
# =========================================================

def calculate_similarity(
    vector_a,
    vector_b
):

    if (
        vector_a is None
        or
        vector_b is None
    ):
        return np.nan

    combined = pd.concat(
        [
            vector_a,
            vector_b
        ],
        axis=1
    ).dropna()

    if len(combined) < 6:
        return np.nan

    a = (
        combined
        .iloc[:, 0]
        .values
    )

    b = (
        combined
        .iloc[:, 1]
        .values
    )

    if (
        np.std(a) == 0
        or
        np.std(b) == 0
    ):
        return np.nan

    correlation = np.corrcoef(
        a,
        b
    )[0, 1]

    if np.isnan(correlation):
        return np.nan

    return (
        (correlation + 1)
        / 2
        * 100
    )


# =========================================================
# BROKER / BANDAR FLOW
# =========================================================

def _get_secret_value(name):
    """Read a Streamlit secret first, then an environment variable."""
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv(name, "").strip()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_broker_summary_day(ticker, trade_date, investor="all", market="RG"):
    """Fetch one completed trading day's broker summary.

    Index Alpha returns one row per broker. A single-day request is used
    intentionally because multi-day requests are aggregated by broker.
    """
    api_key = _get_secret_value("INDEX_ALPHA_API_KEY")
    if not api_key:
        return pd.DataFrame()

    url = "https://api.indexalpha.id/stocks/broker-summary"
    params = {
        "ticker": ticker.replace(".JK", "").upper(),
        "from": str(trade_date),
        "to": str(trade_date),
        "investor": investor,
        "market": market,
    }
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        payload = r.json()
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        numeric_cols = [
            "buy_freq", "buy_volume", "buy_value",
            "sell_freq", "sell_volume", "sell_value",
            "buy_avg", "sell_avg",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df["broker"] = df["code"].astype(str).str.upper()
        df["net_value"] = df["buy_value"] - df["sell_value"]
        df["net_volume"] = df["buy_volume"] - df["sell_volume"]
        df["date"] = pd.Timestamp(trade_date)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_broker_flow_range(ticker, dates, investor="all", market="RG"):
    """Fetch daily broker summaries for a list of trading dates."""
    dates = [pd.Timestamp(x).date() for x in dates]
    if not dates:
        return pd.DataFrame()

    # Parallel requests make a 60-day window practical while respecting
    # normal API rate limits. Individual calls are cached above.
    frames = []
    max_workers = min(6, max(1, len(dates)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_broker_summary_day, ticker, d, investor, market): d
            for d in dates
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                if not result.empty:
                    frames.append(result)
            except Exception:
                continue

    if not frames:
        return pd.DataFrame()

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "net_value"], ascending=[True, False])
        .reset_index(drop=True)
    )


def build_daily_broker_flow_summary(flow_df):
    """Create one row per date showing the leading 3 accumulators/distributors."""
    if flow_df.empty:
        return pd.DataFrame()

    rows = []
    for dt, g in flow_df.groupby("date", sort=True):
        g = g.copy()
        acc = g[g["net_value"] > 0].sort_values("net_value", ascending=False).head(3)
        dist = g[g["net_value"] < 0].sort_values("net_value", ascending=True).head(3)

        acc_text = " • ".join(
            f"{r.broker} ({r.net_value/1e9:+.2f}B)"
            for r in acc.itertuples()
        ) or "-"
        dist_text = " • ".join(
            f"{r.broker} ({abs(r.net_value)/1e9:.2f}B)"
            for r in dist.itertuples()
        ) or "-"

        rows.append({
            "Tanggal": pd.Timestamp(dt).strftime("%d-%m-%Y"),
            "Top 3 Akumulasi": acc_text,
            "Top 3 Distribusi": dist_text,
        })

    return pd.DataFrame(rows)


def render_broker_flow_module(ticker, stock_data, end_date, broker_window=60):
    """Render broker/bandar movement replacing stock-vs-IHSG/similar stocks."""
    st.divider()
    st.subheader("🏦 PERGERAKAN BANDAR / BROKER FLOW")
    st.caption(
        "Akumulasi = net buy broker; distribusi = net sell broker. "
        "Kode broker menunjukkan broker transaksi, bukan identitas investor akhir."
    )

    api_key = _get_secret_value("INDEX_ALPHA_API_KEY")
    if not api_key:
        st.warning(
            "Data broker belum aktif. Tambahkan `INDEX_ALPHA_API_KEY` ke "
            "Streamlit Secrets agar kode broker, akumulasi, dan distribusi harian "
            "dapat ditampilkan."
        )
        st.info(
            "API broker yang digunakan menyediakan buy/sell value, volume, dan "
            "average price per broker. Data historis tersedia mulai 1 Januari 2025."
        )
        return

    # Use actual stock trading dates, then keep only the supported broker-data era.
    dates = pd.to_datetime(stock_data["Date"], errors="coerce").dropna().dt.normalize().drop_duplicates().sort_values()
    dates = dates[dates <= pd.Timestamp(end_date)]
    dates = dates[dates >= pd.Timestamp("2025-01-01")]

    if dates.empty:
        st.info("Belum ada tanggal yang masuk periode data broker.")
        return

    # Window is deliberately configurable so 2/5/10-year seasonality does not
    # trigger hundreds of broker API calls on every page load.
    if broker_window > 0:
        dates = dates.tail(int(broker_window))

    with st.spinner(f"Mengambil broker flow {len(dates)} hari..."):
        flow = fetch_broker_flow_range(ticker, dates.tolist())

    if flow.empty:
        st.error(
            "Data broker tidak berhasil diambil. Periksa API key, kuota API, "
            "atau tanggal perdagangan."
        )
        return

    daily = build_daily_broker_flow_summary(flow)
    end_dt = pd.Timestamp(end_date)
    available_end = flow[flow["date"] <= end_dt]["date"].max()
    end_flow = flow[flow["date"] == available_end].copy()

    # Top 3 on the selected end date.
    top_acc = end_flow.sort_values("net_value", ascending=False).head(3).copy()
    top_dist = end_flow.sort_values("net_value", ascending=True).head(3).copy()

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown("### 🟢 Top 3 Akumulasi")
        acc_show = top_acc[["broker", "buy_value", "sell_value", "net_value", "buy_avg", "sell_avg"]].copy()
        acc_show.columns = ["Broker", "Buy Value", "Sell Value", "Net Buy", "Avg Buy", "Avg Sell"]
        st.dataframe(
            acc_show.style.format({
                "Buy Value": lambda x: f"Rp {x/1e9:,.2f} B",
                "Sell Value": lambda x: f"Rp {x/1e9:,.2f} B",
                "Net Buy": lambda x: f"Rp {x/1e9:,.2f} B",
                "Avg Buy": lambda x: f"Rp {x:,.0f}",
                "Avg Sell": lambda x: f"Rp {x:,.0f}",
            }),
            use_container_width=True, hide_index=True, height=150
        )

    with c2:
        st.markdown("### 🔴 Top 3 Distribusi")
        dist_show = top_dist[["broker", "buy_value", "sell_value", "net_value", "buy_avg", "sell_avg"]].copy()
        dist_show["net_value"] = dist_show["net_value"].abs()
        dist_show.columns = ["Broker", "Buy Value", "Sell Value", "Net Sell", "Avg Buy", "Avg Sell"]
        st.dataframe(
            dist_show.style.format({
                "Buy Value": lambda x: f"Rp {x/1e9:,.2f} B",
                "Sell Value": lambda x: f"Rp {x/1e9:,.2f} B",
                "Net Sell": lambda x: f"Rp {x/1e9:,.2f} B",
                "Avg Buy": lambda x: f"Rp {x:,.0f}",
                "Avg Sell": lambda x: f"Rp {x:,.0f}",
            }),
            use_container_width=True, hide_index=True, height=150
        )

    st.caption(
        f"End date broker summary: {pd.Timestamp(available_end).strftime('%d-%m-%Y')} "
        f"• Window: {pd.Timestamp(dates.min()).strftime('%d-%m-%Y')} – "
        f"{pd.Timestamp(dates.max()).strftime('%d-%m-%Y')}"
    )

    # Daily movement chart: total positive vs total negative broker net value.
    daily_chart = (
        flow.assign(
            Accumulation=flow["net_value"].clip(lower=0),
            Distribution=flow["net_value"].clip(upper=0),
        )
        .groupby("date")[["Accumulation", "Distribution"]]
        .sum()
        .reset_index()
    )

    fig_flow = go.Figure()
    fig_flow.add_trace(go.Bar(
        x=daily_chart["date"], y=daily_chart["Accumulation"],
        name="Akumulasi", marker_color="#22c55e"
    ))
    fig_flow.add_trace(go.Bar(
        x=daily_chart["date"], y=daily_chart["Distribution"],
        name="Distribusi", marker_color="#ef4444"
    ))
    fig_flow.add_hline(y=0, line_width=1, line_color="#777")
    fig_flow.update_layout(
        template="plotly_dark", paper_bgcolor="#11161d", plot_bgcolor="#11161d",
        height=320, barmode="relative",
        yaxis=dict(title="Net Broker Flow (Rp)", tickformat=".2s"),
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    st.plotly_chart(fig_flow, use_container_width=True, config={"displayModeBar": False})

    st.markdown("### 📅 Distribusi Broker Per Hari")
    st.dataframe(
        daily.sort_values("Tanggal", ascending=False),
        use_container_width=True, hide_index=True, height=300
    )

    # Cumulative broker leaderboard across the displayed window.
    cumulative = (
        flow.groupby("broker", as_index=False)["net_value"]
        .sum()
        .sort_values("net_value", ascending=False)
    )
    cum_acc = cumulative.head(3).copy()
    cum_dist = cumulative.sort_values("net_value").head(3).copy()
    q1, q2 = st.columns(2)
    with q1:
        st.markdown("### 🟢 Top 3 Akumulasi Kumulatif")
        st.dataframe(
            cum_acc.rename(columns={"broker":"Broker", "net_value":"Net Buy"})
            .assign(**{"Net Buy": lambda d: d["Net Buy"].map(lambda x: f"Rp {x/1e9:,.2f} B")}),
            use_container_width=True, hide_index=True
        )
    with q2:
        st.markdown("### 🔴 Top 3 Distribusi Kumulatif")
        cum_dist = cum_dist.rename(columns={"broker":"Broker", "net_value":"Net Sell"}).copy()
        cum_dist["Net Sell"] = cum_dist["Net Sell"].abs().map(lambda x: f"Rp {x/1e9:,.2f} B")
        st.dataframe(cum_dist, use_container_width=True, hide_index=True)


# =========================================================
# SIMILAR STOCKS
# =========================================================

@st.cache_data(ttl=3600)
def find_similar_stocks(
    selected_ticker,
    period,
    tickers
):

    target_data = download_stock(
        selected_ticker,
        period
    )

    target_vector = seasonal_vector(
        target_data
    )

    if target_vector is None:
        return pd.DataFrame()

    results = []

    for ticker_item in tickers:

        if ticker_item == selected_ticker:
            continue

        data = download_stock(
            ticker_item,
            period
        )

        if data.empty:
            continue

        vector = seasonal_vector(
            data
        )

        score = calculate_similarity(
            target_vector,
            vector
        )

        if pd.notna(score):

            results.append(
                {
                    "Ticker": ticker_item,
                    "Similarity": score
                }
            )

    if not results:
        return pd.DataFrame()

    result = pd.DataFrame(
        results
    )

    result = (
        result
        .sort_values(
            "Similarity",
            ascending=False
        )
        .reset_index(drop=True)
    )

    result["Rank"] = (
        result.index + 1
    )

    return result[
        [
            "Rank",
            "Ticker",
            "Similarity"
        ]
    ]


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def calculate_technical_indicators(
    data
):

    if data.empty:
        return data

    df = data.copy()

    # -----------------------------------------------------
    # EMA
    # -----------------------------------------------------

    df["EMA20"] = (
        df["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["EMA50"] = (
        df["Close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    df["EMA100"] = (
        df["Close"]
        .ewm(
            span=100,
            adjust=False
        )
        .mean()
    )

    # -----------------------------------------------------
    # BOLLINGER BAND
    # -----------------------------------------------------

    df["BB_Middle"] = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    bb_std = (
        df["Close"]
        .rolling(20)
        .std()
    )

    df["BB_Upper"] = (
        df["BB_Middle"]
        + 2 * bb_std
    )

    df["BB_Lower"] = (
        df["BB_Middle"]
        - 2 * bb_std
    )

    # -----------------------------------------------------
    # RSI 14
    # -----------------------------------------------------

    delta = (
        df["Close"]
        .diff()
    )

    gain = (
        delta.clip(lower=0)
        .rolling(14)
        .mean()
    )

    loss = (
        -delta.clip(upper=0)
        .rolling(14)
        .mean()
    )

    rs = (
        gain / loss.replace(
            0,
            np.nan
        )
    )

    df["RSI14"] = (
        100
        -
        (
            100
            /
            (1 + rs)
        )
    )

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    ema12 = (
        df["Close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        df["Close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    df["MACD"] = (
        ema12 - ema26
    )

    df["MACD_Signal"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["MACD_Hist"] = (
        df["MACD"]
        -
        df["MACD_Signal"]
    )

    # -----------------------------------------------------
    # CMF
    # -----------------------------------------------------

    hl_range = (
        df["High"]
        -
        df["Low"]
    )

    money_flow_multiplier = (
        (
            df["Close"]
            -
            df["Low"]
        )
        -
        (
            df["High"]
            -
            df["Close"]
        )
    ) / hl_range.replace(
        0,
        np.nan
    )

    money_flow_volume = (
        money_flow_multiplier
        *
        df["Volume"]
    )

    df["CMF"] = (
        money_flow_volume
        .rolling(20)
        .sum()
        /
        df["Volume"]
        .rolling(20)
        .sum()
    )

    # -----------------------------------------------------
    # VOLUME AVERAGE
    # -----------------------------------------------------

    df["Volume_MA20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    return df


# =========================================================
# TECHNICAL SUMMARY
# =========================================================

def technical_summary(
    df
):

    if df.empty:
        return {}

    row = df.iloc[-1]

    close = float(row["Close"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    ema100 = float(row["EMA100"])
    rsi = float(row["RSI14"])
    macd = float(row["MACD"])
    signal = float(row["MACD_Signal"])
    cmf = float(row["CMF"])

    volume = float(row["Volume"])
    volume_ma = float(row["Volume_MA20"])

    # -----------------------------------------------------
    # TREND
    # -----------------------------------------------------

    if (
        close > ema20
        and
        ema20 > ema50
        and
        ema50 > ema100
    ):

        trend = "BULLISH"

    elif (
        close < ema20
        and
        ema20 < ema50
        and
        ema50 < ema100
    ):

        trend = "BEARISH"

    else:

        trend = "MIXED"

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    if rsi >= 70:

        rsi_status = "OVERBOUGHT"

    elif rsi <= 30:

        rsi_status = "OVERSOLD"

    elif rsi >= 50:

        rsi_status = "BULLISH ZONE"

    else:

        rsi_status = "BEARISH ZONE"

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    if macd > signal:

        macd_status = "BULLISH"

    else:

        macd_status = "BEARISH"

    # -----------------------------------------------------
    # CMF
    # -----------------------------------------------------

    if cmf > 0.05:

        cmf_status = "MONEY INFLOW"

    elif cmf < -0.05:

        cmf_status = "MONEY OUTFLOW"

    else:

        cmf_status = "NEUTRAL"

    # -----------------------------------------------------
    # VOLUME
    # -----------------------------------------------------

    if volume_ma > 0:

        volume_ratio = (
            volume / volume_ma
        )

    else:

        volume_ratio = np.nan

    if (
        pd.notna(volume_ratio)
        and
        volume_ratio >= 1.5
    ):

        volume_status = "HIGH"

    elif (
        pd.notna(volume_ratio)
        and
        volume_ratio < 0.75
    ):

        volume_status = "LOW"

    else:

        volume_status = "NORMAL"

    return {
        "close": close,
        "trend": trend,
        "rsi": rsi,
        "rsi_status": rsi_status,
        "macd": macd,
        "macd_signal": signal,
        "macd_status": macd_status,
        "cmf": cmf,
        "cmf_status": cmf_status,
        "volume_ratio": volume_ratio,
        "volume_status": volume_status
    }


# =========================================================
# NEWS
# =========================================================

@st.cache_data(ttl=1800)
def get_stock_news(
    ticker
):

    try:

        stock = yf.Ticker(
            ticker + ".JK"
        )

        news = stock.news

    except Exception:

        return pd.DataFrame()

    if not news:
        return pd.DataFrame()

    rows = []

    for item in news:

        content = item.get(
            "content",
            {}
        )

        title = (
            content.get("title")
            or
            item.get("title")
            or
            ""
        )

        publisher = (
            content.get(
                "provider",
                {}
            )
            .get("displayName")
            or
            item.get("publisher")
            or
            ""
        )

        url = (
            content.get(
                "canonicalUrl",
                {}
            )
            .get("url")
            or
            item.get("link")
            or
            ""
        )

        pub_date = (
            content.get(
                "pubDate"
            )
            or
            item.get(
                "providerPublishTime"
            )
        )

        if pub_date:

            try:

                if isinstance(
                    pub_date,
                    (int, float)
                ):

                    date_value = pd.to_datetime(
                        pub_date,
                        unit="s"
                    )

                else:

                    date_value = pd.to_datetime(
                        pub_date
                    )

            except Exception:

                date_value = pd.NaT

        else:

            date_value = pd.NaT

        if title:

            rows.append(
                {
                    "Date": date_value,
                    "Title": title,
                    "Publisher": publisher,
                    "URL": url
                }
            )

    if not rows:
        return pd.DataFrame()

    news_df = pd.DataFrame(
        rows
    )

    news_df = (
        news_df
        .drop_duplicates(
            subset=["Title"]
        )
        .sort_values(
            "Date",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return news_df


# =========================================================
# SIMPLE NEWS SENTIMENT
# =========================================================

def calculate_news_sentiment(
    title
):

    title = str(title).lower()

    positive_words = [
        "profit",
        "laba",
        "revenue",
        "growth",
        "grow",
        "surge",
        "rise",
        "rises",
        "higher",
        "strong",
        "positive",
        "bullish",
        "buy",
        "upgrade",
        "dividend",
        "contract",
        "award",
        "earnings",
        "record",
        "improve",
        "improved",
        "increase",
        "increased",
        "acquisition",
        "expansion"
    ]

    negative_words = [
        "loss",
        "rugi",
        "decline",
        "fall",
        "falls",
        "lower",
        "weak",
        "negative",
        "bearish",
        "sell",
        "downgrade",
        "debt",
        "lawsuit",
        "investigation",
        "risk",
        "cut",
        "cuts",
        "decrease",
        "decreased",
        "warning",
        "scandal"
    ]

    positive_score = sum(
        word in title
        for word in positive_words
    )

    negative_score = sum(
        word in title
        for word in negative_words
    )

    if positive_score > negative_score:

        return "POSITIVE"

    if negative_score > positive_score:

        return "NEGATIVE"

    return "NEUTRAL"


# =========================================================



# =========================================================
# LOAD MASTER STOCK LIST
# =========================================================

@st.cache_data
def load_master():
    try:
        master = pd.read_excel("saham_master.xlsx")
    except Exception as e:
        st.error(f"Gagal membaca saham_master.xlsx: {e}")
        st.stop()
    master.columns = master.columns.astype(str).str.strip()
    if "Ticker" not in master.columns:
        st.error("Kolom 'Ticker' tidak ditemukan di saham_master.xlsx")
        st.stop()
    master["Ticker"] = master["Ticker"].astype(str).str.upper().str.strip().str.replace(".JK", "", regex=False)
    master = master[master["Ticker"] != ""].drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    return master

master = load_master()


# =========================================================
# STATISTICAL DISCOVERY ENGINE
# MODUL TERPISAH DARI TECHNICAL / SEASONALITY / NEWS
# =========================================================

STAT_PERIODS = {
    "1 Tahun": "1y",
    "2 Tahun": "2y",
    "3 Tahun": "3y",
    "5 Tahun": "5y",
    "10 Tahun": "10y",
    "Max": "max",
}

STAT_FORWARD_MAP = {
    "1 Hari": 1,
    "3 Hari": 3,
    "5 Hari": 5,
    "10 Hari": 10,
}


def _flat_ohlcv(data):
    """Normalize yfinance output into standard OHLCV columns."""
    if data is None or data.empty:
        return pd.DataFrame()
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().title() for c in df.columns]
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            if c == "Volume":
                df[c] = 0.0
            else:
                return pd.DataFrame()
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index().dropna(subset=["Close"])


@st.cache_data(ttl=1800)
def download_stat_data(ticker, period):
    """Daily historical data used only by Statistical Discovery."""
    try:
        raw = yf.download(
            ticker + ".JK",
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column",
        )
        return _flat_ohlcv(raw)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def download_stat_market(period):
    """IHSG market return used as a statistical candidate factor."""
    try:
        raw = yf.download(
            "^JKSE",
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column",
        )
        df = _flat_ohlcv(raw)
        if df.empty:
            return pd.DataFrame()
        return df[["Close"]].rename(columns={"Close": "IHSG_Close"})
    except Exception:
        return pd.DataFrame()


def statistical_features(price, market=None):
    """Create candidate explanatory variables without using future data."""
    d = price.copy()
    close = d["Close"]
    ret1 = close.pct_change()

    d["Return_1D"] = ret1
    d["Return_3D"] = close.pct_change(3)
    d["Return_5D"] = close.pct_change(5)
    d["Return_10D"] = close.pct_change(10)
    d["Return_20D"] = close.pct_change(20)
    d["Volatility_20D"] = ret1.rolling(20).std()
    d["Volume_Change"] = d["Volume"].pct_change()
    d["Volume_Ratio_20D"] = d["Volume"] / d["Volume"].rolling(20).median()

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    d["Price_vs_MA20"] = close / ma20 - 1
    d["Price_vs_MA50"] = close / ma50 - 1
    d["MA20_Slope"] = ma20.pct_change(5)
    d["MA50_Slope"] = ma50.pct_change(10)

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI14"] = 100 - 100 / (1 + rs)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    d["MACD"] = macd
    d["MACD_Hist"] = macd - macd_signal
    d["MACD_Slope"] = macd.diff(3)

    std20 = close.rolling(20).std()
    d["BB_Position"] = (close - ma20) / (2 * std20.replace(0, np.nan))

    # ATR percentage rather than absolute price level.
    prev = close.shift(1)
    tr = pd.concat(
        [d["High"] - d["Low"],
         (d["High"] - prev).abs(),
         (d["Low"] - prev).abs()], axis=1
    ).max(axis=1)
    d["ATR_Pct"] = tr.rolling(14).mean() / close

    if market is not None and not market.empty:
        m = market.reindex(d.index).ffill()
        d["IHSG_Return_1D"] = m["IHSG_Close"].pct_change()
        d["IHSG_Return_5D"] = m["IHSG_Close"].pct_change(5)
        d["IHSG_Return_20D"] = m["IHSG_Close"].pct_change(20)
    else:
        d["IHSG_Return_1D"] = np.nan
        d["IHSG_Return_5D"] = np.nan
        d["IHSG_Return_20D"] = np.nan

    return d


def prepare_stat_target(features, forward_days):
    """Forward return target. It is shifted backwards so predictors stay historical."""
    d = features.copy()
    d["Forward_Return"] = d["Close"].shift(-forward_days) / d["Close"] - 1
    d["Forward_Direction"] = (d["Forward_Return"] > 0).astype(float)
    return d


def correlation_table(df, target="Forward_Return"):
    candidates = [c for c in df.columns if c != target and c not in ["Open", "High", "Low", "Close", "Volume", "Forward_Direction"]]
    rows = []
    for col in candidates:
        pair = df[[col, target]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 60:
            continue
        corr = pair[col].corr(pair[target])
        if pd.isna(corr):
            continue
        rows.append({"Variable": col, "Correlation": float(corr), "Abs Correlation": abs(float(corr)), "N": len(pair)})
    if not rows:
        return pd.DataFrame(columns=["Variable", "Correlation", "Abs Correlation", "N"])
    return pd.DataFrame(rows).sort_values("Abs Correlation", ascending=False).reset_index(drop=True)


def rolling_correlation_stability(df, variables, target="Forward_Return", windows=(252, 504)):
    rows = []
    for col in variables:
        vals = []
        for w in windows:
            if len(df) < w:
                continue
            part = df[[col, target]].tail(w).replace([np.inf, -np.inf], np.nan).dropna()
            if len(part) >= 60:
                vals.append(part[col].corr(part[target]))
        if not vals:
            vals = [df[[col, target]].replace([np.inf, -np.inf], np.nan).dropna()[col].corr(df[[col, target]].replace([np.inf, -np.inf], np.nan).dropna()[target])]
        vals = [v for v in vals if pd.notna(v)]
        if vals:
            rows.append({
                "Variable": col,
                "Recent Corr": vals[-1],
                "Mean Corr": float(np.mean(vals)),
                "Corr Std": float(np.std(vals)),
                "Stability": max(0.0, 1.0 - min(1.0, float(np.std(vals)) / 0.25)),
            })
    return pd.DataFrame(rows)


def regression_discovery(df, selected, target="Forward_Return", test_ratio=0.25):
    cols = [c for c in selected if c in df.columns]
    work = df[cols + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < max(100, len(cols) * 20):
        return None

    split = int(len(work) * (1 - test_ratio))
    split = max(60, min(split, len(work) - 30))
    train = work.iloc[:split]
    test = work.iloc[split:]

    X_train = sm.add_constant(train[cols], has_constant="add")
    y_train = train[target]
    model = sm.OLS(y_train, X_train).fit()

    X_test = sm.add_constant(test[cols], has_constant="add")
    pred = model.predict(X_test)
    y_test = test[target]

    test_r2 = np.nan
    if len(y_test) > 1 and y_test.var() > 0:
        test_r2 = 1 - ((y_test - pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
    directional = np.mean(np.sign(pred) == np.sign(y_test)) if len(y_test) else np.nan

    coef = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "P-Value": model.pvalues.values,
    })
    coef = coef[coef["Variable"] != "const"].copy()
    coef["Significant"] = coef["P-Value"] < 0.05

    return {
        "model": model,
        "coef": coef,
        "train_r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "test_r2": float(test_r2) if pd.notna(test_r2) else np.nan,
        "directional_accuracy": float(directional) if pd.notna(directional) else np.nan,
        "train_n": len(train),
        "test_n": len(test),
    }


def statistical_discovery(price, market, forward_days, top_k=6):
    """Main discovery pipeline: correlation -> stability -> regression -> OOS."""
    feat = statistical_features(price, market)
    df = prepare_stat_target(feat, forward_days)
    corr = correlation_table(df)
    if corr.empty:
        return {"data": df, "corr": corr, "stability": pd.DataFrame(), "regression": None, "selected": []}

    # Avoid selecting multiple almost-duplicate return variables at once.
    preferred = corr["Variable"].tolist()
    selected = []
    corr_map = corr.set_index("Variable")["Correlation"].to_dict()
    for col in preferred:
        if len(selected) >= top_k:
            break
        if col.startswith("Return_") and any(x.startswith("Return_") for x in selected):
            # Keep the strongest return horizon only.
            continue
        selected.append(col)

    stability = rolling_correlation_stability(df, selected)
    if not stability.empty:
        merged = corr.merge(stability, on="Variable", how="left")
        merged["Discovery Score"] = (
            merged["Abs Correlation"] * 0.55
            + merged["Stability"].fillna(0) * 0.25
            + (merged["N"] / max(1, len(df))).clip(0, 1) * 0.20
        )
        selected = merged.sort_values("Discovery Score", ascending=False)["Variable"].head(top_k).tolist()
        corr = merged.sort_values("Discovery Score", ascending=False).reset_index(drop=True)

    reg = regression_discovery(df, selected)
    return {"data": df, "corr": corr, "stability": stability, "regression": reg, "selected": selected}


def technical_recommendation(df, selected):
    """Translate statistically selected factors into human-readable technical focus."""
    mapping = {
        "RSI14": "RSI 14",
        "MACD": "MACD",
        "MACD_Hist": "MACD Histogram",
        "MACD_Slope": "MACD Slope",
        "BB_Position": "Bollinger Band",
        "Volume_Change": "Volume Change",
        "Volume_Ratio_20D": "Volume Ratio",
        "Price_vs_MA20": "Price vs MA20",
        "Price_vs_MA50": "Price vs MA50",
        "MA20_Slope": "MA20 Slope",
        "MA50_Slope": "MA50 Slope",
        "ATR_Pct": "ATR / Volatility",
        "Volatility_20D": "Volatility 20D",
        "Return_3D": "Momentum 3D",
        "Return_5D": "Momentum 5D",
        "Return_10D": "Momentum 10D",
        "Return_20D": "Momentum 20D",
        "IHSG_Return_1D": "IHSG Return 1D",
        "IHSG_Return_5D": "IHSG Return 5D",
        "IHSG_Return_20D": "IHSG Return 20D",
        "Return_1D": "Return 1D",
    }
    return [mapping.get(x, x) for x in selected]


def render_statistical_module(ticker, period_label, forward_label, top_k):
    st.markdown("## 🔬 ANALISA STATISTIK")
    st.caption("Statistical Discovery berdiri sendiri. Hasilnya hanya menjadi rekomendasi fokus untuk Analisa Teknikal.")

    period = STAT_PERIODS[period_label]
    forward_days = STAT_FORWARD_MAP[forward_label]

    with st.spinner(f"Menganalisis data statistik {ticker}..."):
        price = download_stat_data(ticker, period)
        market = download_stat_market(period)

    if price.empty or len(price) < 150:
        st.error("Data historis belum cukup untuk Statistical Discovery. Gunakan minimal sekitar 1 tahun data harian.")
        st.stop()

    result = statistical_discovery(price, market, forward_days, top_k=top_k)
    corr = result["corr"]
    reg = result["regression"]
    selected = result["selected"]

    if corr.empty:
        st.warning("Belum ditemukan hubungan statistik yang cukup kuat dari data yang tersedia.")
        st.stop()

    latest = result["data"].iloc[-1]
    latest_date = result["data"].index[-1]

    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observasi", f"{len(price):,}")
    c2.metric("Periode", period_label)
    c3.metric("Forward Target", forward_label)
    c4.metric("Faktor Terpilih", len(selected))

    st.markdown("### 🎯 Statistical Discovery — Faktor yang Paling Relevan")
    display = corr.head(top_k).copy()
    display["Correlation"] = display["Correlation"].map(lambda x: f"{x:.3f}")
    display["Abs Correlation"] = display["Abs Correlation"].map(lambda x: f"{x:.3f}")
    if "Stability" in display:
        display["Stability"] = display["Stability"].map(lambda x: f"{x*100:.0f}%" if pd.notna(x) else "-")
    if "Discovery Score" in display:
        display["Discovery Score"] = display["Discovery Score"].map(lambda x: f"{x*100:.0f}")
    render_table(display, max_height=330)

    st.info("Correlation digunakan untuk menemukan kandidat hubungan, bukan sebagai bukti bahwa hubungan tersebut pasti menyebabkan harga naik/turun.")

    # Heatmap
    st.markdown("### 🌡️ Correlation Heatmap")
    heat_cols = corr.head(min(top_k, 8))["Variable"].tolist() + ["Forward_Return"]
    heat_df = result["data"][heat_cols].replace([np.inf, -np.inf], np.nan).corr()
    fig = go.Figure(go.Heatmap(
        z=heat_df.values,
        x=heat_df.columns,
        y=heat_df.index,
        zmin=-1,
        zmax=1,
        colorscale="RdBu",
        text=np.round(heat_df.values, 2),
        texttemplate="%{text}",
        hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(height=430, template="plotly_dark", paper_bgcolor="#0c1117", plot_bgcolor="#0c1117", margin=dict(l=20,r=20,t=30,b=100))
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True})

    # Regression
    st.markdown("### 📐 Regression & Out-of-Sample Validation")
    if reg is None:
        st.warning("Data belum cukup untuk regression multivariat.")
    else:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Train R²", f"{reg['train_r2']*100:.1f}%")
        r2.metric("Adjusted R²", f"{reg['adj_r2']*100:.1f}%")
        r3.metric("Test R²", f"{reg['test_r2']*100:.1f}%" if pd.notna(reg['test_r2']) else "-")
        r4.metric("OOS Direction", f"{reg['directional_accuracy']*100:.1f}%" if pd.notna(reg['directional_accuracy']) else "-")
        coef_show = reg["coef"].copy()
        coef_show["Coefficient"] = coef_show["Coefficient"].map(lambda x: f"{x:.6f}")
        coef_show["P-Value"] = coef_show["P-Value"].map(lambda x: f"{x:.4f}")
        coef_show["Significant"] = coef_show["Significant"].map(lambda x: "✅" if x else "—")
        render_table(coef_show)

    # Technical focus recommendation
    focus = technical_recommendation(result["data"], selected)
    st.markdown("### 🧭 Rekomendasi Fokus Analisa Teknikal")
    focus_text = " · ".join(f"**{x}**" for x in focus)
    render_html(f'''<div class="ai-engine"><div class="ai-engine-top"><div class="score-orb"><div class="score">{len(focus)}</div><div class="label">FAKTOR</div></div><div class="signal-panel"><div class="small">STATISTICAL TECHNICAL FOCUS</div><div style="font-size:1.05rem;font-weight:800;color:#f5f7fa;margin-top:8px;line-height:1.45">{focus_text}</div><div class="signal-pill">🔬 Dipilih dari Statistical Discovery</div></div></div><div class="v3-small">Gunakan daftar ini sebagai fokus saat berpindah ke menu 📈 Analisa Teknikal. Modul teknikal tetap berdiri sendiri dan tidak dicampur dengan proses discovery.</div></div>''')

    # Current state of selected indicators
    st.markdown("### 📍 Kondisi Faktor Saat Ini")
    state_rows = []
    for var in selected:
        if var not in result["data"].columns:
            continue
        val = result["data"][var].iloc[-1]
        if pd.isna(val):
            state = "Tidak tersedia"
            shown = "-"
        elif "Return" in var or "Volatility" in var or "Pct" in var or "Change" in var or "Slope" in var:
            shown = f"{val*100:.2f}%" if abs(val) < 2 else f"{val:.4f}"
            state = "Positif" if val > 0 else "Negatif"
        elif var == "RSI14":
            shown = f"{val:.1f}"
            state = "Overbought" if val > 70 else "Oversold" if val < 30 else "Normal"
        else:
            shown = f"{val:.4f}"
            state = "Positif" if val > 0 else "Negatif" if val < 0 else "Netral"
        state_rows.append({"Faktor": var, "Nilai Terakhir": shown, "Kondisi": state})
    render_table(pd.DataFrame(state_rows))

    st.markdown("### 🧪 Kesimpulan Mesin")
    if reg is not None and pd.notna(reg["directional_accuracy"]):
        oos = reg["directional_accuracy"]
        if oos >= 0.60:
            conclusion = "Hubungan yang ditemukan memiliki validasi out-of-sample yang cukup menarik untuk dijadikan fokus analisa teknikal."
        elif oos >= 0.52:
            conclusion = "Hubungan masih lemah-moderat. Gunakan sebagai informasi tambahan, bukan sinyal entry tunggal."
        else:
            conclusion = "Hubungan belum cukup kuat pada data out-of-sample; jangan jadikan sebagai basis entry tunggal."
    else:
        conclusion = "Belum cukup data untuk validasi out-of-sample yang memadai."
    st.warning("⚠️ " + conclusion)
    st.caption(f"Data terakhir: {pd.Timestamp(latest_date).strftime('%d-%m-%Y')} • Forward target: {forward_label} • Ini adalah analisis statistik historis, bukan jaminan return masa depan.")



# =========================================================
# IDX SECTOR MODULE
# =========================================================

IDX_SECTORS = {
    "energy": {
        "name": "ENERGY",
        "full_name": "Energy",
        "icon": "🔥",
        "symbol": "JKENERGY",
        "google_symbol": "IDXENERGY:IDX",
        "url": "https://id.investing.com/indices/indonesia-se-energy",
    },
    "basic": {
        "name": "BASIC-IND",
        "full_name": "Basic Materials",
        "icon": "🧪",
        "symbol": "JKBASIC",
        "google_symbol": "IDXBASIC:IDX",
        "url": "https://id.investing.com/indices/indonesia-se-basic-materials",
    },
    "industrial": {
        "name": "INDUSTRIAL",
        "full_name": "Industrials",
        "icon": "🏭",
        "symbol": "JKINDUST",
        "google_symbol": "IDXINDUST:IDX",
        "url": "https://id.investing.com/indices/indonesia-se-industrials",
    },
    "noncyclical": {
        "name": "NON-CYCLICAL",
        "full_name": "Consumer Non-Cyclicals",
        "icon": "🛒",
        "symbol": "JKNONCYC",
        "google_symbol": "IDXNONCYC:IDX",
        "url": "https://id.investing.com/indices/indonesia-se-consumer-noncyclicals",
    },
    "cyclical": {
        "name": "CYCLICAL",
        "full_name": "Consumer Cyclicals",
        "icon": "👕",
        "symbol": "JKCYCLIC",
        "google_symbol": "IDXCYCLIC:IDX",
        "url": "https://id.investing.com/indices/indonesia-se-consumer-cyclical",
    },
    "finance": {
        "name": "FINANCE",
        "full_name": "Financials",
        "icon": "💰",
        "symbol": "JKFINANCE",
        "google_symbol": "IDXFINANCE:IDX",
        "url": "https://id.investing.com/indices/idx-finance",
    },
    "health": {
        "name": "HEALTH",
        "full_name": "Healthcare",
        "icon": "🏥",
        "symbol": "JKHEALTH",
        "google_symbol": "IDXHEALTH:IDX",
        "url": "https://id.investing.com/indices/indonesia-se-healthcare",
    },
    "property": {
        "name": "PROPERTY",
        "full_name": "Properties & Real Estate",
        "icon": "🏠",
        "symbol": "JKPROP",
        "google_symbol": "IDXPROPERT:IDX",
        "url": "https://id.investing.com/indices/idx-cons.-property---real-estate",
    },
    "technology": {
        "name": "TECHNOLOGY",
        "full_name": "Technology",
        "icon": "💻",
        "symbol": "JKTECHNO",
        "google_symbol": "IDXTECHNO:IDX",
        "url": "https://id.investing.com/indices/indonesia-se-technology",
    },
    "infra": {
        "name": "INFRASTRUC",
        "full_name": "Infrastructure",
        "icon": "🛣️",
        "symbol": "JKINFRA",
        "google_symbol": "IDXINFRA:IDX",
        "url": "https://id.investing.com/indices/idx-infrastructure",
    },
    "transport": {
        "name": "TRANSPORT",
        "full_name": "Transportation & Logistics",
        "icon": "✈️",
        "symbol": "JKTRANS",
        "google_symbol": "IDXTRANS:IDX",
        "url": "https://id.investing.com/indices/indonesia-se-transportation",
    },
}

@st.cache_data(ttl=300, show_spinner=False)
def idx_get_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/139.0 Safari/537.36",
        "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    })
    return session

def idx_parse_number(value):
    if value is None:
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    text = (text.replace("Rp","").replace("%","").replace("+","")
            .replace("−","-").replace("–","-").strip())
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".","").replace(",",".")
        else:
            text = text.replace(",","")
    elif "," in text:
        parts=text.split(",")
        text = text.replace(",",".") if len(parts[-1]) <= 2 else text.replace(",","")
    elif "." in text:
        parts=text.split(".")
        if len(parts)==2 and len(parts[1])==3:
            text=text.replace(".","")
    try:
        return float(text)
    except Exception:
        return np.nan

@st.cache_data(ttl=300, show_spinner=False)
def idx_fetch_page(url):
    try:
        r = requests.get(
            url,
            headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/139.0 Safari/537.36",
                     "Accept-Language":"id-ID,id;q=0.9,en;q=0.8"},
            timeout=20,
        )
        return r.text if r.status_code == 200 else None
    except Exception:
        return None

def idx_parse_index_page(html, symbol):
    if not html:
        return {"price":np.nan,"change":np.nan,"status":"NO DATA"}
    soup=BeautifulSoup(html,"html.parser")
    text=soup.get_text(" ",strip=True)
    pattern=(rf"{re.escape(symbol)}.{0,1500}?"
             r"(\d[\d.,]*)\s+([+-]?\d[\d.,]*)\s*\(([+-]?\d[\d.,]*)%\)")
    matches=re.findall(pattern,text,flags=re.IGNORECASE|re.DOTALL)
    for m in matches:
        price=idx_parse_number(m[0]); change=idx_parse_number(m[2])
        if not np.isnan(price) and not np.isnan(change):
            return {"price":price,"change":change,"status":"OK"}
    pattern2=r"(\d[\d.,]*)\s+([+-]?\d[\d.,]*)\s*\(([+-]?\d[\d.,]*)%\)"
    for m in re.findall(pattern2,text):
        price=idx_parse_number(m[0]); change=idx_parse_number(m[2])
        if not np.isnan(price) and not np.isnan(change) and price > 10:
            return {"price":price,"change":change,"status":"OK"}
    return {"price":np.nan,"change":np.nan,"status":"PARSE FAILED"}

@st.cache_data(ttl=300, show_spinner=False)
def google_finance_quote(symbol):
    """Fallback quote source for Community Cloud."""
    try:
        url = f"https://www.google.com/finance/quote/{symbol}?hl=id"
        r = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/139.0 Safari/537.36",
                "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
            },
            timeout=20,
        )
        if r.status_code != 200:
            return {"price":np.nan,"change":np.nan,"status":f"HTTP {r.status_code}"}
        text = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
        key = re.escape(symbol.split(":")[0])
        patterns = [
            rf"{key}.*?(\d[\d.,]*)\s*([+-]\d[\d.,]*)\s*%\s*\(",
            rf"{key}.*?(\d[\d.,]*)\s*([+-]\d[\d.,]*)\s*%",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.I|re.S)
            if m:
                price = idx_parse_number(m.group(1))
                change = idx_parse_number(m.group(2))
                if np.isfinite(price) and np.isfinite(change) and price > 10:
                    return {"price":price,"change":change,"status":"OK"}
        return {"price":np.nan,"change":np.nan,"status":"PARSE FAILED"}
    except Exception as e:
        return {"price":np.nan,"change":np.nan,"status":f"ERROR {type(e).__name__}"}


@st.cache_data(ttl=300, show_spinner=False)
def idx_get_sector(code):
    cfg=IDX_SECTORS[code]
    html=idx_fetch_page(cfg["url"])
    result=idx_parse_index_page(html,cfg["symbol"])

    # Community Cloud fallback: Investing.com can return an anti-bot page.
    if not (np.isfinite(result["price"]) and np.isfinite(result["change"])):
        gf = google_finance_quote(cfg.get("google_symbol", "")) if cfg.get("google_symbol") else None
        if gf and np.isfinite(gf["price"]) and np.isfinite(gf["change"]):
            result = gf

    return {
        "code":code, "name":cfg["name"], "full_name":cfg["full_name"],
        "icon":cfg["icon"], "symbol":cfg["symbol"],
        "price":result["price"], "change":result["change"],
        "status":result["status"], "url":cfg["url"]
    }

@st.cache_data(ttl=300, show_spinner=False)
def idx_get_all_sectors():
    return [idx_get_sector(code) for code in IDX_SECTORS]

@st.cache_data(ttl=300, show_spinner=False)
def idx_get_ihsg_overview():
    # Primary: Yahoo Finance ^JKSE.
    try:
        raw = yf.download(
            "^JKSE", period="5d", interval="1d", auto_adjust=False,
            progress=False, threads=False, group_by="column"
        )
        raw = _flat_ohlcv(raw)
        if not raw.empty:
            close = float(raw["Close"].iloc[-1])
            change = np.nan
            if len(raw) >= 2:
                prev = float(raw["Close"].iloc[-2])
                if prev:
                    change = (close / prev - 1) * 100
            if np.isfinite(close):
                return {"price":close,"change":change,"status":"OK"}
    except Exception:
        pass
    return google_finance_quote("COMPOSITE:IDX")


def render_idx_sector_module():
    st.markdown("## 🇮🇩 SEKTORAL IDX")
    st.caption("Kondisi 11 sektor IDX-IC. Data ditampilkan sebagai market overview, bukan hasil perhitungan seasonality.")

    sectors = idx_get_all_sectors()
    valid = [x for x in sectors if pd.notna(x["change"])]
    ihsg = idx_get_ihsg_overview()

    if valid:
        best=max(valid,key=lambda x:x["change"])
        worst=min(valid,key=lambda x:x["change"])
        avg=float(np.mean([x["change"] for x in valid]))
        a,b,c,d=st.columns(4)
        a.metric("IHSG", f'{ihsg["price"]:,.2f}' if np.isfinite(ihsg["price"]) else "N/A",
                 f'{ihsg["change"]:+.2f}%' if np.isfinite(ihsg["change"]) else None)
        b.metric("Sektor Terkuat", best["full_name"], f'{best["change"]:+.2f}%')
        c.metric("Sektor Terlemah", worst["full_name"], f'{worst["change"]:+.2f}%')
        d.metric("Rata-rata 11 Sektor", f"{avg:+.2f}%")
    else:
        st.warning("Data sektor IDX belum berhasil dibaca dari sumber data.")
        if np.isfinite(ihsg["price"]):
            st.metric("IHSG", f'{ihsg["price"]:,.2f}',
                      f'{ihsg["change"]:+.2f}%' if np.isfinite(ihsg["change"]) else None)

    st.markdown("### 📊 Kondisi Sektor Hari Ini")
    for start in range(0,len(sectors),3):
        cols=st.columns(3,gap="medium")
        for col,data in zip(cols,sectors[start:start+3]):
            with col:
                with st.container(border=True):
                    st.markdown(f"### {data['icon']} {data['name']}")
                    st.caption(f"{data['full_name']} • {data['symbol']}")
                    if pd.notna(data["change"]):
                        st.metric("Perubahan 1D", f'{data["change"]:+.2f}%',
                                  delta=f'{data["change"]:+.2f}%')
                    else:
                        st.metric("Perubahan 1D","N/A")
                    if pd.notna(data["price"]):
                        st.caption(f'Index: {data["price"]:,.2f}')
                    else:
                        st.caption(f'Status data: {data["status"]}')
                    if st.button("Lihat sumber", key=f"idx_src_{data['code']}", use_container_width=True):
                        st.markdown(f'[{data["url"]}]({data["url"]})')

    st.divider()
    st.caption("Sumber data sektoral: Investing.com • Refresh cache sekitar 5 menit.")


# =========================================================
# UNIFIED SIDEBAR
# =========================================================

st.sidebar.header("⚙️ DASHBOARD SETTINGS")

analysis_mode = st.sidebar.radio(
    "MENU UTAMA",
    [
        "📅 Screening",
        "🇮🇩 Sektoral IDX",
        "🔬 Analisa Statistik",
        "📊 Analisa Saham"
    ],
    index=0
)

st.sidebar.divider()

ticker_input = st.sidebar.text_input("Kode Saham", "BBRI")
ticker_input = ticker_input.upper().strip().replace(".JK", "")

# Internal Yahoo Finance symbol. The user only sees the clean ticker code.
ticker = ticker_input

if analysis_mode.startswith("📅"):
    st.sidebar.subheader("📅 Screening Seasonality")
    period_label = st.sidebar.selectbox(
        "Seasonality History",
        list(PERIOD_OPTIONS.keys()),
        index=3
    )
    period = PERIOD_OPTIONS[period_label]
    # Similar Stocks is no longer rendered in the main seasonality panel.
    # Broker Flow replaces that section.
    analyze = True
elif analysis_mode.startswith("🇮🇩"):
    st.sidebar.subheader("🇮🇩 Sektoral IDX")
    st.sidebar.caption("Overview kondisi 11 sektor IDX-IC.")
    analyze = True
elif analysis_mode.startswith("🔬"):
    st.sidebar.subheader("🧪 Statistical Discovery Settings")
    stat_period_label = st.sidebar.selectbox(
        "Periode Data Statistik",
        list(STAT_PERIODS.keys()),
        index=3,
        help="Periode histori harian yang digunakan untuk correlation, regression, stability, dan out-of-sample validation."
    )
    stat_forward_label = st.sidebar.selectbox(
        "Target Forward Return",
        list(STAT_FORWARD_MAP.keys()),
        index=2,
        help="Return masa depan yang dicari hubungan statistiknya."
    )
    stat_top_k = st.sidebar.slider("Jumlah Faktor yang Dicari", 3, 8, 6)
    st.sidebar.caption("🔬 Modul ini terpisah dari Technical, Seasonality, dan News.")
    st.sidebar.caption("📈 Hasilnya hanya menjadi rekomendasi fokus indikator untuk Technical Analysis.")
    analyze = True
else:
    st.sidebar.subheader("⏱️ Time Frame")
    timeframe = st.sidebar.selectbox(
        "Interval Data",
        list(TIMEFRAME_CONFIG.keys()),
        index=3
    )
    tf_config = TIMEFRAME_CONFIG[timeframe]
    interval = tf_config["interval"]

    st.sidebar.subheader("📅 Periode Data")
    today = date.today()
    default_days = tf_config["default_days"]
    default_start = today - timedelta(days=default_days)
    start_date = st.sidebar.date_input("Tanggal Mulai", value=default_start)
    end_date = st.sidebar.date_input("Tanggal Akhir", value=today)

    if start_date > end_date:
        st.sidebar.error("Tanggal mulai harus ≤ tanggal akhir.")
        st.stop()

    days_requested = (end_date - start_date).days
    max_days = tf_config["max_days"]
    if max_days is not None and days_requested > max_days:
        st.sidebar.warning(
            f"Maksimal sekitar {max_days} hari untuk {timeframe}."
        )
        st.info(f"⚠️ Data {timeframe} memiliki keterbatasan histori intraday.")
        st.stop()

    st.sidebar.subheader("📐 Technical Analysis")
    lookback_sr = st.sidebar.slider(
        "Lookback Support / Resistance", 20, 200, 60, 10
    )

    st.sidebar.subheader("👁️ Chart Levels")
    show_support = st.sidebar.checkbox("Support", True)
    show_resistance = st.sidebar.checkbox("Resistance", True)
    show_tp1 = st.sidebar.checkbox("TP1", True)
    show_tp2 = st.sidebar.checkbox("TP2", True)
    show_sl1 = st.sidebar.checkbox("SL1", True)
    show_sl2 = st.sidebar.checkbox("SL2", True)
    show_price = st.sidebar.checkbox("Harga", True)

    st.sidebar.subheader("📊 Indicator Visibility")
    show_rsi = st.sidebar.checkbox("RSI 14", False)
    show_macd = st.sidebar.checkbox("MACD", False)
    show_cmo = st.sidebar.checkbox("CMO 14", False)
    show_bollinger = st.sidebar.checkbox("Bollinger Band", False)
    show_volume_chart = st.sidebar.checkbox("Volume", False)
    st.sidebar.caption("🕯️ Candlestick / Price Action selalu tampil.")
    st.sidebar.caption("🌫️ IHSG transparan di belakang candle sebagai market context.")
    st.sidebar.caption("📍 Panel indikator muncul di bawah Price Action.")

    analyze = True


# =========================================================
# STOCK HEADER
# =========================================================

if ticker_input and not analysis_mode.startswith("🇮🇩"):
    stock_rows = master[master["Ticker"] == ticker_input]
    if not stock_rows.empty:
        stock_info = stock_rows.iloc[0]
        sektor = stock_info["Sektor"] if "Sektor" in master.columns else "-"
        sub_sektor = stock_info["Sub Sektor"] if "Sub Sektor" in master.columns else "-"
        st.subheader(f"{ticker_input}  |  {sektor}")
        st.caption(
            f"{sub_sektor}  •  "
            f"{'Screening' if analysis_mode.startswith('📅') else 'Sektoral IDX' if analysis_mode.startswith('🇮🇩') else 'Statistical Discovery' if analysis_mode.startswith('🔬') else 'Technical Analysis'}"
        )



st.markdown("""
<div class="dashboard-header">
  <div class="dashboard-title">📊 STOCK ANALYSIS TERMINAL</div>
  <div class="dashboard-subtitle">Technical Analysis • Statistical Discovery • Seasonality • News & Sentiment • Yahoo Finance</div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# IDX SECTOR MODE
# =========================================================
if analysis_mode.startswith("🇮🇩"):
    render_idx_sector_module()
    st.markdown("<div class=\"footer\">IDX Market • Sector Overview • Investing.com</div>", unsafe_allow_html=True)
    st.stop()

# =========================================================
# STATISTICAL ANALYSIS MODE
# =========================================================
if analysis_mode.startswith("🔬"):
    render_statistical_module(ticker, stat_period_label, stat_forward_label, stat_top_k)
    st.markdown("<div class=\"footer\">Yahoo Finance • Statistical Discovery</div>", unsafe_allow_html=True)
    st.stop()


# SEASONALITY MODE
# =========================================================
# =========================================================

if analysis_mode.startswith("📅"):

    # =====================================================
    # DOWNLOAD STOCK
    # =====================================================

    with st.spinner(
        f"Mengambil data {ticker}..."
    ):

        stock_data = download_stock(
            ticker,
            period
        )

    if stock_data.empty:

        st.error(
            f"Data {ticker} tidak tersedia."
        )

        st.stop()

    complete_data = remove_current_month(
        stock_data
    )

    if complete_data.empty:

        st.error(
            "Belum tersedia bulan historis yang selesai."
        )

        st.stop()

    seasonal = calculate_seasonality(
        stock_data
    )

    weekly_summary = (
        calculate_weekly_summary(
            stock_data
        )
    )

    if seasonal.empty:

        st.error(
            "Tidak cukup data untuk seasonality."
        )

        st.stop()

    first_date = (
        complete_data["Date"]
        .min()
        .strftime("%d %b %Y")
    )

    last_date = (
        complete_data["Date"]
        .max()
        .strftime("%d %b %Y")
    )

    current_analysis = (
        create_current_month_analysis(
            seasonal,
            period_label,
            first_date,
            last_date
        )
    )

    current_week_decision = (
        get_current_month_week_decision(
            weekly_summary
        )
    )

    # =====================================================
    # CURRENT MONTH
    # =====================================================

    st.divider()

    st.subheader(
        "🎯 CURRENT MONTH SEASONALITY"
    )

    if (
        current_analysis is None
        or
        not current_analysis["available"]
    ):

        st.warning(
            f"Belum tersedia data historis "
            f"untuk {current_analysis['month']} "
            f"dalam periode {period_label}."
        )

    else:

        ca = current_analysis

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Month",
            ca["month"]
        )

        c2.metric(
            "Average Return",
            f"{ca['avg_return']:.2%}"
        )

        c3.metric(
            "Median Return",
            f"{ca['median_return']:.2%}"
        )

        c4.metric(
            "Win Rate",
            f"{ca['win_rate']:.1%}"
        )

        c5.metric(
            "Observations",
            str(ca["observations"])
        )

        if ca["decision"] == "FAVORABLE":

            st.success(
                f"🟢 **SEASONALITY: FAVORABLE**  |  "
                f"**{ca['evidence']}**"
            )

        elif ca["decision"] == "UNFAVORABLE":

            st.error(
                f"🔴 **SEASONALITY: UNFAVORABLE**  |  "
                f"**{ca['evidence']}**"
            )

        else:

            st.warning(
                f"🟡 **SEASONALITY: NEUTRAL / MIXED**  |  "
                f"**{ca['evidence']}**"
            )

        e1, e2, e3, e4, e5 = st.columns(5)

        e1.metric(
            "Bullish",
            f"{ca['bullish']} / {ca['observations']}"
        )

        e2.metric(
            "Bearish",
            f"{ca['bearish']} / {ca['observations']}"
        )

        e3.metric(
            "Flat",
            f"{ca['flat']} / {ca['observations']}"
        )

        e4.metric(
            "History",
            period_label
        )

        e5.metric(
            "Completed Until",
            last_date
        )

        st.caption(
            f"Seasonality dihitung dari "
            f"{first_date} sampai {last_date}. "
            f"Bulan berjalan tidak dimasukkan ke histori."
        )

        if ca["observations"] < 3:

            st.warning(
                "⚠️ Evidence sangat terbatas. "
                "Seasonality belum cukup kuat untuk "
                "dijadikan dasar utama entry."
            )

        elif ca["observations"] < 5:

            st.warning(
                "⚠️ Evidence masih rendah. "
                "Gunakan seasonality sebagai faktor pendukung."
            )

        else:

            st.info(
                "ℹ️ Evidence relatif lebih memadai "
                "karena tersedia minimal 5 observasi."
            )

    # =====================================================
    # MONTHLY SEASONALITY
    # =====================================================

    st.divider()

    st.subheader(
        "📅 MONTHLY SEASONALITY"
    )

    left, right = st.columns(
        [1.25, 1],
        gap="small"
    )

    with left:

        fig = go.Figure()

        for _, row in seasonal.iterrows():

            value = row[
                "Average_Return"
            ]

            color = (
                "#35d07f"
                if value >= 0
                else "#ff5c5c"
            )

            fig.add_trace(
                go.Bar(
                    x=[
                        row["Month_Name"]
                    ],
                    y=[value],
                    marker_color=color,
                    showlegend=False
                )
            )

        fig.add_hline(
            y=0,
            line_width=1,
            line_color="#888888"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#11161d",
            plot_bgcolor="#11161d",
            title="Average Monthly Return",
            title_font_size=13,
            yaxis_tickformat=".1%",
            height=330,
            margin=dict(
                l=10,
                r=10,
                t=38,
                b=5
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

    with right:

        monthly_display = seasonal[
            [
                "Month_Name",
                "Average_Return",
                "Median_Return",
                "Win_Rate",
                "Bullish",
                "Bearish",
                "Observations"
            ]
        ].copy()

        monthly_display.columns = [
            "Month",
            "Average",
            "Median",
            "Win Rate",
            "Bullish",
            "Bearish",
            "Obs"
        ]

        st.dataframe(
            monthly_display.style.format(
                {
                    "Average":
                    "{:.2%}",

                    "Median":
                    "{:.2%}",

                    "Win Rate":
                    "{:.1%}",

                    "Bullish":
                    "{:.0f}",

                    "Bearish":
                    "{:.0f}",

                    "Obs":
                    "{:.0f}"
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=330
        )

    # =====================================================
    # HEATMAP + WEEKLY TIMING
    # =====================================================

    st.divider()

    st.subheader(
        "🔥 SEASONALITY HEATMAP & WEEKLY TIMING"
    )

    heatmap_col, timing_col = st.columns(
        [1.65, 1],
        gap="small"
    )

    with heatmap_col:

        st.markdown(
            "**Historical Year × Month — Net Daily Movement**"
        )

        monthly_movement = calculate_monthly_net_daily_movement(
            stock_data
        )

        if monthly_movement.empty:

            st.warning(
                "Belum tersedia data heatmap."
            )

        else:

            # Heatmap value = total bullish daily movement + total bearish
            # daily movement for each completed Year x Month.
            historical_heatmap = (
                monthly_movement
                .pivot(
                    index="Year",
                    columns="Month",
                    values="Net_Daily_Movement"
                )
                .reindex(columns=range(1, 13))
            )

            historical_heatmap.columns = [
                MONTH_NAMES[x]
                for x in historical_heatmap.columns
            ]

            historical_text = historical_heatmap.apply(
                lambda column: column.map(
                    lambda x: ""
                    if pd.isna(x)
                    else f"{x:+.1f}%"
                )
            )

            current_month_short = (
                MONTH_NAMES[datetime.now().month]
            )

            current_month_index = (
                list(historical_heatmap.columns).index(
                    current_month_short
                )
                if current_month_short in historical_heatmap.columns
                else None
            )

            fig_heat = go.Figure(
                data=go.Heatmap(
                    z=historical_heatmap.values,
                    x=historical_heatmap.columns,
                    y=[str(x) for x in historical_heatmap.index],
                    text=historical_text.values,
                    texttemplate="%{text}",
                    textfont=dict(
                        size=9,
                        color="white"
                    ),
                    colorscale=[
                        [0.00, "#b91c1c"],
                        [0.20, "#ef4444"],
                        [0.45, "#30343b"],
                        [0.50, "#18181b"],
                        [0.55, "#30343b"],
                        [0.80, "#22c55e"],
                        [1.00, "#15803d"]
                    ],
                    zmid=0,
                    colorbar=dict(
                        title="Net Daily Movement",
                        ticksuffix="%",
                        thickness=12
                    ),
                    hovertemplate=(
                        "<b>%{x} %{y}</b><br>"
                        "Net Daily Movement: %{z:+.2f}%<br>"
                        "<extra></extra>"
                    )
                )
            )

            fig_heat.update_layout(
                template="plotly_dark",
                paper_bgcolor="#11161d",
                plot_bgcolor="#11161d",
                height=450,
                margin=dict(
                    l=15,
                    r=5,
                    t=18,
                    b=8
                ),
                xaxis=dict(
                    side="top",
                    tickfont=dict(size=10)
                ),
                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(size=10)
                )
            )

            if current_month_index is not None:
                fig_heat.add_vline(
                    x=current_month_index,
                    line_width=2,
                    line_dash="dash",
                    line_color="#f5a623"
                )

            st.plotly_chart(
                fig_heat,
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )

            st.caption(
                f"Histori {period_label}. "
                f"Heatmap = total return harian bullish + total return "
                f"harian bearish dalam bulan tersebut, tanpa dibagi jumlah "
                f"hari trading. Garis kuning = bulan berjalan "
                f"({current_month_short})."
            )

            # Detail bulan terakhir untuk memudahkan audit angka heatmap.
            st.markdown("**Detail Net Daily Movement**")

            detail_movement = monthly_movement.copy()
            detail_movement["Month"] = detail_movement["Month"].map(MONTH_NAMES)
            detail_movement = detail_movement.rename(
                columns={
                    "Year": "Tahun",
                    "Month": "Bulan",
                    "Bullish_Movement": "Bullish Movement",
                    "Bearish_Movement": "Bearish Movement",
                    "Net_Daily_Movement": "Net Daily Movement",
                    "Bullish_Days": "Bullish Days",
                    "Bearish_Days": "Bearish Days",
                    "Flat_Days": "Flat Days",
                    "Trading_Days": "Trading Days",
                    "Bullish_Rate": "Bullish Rate"
                }
            )

            detail_movement = detail_movement[
                [
                    "Tahun",
                    "Bulan",
                    "Bullish Movement",
                    "Bearish Movement",
                    "Net Daily Movement",
                    "Bullish Days",
                    "Bearish Days",
                    "Flat Days",
                    "Trading Days",
                    "Bullish Rate"
                ]
            ].copy()

            st.dataframe(
                detail_movement.style.format(
                    {
                        "Bullish Movement": "{:+.2f}%",
                        "Bearish Movement": "{:+.2f}%",
                        "Net Daily Movement": "{:+.2f}%",
                        "Bullish Rate": "{:.1%}"
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=240
            )

    with timing_col:

        current_month_number = (
            datetime.now().month
        )

        current_month_name = (
            MONTH_NAMES_FULL[
                current_month_number
            ]
        )

        st.markdown(
            f"**📅 Weekly Timing — "
            f"{current_month_name}**"
        )

        current_week_summary = (
            weekly_summary[
                weekly_summary["Month"]
                == current_month_number
            ]
            .sort_values("Week")
            .copy()
        )

        if current_week_summary.empty:

            st.warning(
                "Belum tersedia weekly seasonality "
                "untuk bulan berjalan."
            )

        else:

            timing_display = (
                current_week_summary[
                    [
                        "Week_Name",
                        "Average_Return",
                        "Median_Return",
                        "Win_Rate",
                        "Bullish",
                        "Bearish",
                        "Observations"
                    ]
                ]
                .copy()
            )

            timing_display.columns = [
                "Week",
                "Average",
                "Median",
                "Win Rate",
                "Bull",
                "Bear",
                "Obs"
            ]

            st.dataframe(
                timing_display.style.format(
                    {
                        "Average":
                        "{:.2%}",

                        "Median":
                        "{:.2%}",

                        "Win Rate":
                        "{:.1%}",

                        "Bull":
                        "{:.0f}",

                        "Bear":
                        "{:.0f}",

                        "Obs":
                        "{:.0f}"
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=215
            )

            if current_week_decision is not None:

                best_entry = (
                    current_week_decision[
                        "best_entry"
                    ]
                )

                weakest = (
                    current_week_decision[
                        "weakest"
                    ]
                )

                if best_entry is not None:

                    st.success(
                        f"🟢 **Best Entry: "
                        f"{best_entry['Week_Name']}**\n\n"
                        f"Avg "
                        f"**{best_entry['Average_Return']:.2%}** • "
                        f"Win Rate "
                        f"**{best_entry['Win_Rate']:.1%}**"
                    )

                else:

                    st.warning(
                        "Tidak ada week yang memenuhi "
                        "kriteria entry kuat."
                    )

                if weakest is not None:

                    st.error(
                        f"🔴 **Weakest Phase: "
                        f"{weakest['Week_Name']}**\n\n"
                        f"Avg "
                        f"**{weakest['Average_Return']:.2%}** • "
                        f"Win Rate "
                        f"**{weakest['Win_Rate']:.1%}**"
                    )

                else:

                    st.info(
                        "Tidak ada week dengan kelemahan "
                        "historis yang kuat."
                    )

            st.caption(
                "Week 1–4 berdasarkan urutan "
                "trading day setiap bulan."
            )

    # =====================================================
    # =====================================================
    # HISTORICAL ZODIAC HEATMAP
    # =====================================================

    with st.expander(
        "♈ Historical Heatmap berdasarkan Zodiac",
        expanded=False
    ):

        zodiac_heatmap = calculate_zodiac_historical_returns(
            stock_data
        )

        if zodiac_heatmap.empty:

            st.info(
                "Belum tersedia data yang cukup untuk "
                "Historical Zodiac Heatmap."
            )

        else:

            zodiac_text = format_zodiac_heatmap_text(
                zodiac_heatmap
            )

            fig_zodiac = go.Figure(
                data=go.Heatmap(
                    z=zodiac_heatmap.values,
                    x=zodiac_heatmap.columns,
                    y=[
                        str(x)
                        for x in zodiac_heatmap.index
                    ],
                    text=zodiac_text.values,
                    texttemplate="%{text}",
                    textfont=dict(
                        size=9
                    ),
                    colorscale=[
                        [0.00, "#b91c1c"],
                        [0.50, "#18181b"],
                        [1.00, "#15803d"]
                    ],
                    zmid=0,
                    colorbar=dict(
                        title="Return",
                        tickformat=".0%"
                    ),
                    hovertemplate=
                    "<b>%{x}</b><br>"
                    "Year: %{y}<br>"
                    "Historical Return: %{z:.2%}"
                    "<extra></extra>"
                )
            )

            fig_zodiac.update_layout(
                template="plotly_dark",
                paper_bgcolor="#11161d",
                plot_bgcolor="#11161d",
                height=390,
                yaxis=dict(
                    autorange="reversed",
                    title="Year"
                ),
                xaxis=dict(
                    title="Zodiac Period"
                ),
                margin=dict(
                    l=15,
                    r=15,
                    t=15,
                    b=10
                )
            )

            st.plotly_chart(
                fig_zodiac,
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )

            # Zodiac summary
            zodiac_summary = (
                zodiac_heatmap
                .mean(axis=0)
                .to_frame("Average Return")
            )

            zodiac_summary["Win Rate"] = (
                (zodiac_heatmap > 0)
                .mean(axis=0)
            )

            zodiac_summary["Observations"] = (
                zodiac_heatmap.notna()
                .sum(axis=0)
            )

            zodiac_summary = (
                zodiac_summary
                .reindex(ZODIAC_ORDER)
            )

            st.markdown(
                "### 📊 Ringkasan Performa Zodiac"
            )

            summary_display = zodiac_summary.copy()

            summary_display["Average Return"] = (
                summary_display["Average Return"]
                .apply(
                    lambda x:
                    ""
                    if pd.isna(x)
                    else f"{x:.2%}"
                )
            )

            summary_display["Win Rate"] = (
                summary_display["Win Rate"]
                .apply(
                    lambda x:
                    ""
                    if pd.isna(x)
                    else f"{x:.1%}"
                )
            )

            summary_display.columns = [
                "Average Return",
                "Win Rate",
                "Observations"
            ]

            st.dataframe(
                summary_display,
                use_container_width=True,
                hide_index=False,
                height=460
            )

            best_zodiac = (
                zodiac_summary["Average Return"]
                .dropna()
                .idxmax()
                if not zodiac_summary["Average Return"]
                .dropna()
                .empty
                else None
            )

            worst_zodiac = (
                zodiac_summary["Average Return"]
                .dropna()
                .idxmin()
                if not zodiac_summary["Average Return"]
                .dropna()
                .empty
                else None
            )

            if best_zodiac is not None:
                best_return = zodiac_summary.loc[
                    best_zodiac,
                    "Average Return"
                ]
                best_win = zodiac_summary.loc[
                    best_zodiac,
                    "Win Rate"
                ]

                st.success(
                    f"🟢 **Zodiac historis terkuat: "
                    f"{best_zodiac}** — "
                    f"Average Return **{best_return:.2%}**, "
                    f"Win Rate **{best_win:.1%}**"
                )

            if worst_zodiac is not None:
                worst_return = zodiac_summary.loc[
                    worst_zodiac,
                    "Average Return"
                ]
                worst_win = zodiac_summary.loc[
                    worst_zodiac,
                    "Win Rate"
                ]

                st.error(
                    f"🔴 **Zodiac historis terlemah: "
                    f"{worst_zodiac}** — "
                    f"Average Return **{worst_return:.2%}**, "
                    f"Win Rate **{worst_win:.1%}**"
                )

            st.caption(
                "Return dihitung sebagai compounded daily return "
                "untuk setiap periode zodiac pada masing-masing tahun. "
                "Zodiac digunakan sebagai analisis seasonality tambahan, "
                "bukan sebagai sinyal entry tunggal."
            )

    # =====================================================
    # BROKER / BANDAR FLOW — replaces STOCK VS IHSG + SIMILAR STOCKS
    # =====================================================
    broker_window = st.sidebar.selectbox(
        "Broker Flow History",
        [5, 20, 40, 60, 120, 250],
        index=0,
        help="Jumlah hari perdagangan yang ditampilkan pada Broker Flow."
    )

    render_broker_flow_module(
        ticker=ticker,
        stock_data=stock_data,
        end_date=pd.to_datetime(stock_data["Date"]).max(),
        broker_window=broker_window,
    )

    # =========================================================
# TECHNICAL ANALYSIS MODE — ORIGINAL ANALYSIS ENGINE
# =========================================================
if analysis_mode.startswith("📊"):


    # Internal Yahoo Finance symbol used by the original technical engine.
    ticker = ticker_input + ".JK"

    # LOAD DATA
    # =========================================================

    if not ticker_input:

        st.warning(
            "Masukkan kode saham."
        )

        st.stop()


    try:

        buffer_days = 400

        download_start = (
            pd.Timestamp(start_date) -
            pd.Timedelta(days=buffer_days)
        ).date()

        download_end = (
            pd.Timestamp(end_date) +
            pd.Timedelta(days=1)
        ).date()

        raw = ambil_data(
            ticker,
            download_start,
            download_end,
            interval
        )

        if raw.empty:

            st.error(
                f"Data {ticker_input} tidak ditemukan."
            )

            st.stop()

        df = add_indicators(
            raw
        )

        df.index = pd.to_datetime(
            df.index
        )

        if df.index.tz is not None:

            df.index = (
                df.index
                .tz_localize(None)
            )

        df = (
            df.sort_index()
            .dropna(
                subset=["Close"]
            )
        )

    except Exception as e:

        st.error(
            f"Gagal mengambil data: {e}"
        )

        st.stop()


    # =========================================================
    # FILTER
    # =========================================================

    range_start = pd.Timestamp(
        start_date
    )

    range_end = (
        pd.Timestamp(end_date) +
        pd.Timedelta(days=1)
    )

    hist = df[
        (
            df.index >= range_start
        )
        &
        (
            df.index < range_end
        )
    ].copy()


    if hist.empty:

        st.error(
            "Tidak ada data pada periode tersebut."
        )

        st.stop()


    # =========================================================
    # IHSG OVERLAY DATA - ROBUST
    # =========================================================
    # IHSG hanya sebagai market context visual. Tidak masuk ke AI score,
    # regression, backtest, atau indikator saham.
    #
    # OHLC IHSG dinormalisasi ke skala harga saham berdasarkan close awal.
    # Tujuannya membandingkan PERGERAKAN market dengan saham, bukan level
    # absolut IHSG vs harga saham.
    try:
        ihsg_raw = ambil_data("^JKSE", download_start, download_end, interval)

        # Fallback ke daily bila Yahoo tidak mengembalikan data IHSG pada
        # interval yang dipilih.
        if ihsg_raw.empty:
            ihsg_raw = ambil_data("^JKSE", download_start, download_end, "1d")

        if not ihsg_raw.empty:
            ihsg_raw.index = pd.to_datetime(ihsg_raw.index)
            if getattr(ihsg_raw.index, "tz", None) is not None:
                ihsg_raw.index = ihsg_raw.index.tz_localize(None)
            ihsg_raw = ihsg_raw.sort_index()
            ihsg_hist = ihsg_raw[
                (ihsg_raw.index >= range_start) &
                (ihsg_raw.index < range_end)
            ].dropna(subset=["Open", "High", "Low", "Close"]).copy()
        else:
            ihsg_hist = pd.DataFrame()
    except Exception:
        ihsg_hist = pd.DataFrame()

    # Normalisasi IHSG ke skala harga saham agar candle IHSG dapat
    # menjadi background overlay pada chart yang sama.
    if not ihsg_hist.empty and not hist.empty:
        try:
            stock_ref = float(hist["Close"].iloc[0])
            ihsg_ref = float(ihsg_hist["Close"].iloc[0])
            if np.isfinite(stock_ref) and np.isfinite(ihsg_ref) and ihsg_ref != 0:
                ihsg_scale = stock_ref / ihsg_ref
                for col in ["Open", "High", "Low", "Close"]:
                    ihsg_hist[f"{col}_NORM"] = ihsg_hist[col] * ihsg_scale
            else:
                ihsg_hist = pd.DataFrame()
        except Exception:
            ihsg_hist = pd.DataFrame()


    # =========================================================
    # LATEST DATA
    # =========================================================

    latest = hist.iloc[-1]

    tanggal_data = hist.index[-1]

    close_now = float(
        latest["Close"]
    )

    volume_now = float(
        latest["Volume"]
    )

    rsi_now = float(
        latest["RSI14"]
    )

    cmo_now = float(
        latest["CMO14"]
    )

    macd_now = float(
        latest["MACD"]
    )

    macd_signal_now = float(
        latest["MACD_SIGNAL"]
    )

    cmf_now = float(
        latest["CMF20"]
    )

    current_score = int(
        latest["SCORE_HISTORIS"]
    )


    macd_bullish = (
        macd_now > macd_signal_now
    )

    cmf_positive = (
        cmf_now > 0
    )

    cmo_positive = (
        cmo_now > 0
    )

    volume_high = (
        volume_now >
        latest["VOL_MEDIAN20"]
    )


    # =========================================================
    # RSI STATUS
    # =========================================================

    if rsi_now < 30:

        rsi_status = "🔵 OVERSOLD"

    elif rsi_now > 70:

        rsi_status = "🔴 OVERBOUGHT"

    else:

        rsi_status = "🟡 NORMAL"


    # =========================================================
    # CMO STATUS
    # =========================================================

    if cmo_now >= 50:

        cmo_status = "🟢 MOMENTUM KUAT"

    elif cmo_now > 0:

        cmo_status = "🟢 MOMENTUM POSITIF"

    elif cmo_now <= -50:

        cmo_status = "🔴 MOMENTUM LEMAH"

    else:

        cmo_status = "🔴 MOMENTUM NEGATIF"


    # =========================================================
    # LEVELS
    # =========================================================

    levels = cari_support_resistance(
        hist,
        lookback_sr
    )

    support = levels["support"]
    support_low = levels["support_low"]
    support_high = levels["support_high"]

    resistance = levels["resistance"]
    resistance_low = levels["resistance_low"]
    resistance_high = levels["resistance_high"]

    atr_now = levels["atr"]


    tp1, tp2, sl1, sl2 = hitung_tp_sl(
        close_now,
        support,
        resistance,
        atr_now
    )


    # =========================================================
    # HISTORICAL DATA
    # =========================================================

    eval_data = hist.iloc[:-1].copy()

    score_stats = empirical_score_stats(
        eval_data
    )


    if (
        not score_stats.empty
        and
        current_score in
        score_stats["SCORE_HISTORIS"].values
    ):

        row = score_stats[
            score_stats["SCORE_HISTORIS"] ==
            current_score
        ].iloc[0]

        probability_up = float(
            row["Probabilitas_Naik"]
        )

        expectancy = float(
            row["Avg_Return"]
        )

    else:

        probability_up = np.nan
        expectancy = np.nan


    # =========================================================
    # V3 HEADER
    # =========================================================
    render_html(f"""
    <div class="dashboard-header" style="display:flex;justify-content:space-between;align-items:flex-start;gap:20px;">
      <div>
        <div class="dashboard-title"><span style="color:#d946ef">AI</span> STOCK ANALYSIS <span style="color:#7f8b99;font-size:.55em">GLOBAL MARKET TERMINAL v14</span></div>
        <div class="dashboard-subtitle"><span class="live-dot"></span>MARKET ANALYTICS ENGINE</div>
      </div>
      <div style="text-align:right;color:#7f8b99;font-size:.70rem;line-height:1.6;">
        Last update: {tanggal_data.strftime('%d-%m-%Y %H:%M')}<br>
        Data source: Yahoo Finance <span class="live-dot" style="margin-left:5px"></span>
      </div>
    </div>
    """)

    # =========================================================
    # TOP KPI CARDS
    # =========================================================
    c1,c2,c3,c4,c5,c6 = st.columns(6)

    kpi = [
        (c1,"💵 Harga",f"Rp {close_now:,.0f}",f"{((close_now / hist['Close'].iloc[-2])-1)*100:+.2f}%" if len(hist)>1 else "-"),
        (c2,"🟣 RSI 14",f"{rsi_now:.2f}",rsi_status.replace("🟡 ","").replace("🟢 ","").replace("🔴 ","")),
        (c3,"🟢 CMF 20",f"{cmf_now:.3f}","Akumulasi" if cmf_positive else "Distribusi"),
        (c4,"〽️ CMO 14",f"{cmo_now:.2f}","Momentum Positif" if cmo_positive else "Momentum Negatif"),
        (c5,"📊 Volume",f"{volume_now/1e6:,.2f} jt",f"{(latest['VOL_RATIO']-1)*100:+.2f}% vs MA20" if pd.notna(latest['VOL_RATIO']) else "-"),
        (c6,"🧠 AI Score",f"{current_score}/6", "Bullish" if current_score>=4 else ("Neutral" if current_score==3 else "Bearish")),
    ]
    for col,label,value,note in kpi:
        with col:
            render_html(f"""
            <div class="ai-card" style="height:100%;margin:0;">
              <div class="ai-card-title">{label}</div>
              <div class="ai-card-value">{value}</div>
              <div class="ai-card-sub">{note}</div>
            </div>
            """)

    # =========================================================
    # PRICE ACTION - ALWAYS VISIBLE
    # =========================================================
    render_html('''
    <div class="section-header"><span>📈</span><b>PRICE ACTION <span style="color:#697685;font-size:.68rem;font-weight:500">• BBRI + IHSG MARKET OVERLAY</span></b><div class="section-line"></div></div>
    ''')

    fig = go.Figure()

    # =========================================================
    # IHSG TRANSPARENT CANDLE - BACKGROUND
    # =========================================================
    # IHSG ditambahkan terlebih dahulu agar menjadi layer belakang.
    # Warnanya abu-abu transparan sehingga candle saham tetap dominan.
    if not ihsg_hist.empty:
        fig.add_trace(go.Candlestick(
            x=ihsg_hist.index,
            open=ihsg_hist["Open_NORM"],
            high=ihsg_hist["High_NORM"],
            low=ihsg_hist["Low_NORM"],
            close=ihsg_hist["Close_NORM"],
            name="IHSG • MARKET CONTEXT",
            increasing=dict(
                line=dict(color="rgba(148,163,184,0.42)", width=1),
                fillcolor="rgba(148,163,184,0.16)"
            ),
            decreasing=dict(
                line=dict(color="rgba(100,116,139,0.38)", width=1),
                fillcolor="rgba(100,116,139,0.11)"
            ),
            opacity=0.42,
            whiskerwidth=0.35,
            hovertemplate="IHSG normalized<br>%{x}<br>O: %{open:,.0f}<br>H: %{high:,.0f}<br>L: %{low:,.0f}<br>C: %{close:,.0f}<extra></extra>"
        ))

    # =========================================================
    # PRIMARY STOCK CANDLE - FOREGROUND
    # =========================================================
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist["Open"],
        high=hist["High"],
        low=hist["Low"],
        close=hist["Close"],
        name=ticker_input,
        increasing=dict(line=dict(color="#00e676", width=1), fillcolor="#00e676"),
        decreasing=dict(line=dict(color="#ff4560", width=1), fillcolor="#ff4560")
    ))

    for col,name,color,width in [
        ("MA20","SMA 20","#00e5ff",1.6),
        ("MA50","SMA 50","#ffb300",1.4),
        ("MA200","SMA 200","#ab47bc",1.4),
    ]:
        if col in hist.columns:
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist[col], name=name, mode="lines",
                line=dict(color=color,width=width)
            ))

    if show_support:
        fig.add_hrect(y0=support_low,y1=support_high,line_width=0,fillcolor="rgba(0,230,118,.06)")
        fig.add_hline(y=support,line_dash="dash",line_width=1.1,line_color="#00e676")
    if show_resistance:
        fig.add_hrect(y0=resistance_low,y1=resistance_high,line_width=0,fillcolor="rgba(255,69,96,.05)")
        fig.add_hline(y=resistance,line_dash="dash",line_width=1.1,line_color="#ff4560")
    if show_tp1: fig.add_hline(y=tp1,line_dash="dot",line_width=1,line_color="#00e676")
    if show_tp2: fig.add_hline(y=tp2,line_dash="dot",line_width=1,line_color="#29b6f6")
    if show_sl1: fig.add_hline(y=sl1,line_dash="dot",line_width=1,line_color="#ff5252")
    if show_sl2: fig.add_hline(y=sl2,line_dash="dot",line_width=1,line_color="#ff1744")
    if show_price: fig.add_hline(y=close_now,line_dash="solid",line_width=1,line_color="#78909c")

    def add_right_label(text,y,color):
        fig.add_annotation(
            x=1.075,y=y,xref="paper",yref="y",text=text,showarrow=False,xanchor="left",
            font=dict(size=10,color=color),bgcolor="rgba(8,11,16,.78)",borderwidth=0
        )

    if show_resistance: add_right_label(f"RESISTANCE  {resistance:,.0f}",resistance,"#ff4560")
    if show_price: add_right_label(f"HARGA  {close_now:,.0f}",close_now,"#f5f7fa")
    if show_support: add_right_label(f"SUPPORT  {support:,.0f}",support,"#00e676")
    if show_sl1: add_right_label(f"SL 1  {sl1:,.0f}",sl1,"#ff5252")
    if show_sl2: add_right_label(f"SL 2  {sl2:,.0f}",sl2,"#ff1744")
    if show_tp1: add_right_label(f"TP1  {tp1:,.0f}",tp1,"#00e676")
    if show_tp2: add_right_label(f"TP2  {tp2:,.0f}",tp2,"#29b6f6")

    fig.update_layout(
        template="plotly_dark",
        height=470,
        paper_bgcolor="#0c1117",
        plot_bgcolor="#0c1117",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=20,r=205,t=38,b=20),
        font=dict(color="#aeb8c2"),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,x=0),
        hoverlabel=dict(bgcolor="#111820",bordercolor="#344252",font=dict(color="#f5f7fa")),
        yaxis=dict(showgrid=True,gridcolor="#1b242e",zeroline=False),
    )
    fig.update_xaxes(showgrid=True,gridcolor="#1b242e")

    fig.add_annotation(
        x=0.985,y=1.015,xref="paper",yref="paper",
        text="▨ IHSG transparan • normalized market context",
        showarrow=False,xanchor="right",
        font=dict(size=8,color="#8b98a8")
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo":False,"responsive":True}
    )


    # =========================================================
    # TRADING PLAN
    # =========================================================
    render_html('''
    <div class="section-header"><span>🎯</span><b>TRADING PLAN</b><div class="section-line"></div></div>
    ''')
    plan_cols=st.columns(6)
    plan_items=[
        ("CURRENT PRICE",f"Rp {close_now:,.0f}","#29b6f6"),
        ("SUPPORT",f"Rp {support:,.0f}","#00e676"),
        ("RESISTANCE",f"Rp {resistance:,.0f}","#ff4560"),
        ("TP1",f"Rp {tp1:,.0f}","#00e676"),
        ("TP2",f"Rp {tp2:,.0f}","#29b6f6"),
        ("SL1 / SL2",f"Rp {sl1:,.0f} / Rp {sl2:,.0f}","#ff4560"),
    ]
    for col,(label,val,color) in zip(plan_cols,plan_items):
        with col:
            render_html(f'''<div class="trade-card"><div class="trade-label">{label}</div><div class="trade-value" style="color:{color}">{val}</div></div>''')
    st.caption(f"Resistance: {((resistance/close_now)-1)*100:.2f}% | Support: {((support/close_now)-1)*100:.2f}% | ATR: {atr_now:,.2f}")

    # =========================================================
    # OPTIONAL INDICATOR PANELS
    # POSISI: TEPAT DI BAWAH PRICE ACTION / TRADING PLAN
    # Candlestick price action tetap selalu tampil.
    # Checkbox sidebar hanya mengatur panel indikator tambahan.
    # =========================================================
    selected_indicators=[]
    if show_rsi: selected_indicators.append("RSI")
    if show_macd: selected_indicators.append("MACD")
    if show_cmo: selected_indicators.append("CMO")
    if show_bollinger: selected_indicators.append("Bollinger")
    if show_volume_chart: selected_indicators.append("Volume")

    if selected_indicators:
        render_html('''<div class="section-header"><span>📊</span><b>TECHNICAL INDICATORS · BELOW PRICE ACTION</b><div class="section-line"></div></div>''')


    def dark_chart(fig,height=300):
        fig.update_layout(template="plotly_dark",height=height,paper_bgcolor="#0c1117",plot_bgcolor="#0c1117",
                          margin=dict(l=35,r=25,t=35,b=25),hovermode="x unified",font=dict(color="#aeb8c2"),
                          legend=dict(orientation="h",yanchor="bottom",y=1.02,x=0),
                          hoverlabel=dict(bgcolor="#111820",bordercolor="#344252",font=dict(color="#f5f7fa")))
        fig.update_xaxes(showgrid=True,gridcolor="#1b242e")
        fig.update_yaxes(showgrid=True,gridcolor="#1b242e")
        return fig

    if show_rsi:
        render_html('''<div class="section-header"><span>📈</span><b>RSI 14</b><div class="section-line"></div></div>''')
        fr=go.Figure(); fr.add_trace(go.Scatter(x=hist.index,y=hist["RSI14"],name="RSI 14",line=dict(color="#a66cff",width=2)))
        fr.add_hline(y=70,line_dash="dash",line_color="#ff4560"); fr.add_hline(y=30,line_dash="dash",line_color="#00e676"); fr.add_hline(y=50,line_dash="dot",line_color="#697685")
        fr.update_yaxes(range=[0,100]); st.plotly_chart(dark_chart(fr,260),use_container_width=True,config={"displaylogo":False,"responsive":True})

    if show_macd:
        render_html('''<div class="section-header"><span>📉</span><b>MACD</b><div class="section-line"></div></div>''')
        fm=go.Figure(); fm.add_trace(go.Scatter(x=hist.index,y=hist["MACD"],name="MACD",line=dict(color="#00b8ff",width=2))); fm.add_trace(go.Scatter(x=hist.index,y=hist["MACD_SIGNAL"],name="Signal",line=dict(color="#ffb300",width=1.5))); fm.add_trace(go.Bar(x=hist.index,y=hist["MACD_HIST"],name="Histogram",marker_color="#455a64")); fm.add_hline(y=0,line_dash="dash",line_color="#697685")
        st.plotly_chart(dark_chart(fm,280),use_container_width=True,config={"displaylogo":False,"responsive":True})

    if show_cmo:
        render_html('''<div class="section-header"><span>〽️</span><b>CMO 14</b><div class="section-line"></div></div>''')
        fc=go.Figure(); fc.add_trace(go.Scatter(x=hist.index,y=hist["CMO14"],name="CMO 14",line=dict(color="#ff9100",width=2))); fc.add_hline(y=50,line_dash="dash",line_color="#00e676"); fc.add_hline(y=0,line_color="#697685"); fc.add_hline(y=-50,line_dash="dash",line_color="#ff4560"); fc.update_yaxes(range=[-100,100])
        st.plotly_chart(dark_chart(fc,260),use_container_width=True,config={"displaylogo":False,"responsive":True})

    if show_bollinger:
        render_html('''<div class="section-header"><span>📏</span><b>BOLLINGER BAND</b><div class="section-line"></div></div>''')
        fb=go.Figure(); fb.add_trace(go.Scatter(x=hist.index,y=hist["BB_UPPER"],name="BB Upper",line=dict(color="#697685",width=1))); fb.add_trace(go.Scatter(x=hist.index,y=hist["BB_LOWER"],name="BB Lower",line=dict(color="#697685",width=1))); fb.add_trace(go.Scatter(x=hist.index,y=hist["MA20"],name="MA20",line=dict(color="#00b8ff",width=1.5))); fb.add_trace(go.Scatter(x=hist.index,y=hist["Close"],name="Close",line=dict(color="#ffffff",width=1.2)))
        st.plotly_chart(dark_chart(fb,300),use_container_width=True,config={"displaylogo":False,"responsive":True})

    if show_volume_chart:
        render_html('''<div class="section-header"><span>📦</span><b>VOLUME</b><div class="section-line"></div></div>''')
        fv=go.Figure(); vol_colors=np.where(hist["Close"]>=hist["Open"],"#00c853","#ff4560"); fv.add_trace(go.Bar(x=hist.index,y=hist["Volume"],name="Volume",marker_color=vol_colors)); fv.add_trace(go.Scatter(x=hist.index,y=hist["VOL_MA20"],name="Volume MA20",line=dict(color="#00b8ff",width=1.5)))
        st.plotly_chart(dark_chart(fv,280),use_container_width=True,config={"displaylogo":False,"responsive":True})

    # =========================================================
    # CMF/CMO + STATISTICAL SIGNAL
    # =========================================================
    left,right=st.columns([1.25,1])
    with left:
        render_html('''<div class="section-header"><span>🌐</span><b>CMF VS CMO</b><div class="section-line"></div></div>''')
        comparison_indicator=pd.DataFrame({
            "Indikator":["CMF 20","CMO 14"],
            "Nilai":[f"{cmf_now:.3f}",f"{cmo_now:.2f}"],
            "Kondisi":["🟢 Akumulasi" if cmf_positive else "🔴 Distribusi","🟢 Momentum Positif" if cmo_positive else "🔴 Momentum Negatif"],
            "Fokus":["Money Flow / Volume","Momentum Harga"]
        })
        render_table(comparison_indicator)
    with right:
        render_html('''<div class="section-header"><span>◔</span><b>STATISTICAL SIGNAL</b><div class="section-line"></div></div>''')
        prob_txt="-" if pd.isna(probability_up) else f"{probability_up*100:.2f}%"
        exp_txt="-" if pd.isna(expectancy) else f"{expectancy*100:.2f}%"
        prob_cls="green" if pd.notna(probability_up) and probability_up>=.5 else "blue"
        render_html(f'''
        <div class="stat-signal-wrap">
          <div class="stat-card green"><div class="lbl">Current Score</div><div class="val">{current_score}/6</div><div class="sub">Composite indicator score</div></div>
          <div class="stat-card {prob_cls}"><div class="lbl">Probability Naik</div><div class="val">{prob_txt}</div><div class="sub">Historical probability</div></div>
          <div class="stat-card purple"><div class="lbl">Expectancy</div><div class="val">{exp_txt}</div><div class="sub">Average next-period return</div></div>
        </div>''')

    # =========================================================
    # INDICATOR INFLUENCE + AI DECISION ENGINE
    # =========================================================
    left,right=st.columns([1.1,1])
    with left:
        render_html('''<div class="section-header"><span>📐</span><b>INDICATOR INFLUENCE</b><div class="section-line"></div></div>''')
        pval=regression_pvalues(eval_data)
        if pval.empty:
            st.info("Data tidak cukup untuk regresi.")
        else:
            pval_show=pval.copy()
            pval_show["Coefficient"]=pval_show["Coefficient"].map(lambda x:f"{x:.6f}")
            pval_show["P-value"]=pval_show["P-value"].map(lambda x:f"{x:.6g}")
            render_table(pval_show)

    market_score=0
    reasons=[]
    if rsi_now>50: market_score+=1; reasons.append((True,"RSI > 50 → momentum positif"))
    else: reasons.append((False,"RSI ≤ 50 → momentum belum kuat"))
    if macd_bullish: market_score+=1; reasons.append((True,"MACD bullish"))
    else: reasons.append((False,"MACD bearish"))
    if cmf_positive: market_score+=1; reasons.append((True,"CMF positif → akumulasi relatif kuat"))
    else: reasons.append((False,"CMF negatif → distribusi relatif kuat"))
    if cmo_positive: market_score+=1; reasons.append((True,"CMO positif → momentum harga positif"))
    else: reasons.append((False,"CMO negatif → momentum harga negatif"))
    if close_now>latest["MA20"]: market_score+=1; reasons.append((True,"Harga di atas MA20"))
    else: reasons.append((False,"Harga di bawah MA20"))
    if volume_high: market_score+=1; reasons.append((True,"Volume di atas median 20 candle"))
    else: reasons.append((False,"Volume normal/rendah"))
    if market_score>=4:
        simple_signal="CENDERUNG BULLISH"; pill_class=""
    elif market_score==3:
        simple_signal="WAIT"; pill_class="wait"
    else:
        simple_signal="CENDERUNG BEARISH"; pill_class="bear"

    with right:
        render_html('''<div class="section-header"><span>🤖</span><b>AI DECISION ENGINE</b><div class="section-line"></div></div>''')
        reason_html=''.join(f'''<div class="reason-item {"ok" if ok else ""}"><span class="reason-dot"></span><span>{txt}</span></div>''' for ok,txt in reasons)
        render_html(f'''
        <div class="ai-engine">
          <div class="ai-engine-top">
            <div class="score-orb"><div class="score">{market_score}/6</div><div class="label">AI Score</div></div>
            <div class="signal-panel"><div class="small">Market Signal</div><div class="signal-pill {pill_class}"><span>●</span>{simple_signal}</div><div class="analytics-note">RSI · MACD · CMF · CMO · MA20 · Volume</div></div>
          </div>
          <div class="reason-list">{reason_html}</div>
        </div>''')

    # =========================================================
    # BACKTEST SCORE ENGINE + BEST THRESHOLD
    # =========================================================
    render_html('''<div class="section-header"><span>🧪</span><b>BACKTEST SCORE ENGINE</b><div class="section-line"></div></div>''')
    if len(eval_data)<30:
        st.warning("Histori terlalu sedikit untuk backtest.")
        comparison=pd.DataFrame()
        best_threshold=3
        best={"TOTAL_RETURN":0,"WIN_RATE":0,"JUMLAH_TRADE":0,"MAX_DRAWDOWN":0,"PROFIT_FACTOR":0}
    else:
        thresholds=list(range(1,7))
        backtests=[jalankan_backtest(eval_data,t) for t in thresholds]
        comparison=pd.DataFrame([{
            "Threshold":x["THRESHOLD"],"Jumlah Trade":x["JUMLAH_TRADE"],"Win Rate":x["WIN_RATE"],
            "Avg Return":x["AVG_RETURN"],"Total Return":x["TOTAL_RETURN"],"Max Drawdown":x["MAX_DRAWDOWN"],"Profit Factor":x["PROFIT_FACTOR"]
        } for x in backtests])
        display_comparison=comparison.copy()
        for col in ["Win Rate","Avg Return","Total Return","Max Drawdown"]:
            display_comparison[col]=display_comparison[col].map(lambda x:f"{x*100:.2f}%")
        display_comparison["Profit Factor"]=display_comparison["Profit Factor"].map(lambda x:"∞" if np.isinf(x) else f"{x:.2f}")
        bleft,bright=st.columns([2.3,1])
        with bleft:
            render_table(display_comparison)
        valid=comparison[comparison["Jumlah Trade"]>=10].copy()
        if not valid.empty:
            valid["PF_SORT"]=valid["Profit Factor"].replace(np.inf,999999)
            best_row=valid.sort_values(["PF_SORT","Total Return"],ascending=False).iloc[0]
            best_threshold=int(best_row["Threshold"])
        else:
            best_threshold=3
        best=jalankan_backtest(eval_data,best_threshold)
        with bright:
            pf_text = "∞" if np.isinf(best["PROFIT_FACTOR"]) else f"{best['PROFIT_FACTOR']:.2f}"
            tr_cls="positive" if best["TOTAL_RETURN"]>=0 else "negative"
            dd_cls="negative" if best["MAX_DRAWDOWN"]<0 else "positive"
            wr_cls="positive" if best["WIN_RATE"]>=.5 else "negative"
            render_html(f'''
            <div class="threshold-card">
              <div class="threshold-badge">🏆 Optimal Backtest</div>
              <div class="threshold-title">Best BUY Threshold</div>
              <div class="threshold-number">≥ {best_threshold}</div>
              <div class="threshold-sub">Kombinasi Profit Factor dan Total Return terbaik.</div>
              <div class="threshold-stats">
                <div class="threshold-stat"><div class="lbl">Total Return</div><div class="val {tr_cls}">{best["TOTAL_RETURN"]*100:.2f}%</div></div>
                <div class="threshold-stat"><div class="lbl">Win Rate</div><div class="val {wr_cls}">{best["WIN_RATE"]*100:.2f}%</div></div>
                <div class="threshold-stat"><div class="lbl">Jumlah Trade</div><div class="val">{best["JUMLAH_TRADE"]}</div></div>
                <div class="threshold-stat"><div class="lbl">Max Drawdown</div><div class="val {dd_cls}">{best["MAX_DRAWDOWN"]*100:.2f}%</div></div>
                <div class="threshold-stat"><div class="lbl">Profit Factor</div><div class="val">{pf_text}</div></div>
                <div class="threshold-stat"><div class="lbl">Current Score</div><div class="val">{current_score}/6</div></div>
              </div>
            </div>''')

    # =========================================================
    # FINAL AI DECISION + REASONS + TRADING SUMMARY
    # =========================================================
    score_ok=current_score>=best_threshold
    prob_ok=pd.notna(probability_up) and probability_up>=0.50
    expectancy_ok=pd.notna(expectancy) and expectancy>0
    macd_ok=macd_bullish
    cmf_ok=cmf_positive
    cmo_ok=cmo_positive
    final_points=sum([score_ok,prob_ok,expectancy_ok,macd_ok,cmf_ok,cmo_ok])
    if final_points>=5:
        final_signal="🟢 BUY"; signal_class="signal-buy"
    elif final_points>=3:
        final_signal="🟡 WAIT"; signal_class="signal-wait"
    else:
        final_signal="🔴 AVOID"; signal_class="signal-avoid"

    f1,f2,f3=st.columns([1.15,1.05,1.15])
    with f1:
        render_html('''<div class="section-header"><span>🎯</span><b>FINAL AI DECISION</b><div class="section-line"></div></div>''')
        conf=final_points/6*100
        prob_text="-" if pd.isna(probability_up) else f"{probability_up*100:.2f}%"
        render_html(f'''
        <div class="{signal_class}">
          <div style="display:flex;gap:8px;align-items:stretch;">
            <div style="flex:1"><div class="signal-title">FINAL SIGNAL</div><div class="signal-value">{final_signal}</div></div>
            <div style="flex:1"><div class="ai-card-title">AI CONFIDENCE SCORE</div><div class="ai-card-value">{conf:.0f}%</div><div style="height:8px;background:#202a34;border-radius:8px;margin-top:7px"><div style="width:{conf:.0f}%;height:8px;background:#00c853;border-radius:8px"></div></div></div>
            <div style="flex:1"><div class="ai-card-title">PROBABILITY</div><div class="ai-card-value">{prob_text}</div><div class="ai-card-sub">Historical probability</div></div>
          </div>
        </div>''')
    with f2:
        render_html('''<div class="section-header"><span>🔎</span><b>AI DECISION REASONS</b><div class="section-line"></div></div>''')
        items=[
            (score_ok,f"Score: {current_score} dari maksimal 6"),
            (prob_ok,"Probability naik: " + (f"{probability_up*100:.2f}%" if pd.notna(probability_up) else "-")),
            (expectancy_ok,"Expectancy: " + (f"{expectancy*100:.2f}%" if pd.notna(expectancy) else "-")),
            (macd_ok,"MACD: bullish" if macd_ok else "MACD: bearish"),
            (cmf_ok,"CMF: positif (akumulasi)" if cmf_ok else "CMF: negatif (distribusi)"),
            (cmo_ok,f"CMO: positif ({cmo_now:.2f})" if cmo_ok else f"CMO: negatif ({cmo_now:.2f})"),
            (volume_high,"Volume di atas median" if volume_high else "Volume normal/rendah")]
        render_html('<div class="ai-card">'+''.join(f"<div style='margin:5px 0'>{'🟢' if ok else '⚪'} {txt}</div>" for ok,txt in items)+'</div>')
# =========================================================
# DATA + NOTES

# =========================================================
# FOOTER
# =========================================================
st.markdown("<div class=\"footer\">Yahoo Finance • Technical Analysis + Statistical Discovery + Seasonality + News Terminal</div>", unsafe_allow_html=True)
