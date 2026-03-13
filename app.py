"""
AgriIntel — Rice Yield Failure Prediction Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
#  PAGE CONFIG  ← must be absolute first call
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="AgriIntel · Rice Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ═══════════════════════════════════════════
   INDIAN RICE FARMING PALETTE
   Inspired by: paddy fields (deep emerald),
   harvest season (golden amber), flooded
   fields (muddy water brown), laterite soil
   (terracotta red), husk/grain (warm cream)
═══════════════════════════════════════════ */
:root {
    /* Backgrounds — deep paddy field at dusk */
    --bg:        #0D1A0A;   /* near-black soil green */
    --surf:      #142410;   /* waterlogged field surface */
    --surf2:     #1A2E14;   /* slightly lighter panel */
    --border:    #2D4A20;   /* field bund / earthen border */

    /* Primary accent — young seedling green */
    --accent:    #7CFC00;   /* bright lime — fresh paddy shoots */

    /* Harvest gold — ripening grain */
    --harvest:   #FFD700;   /* golden yellow — ripe rice panicles */
    --amber:     #E8A020;   /* deeper amber — late harvest */

    /* Danger — crop stress / failure */
    --red:       #FF5722;   /* terracotta red — laterite soil / failure */

    /* Water — flooded paddy field */
    --water:     #4FC3F7;   /* shallow irrigation water / sky reflection */
    --blue:      #4FC3F7;

    /* Soil tones */
    --soil:      #8D6E63;   /* wet earth brown */
    --husk:      #F5DEB3;   /* wheat / rice husk cream */

    /* Text — cream-tinted for warm feel */
    --text:      #E8F5D0;   /* warm cream-green — readable on dark */
    --muted:     #A5C880;   /* mid-green — secondary labels */
    --dim:       #6B9E4A;   /* darker green — captions */
}

html, body, [class*="css"], .stApp {
    font-family: 'JetBrains Mono', monospace !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background-color: var(--bg) !important; }

#MainMenu, footer, header { visibility: hidden; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A2E14 0%, #0D1A0A 100%) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
    background-color: #2D4A20 !important;
    color: var(--harvest) !important;
}

.brand {
    font-family: 'Syne', sans-serif;
    font-size: 24px;
    font-weight: 800;
    letter-spacing: 3px;
    color: var(--text);
    padding: 4px 0 2px 0;
}
.brand-green { color: var(--harvest); }
.brand-sub {
    font-size: 9px;
    letter-spacing: 2.5px;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 20px;
}

@keyframes pulse {
    0%,100% { opacity:1; box-shadow: 0 0 0 0 rgba(255,215,0,0.5); }
    50%      { opacity:.5; box-shadow: 0 0 0 6px rgba(255,215,0,0); }
}
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}

.page-header {
    padding: 10px 0 18px 0;
    border-bottom: 2px solid var(--border);
    margin-bottom: 22px;
}
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 2px;
    color: var(--text);
    line-height: 1.1;
}
.page-title span { color: var(--harvest); }
.page-sub {
    font-size: 10px;
    letter-spacing: 2px;
    color: var(--muted);
    text-transform: uppercase;
    margin-top: 4px;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}
.kpi-card {
    background: linear-gradient(135deg, var(--surf2), var(--surf));
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-card.green::before  { background: linear-gradient(90deg,transparent,var(--accent),transparent); }
.kpi-card.gold::before   { background: linear-gradient(90deg,transparent,var(--harvest),transparent); }
.kpi-card.amber::before  { background: linear-gradient(90deg,transparent,var(--amber),transparent); }
.kpi-card.red::before    { background: linear-gradient(90deg,transparent,var(--red),transparent); }
.kpi-card.blue::before   { background: linear-gradient(90deg,transparent,var(--blue),transparent); }

.kpi-icon {
    font-size: 26px;
    position: absolute;
    top: 14px; right: 16px;
    opacity: 0.9;
    filter: drop-shadow(0 0 6px currentColor);
}
.kpi-label {
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--husk);
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    line-height: 1;
}
.kpi-card.green .kpi-value { color: var(--accent); }
.kpi-card.gold  .kpi-value { color: var(--harvest); }
.kpi-card.amber .kpi-value { color: var(--amber); }
.kpi-card.red   .kpi-value { color: var(--red); }
.kpi-card.blue  .kpi-value { color: var(--water); }
.kpi-sub {
    font-size: 9px;
    color: var(--muted);
    margin-top: 5px;
    letter-spacing: 1px;
}

.sec-head {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--harvest);
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--surf) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    background: transparent !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--harvest) !important;
    border-bottom: 2px solid var(--harvest) !important;
    background: var(--surf2) !important;
}

.stNumberInput input, .stSelectbox > div > div,
.stTextInput input, .stMultiSelect > div {
    background: var(--surf) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 3px !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2D4A00, #1A3000) !important;
    border: 1px solid rgba(255,215,0,0.4) !important;
    color: var(--harvest) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    padding: 12px !important;
    transition: all .25s ease !important;
}
.stButton > button:hover {
    border-color: var(--harvest) !important;
    box-shadow: 0 0 18px rgba(255,215,0,0.2) !important;
}

.pred-wrap {
    border-radius: 4px;
    padding: 22px 26px;
    margin-top: 14px;
}
.pred-wrap.high   { background:#1C0A06; border:2px solid rgba(255,87,34,0.6); }
.pred-wrap.medium { background:#1C1406; border:2px solid rgba(232,160,32,0.6); }
.pred-wrap.low    { background:#091A06; border:2px solid rgba(124,252,0,0.5); }
.pred-pct {
    font-family: 'Syne', sans-serif;
    font-size: 54px;
    font-weight: 800;
    line-height: 1;
}
.pred-wrap.high   .pred-pct { color: #FF5722; }
.pred-wrap.medium .pred-pct { color: var(--amber); }
.pred-wrap.low    .pred-pct { color: var(--accent); text-shadow: 0 0 20px rgba(124,252,0,0.4); }
.pred-badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 2px;
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-top: 8px;
}
.badge-high   { background:rgba(255,87,34,0.15); border:1px solid #FF5722; color:#FF5722; }
.badge-medium { background:rgba(255,215,0,0.15);  border:1px solid #FFD700; color:#FFD700; }
.badge-low    { background:#39ff6e22; border:1px solid var(--accent);color:var(--accent); }
.pred-bar-bg {
    height: 5px;
    background: var(--border);
    border-radius: 3px;
    margin-top: 14px;
}

.driver-card {
    background: var(--surf);
    border-left: 3px solid;
    border-radius: 0 3px 3px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    border-top: 1px solid var(--border);
    border-right: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
}
.driver-title {
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 4px;
    font-weight: 700;
}
.driver-desc { font-size: 11px; color: var(--husk); font-weight: 500; }

.derived-row {
    display: grid;
    grid-template-columns: repeat(3,1fr);
    gap: 8px;
    margin: 10px 0 14px 0;
}
.derived-box {
    background: var(--surf);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 10px;
    text-align: center;
}
.derived-label { font-size: 10px; letter-spacing: 1.5px; color: var(--husk); text-transform: uppercase; font-weight: 600; }
.derived-val   { font-family: 'Syne', sans-serif; font-size: 18px; font-weight: 800; color: var(--harvest); margin-top: 2px; }

.risk-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--surf);
    border-left: 3px solid;
    border-top: 1px solid var(--border);
    border-right: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    border-radius: 0 3px 3px 0;
    padding: 10px 14px;
    margin-bottom: 6px;
}
.risk-state   { font-size: 12px; color: var(--text); }
.risk-records { font-size: 10px; color: var(--husk); margin-top: 2px; }
.risk-pct {
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 800;
    text-align: right;
}
.risk-lbl { font-size: 9px; letter-spacing: 1.5px; text-align: right; }

.footer {
    border-top: 1px solid var(--border);
    padding: 12px 0 6px 0;
    display: flex;
    justify-content: space-between;
    font-size: 9px;
    color: var(--dim);
    letter-spacing: 1px;
    margin-top: 30px;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--harvest); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PLOTLY THEME
# ══════════════════════════════════════════════
# Indian rice farming colour palette for charts
# green=seedling, gold=harvest, blue=water, red=laterite/failure, brown=soil
PT = dict(
    paper_bgcolor="#142410",   # waterlogged field surface
    plot_bgcolor="#0D1A0A",    # deep paddy soil
    font=dict(family="JetBrains Mono, monospace", color="#E8F5D0", size=12),
    xaxis=dict(gridcolor="#2D4A20", linecolor="#2D4A20",
               tickfont=dict(color="#A5C880"), title_font=dict(color="#F5DEB3")),
    yaxis=dict(gridcolor="#2D4A20", linecolor="#2D4A20",
               tickfont=dict(color="#A5C880"), title_font=dict(color="#F5DEB3")),
    colorway=["#7CFC00","#FFD700","#4FC3F7","#FF5722","#F5DEB3"],
    legend=dict(
        bgcolor="#1A2E14",
        bordercolor="#2D4A20",
        borderwidth=1,
        font=dict(color="#F5DEB3", size=13),       # husk cream — max readable
        title_font=dict(color="#FFD700", size=13)  # harvest gold title
    ),
    title_font=dict(family="Syne, sans-serif", size=13, color="#FFD700"),
)

def theme(fig):
    fig.update_layout(**PT)
    return fig

# ══════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════
@st.cache_data
def load_data():
    try:
        df   = pd.read_excel("rice_dataset.xlsx")
        rice = df[df["Crop"] == "Rice"].copy()
    except Exception:
        np.random.seed(42)
        states  = ["Punjab","Haryana","Uttar Pradesh","Bihar","West Bengal",
                   "Odisha","Tamil Nadu","Karnataka","Assam","Jharkhand"]
        seasons = ["Kharif","Rabi","Whole Year"]
        rows = []
        for s in states:
            base = np.random.uniform(1.8, 3.2)
            for y in range(2005, 2024):
                for _ in range(np.random.randint(1, 3)):
                    rows.append(dict(
                        State=s, Season=np.random.choice(seasons),
                        Crop_Year=y,
                        Area=np.random.uniform(200, 8000),
                        Annual_Rainfall=np.random.uniform(600, 2600),
                        Fertilizer=np.random.uniform(80, 600),
                        Pesticide=np.random.uniform(5, 80),
                        Yield=base * np.random.uniform(0.55, 1.45)
                    ))
        rice = pd.DataFrame(rows)

    rice = rice.sort_values(["State", "Crop_Year"])
    rice["Rainfall_per_Area"]   = rice["Annual_Rainfall"] / rice["Area"]
    rice["Fertilizer_per_Area"] = rice["Fertilizer"]      / rice["Area"]
    rice["Pesticide_per_Area"]  = rice["Pesticide"]       / rice["Area"]
    rice["5yr_avg"] = rice.groupby("State")["Yield"] \
                          .transform(lambda x: x.rolling(5, min_periods=1).mean())
    rice["Yield_Failure"] = (rice["Yield"] < 0.7 * rice["5yr_avg"]).astype(int)
    return rice

@st.cache_resource
def load_model():
    try:
        return joblib.load("xgb_pipeline.pkl")
    except Exception:
        return None

rice_df = load_data()
model   = load_model()

# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════
def risk_color(p):
    return "#FF5722" if p > 0.6 else "#E8A020" if p > 0.35 else "#7CFC00"

def risk_label(p):
    return "HIGH" if p > 0.6 else "MEDIUM" if p > 0.35 else "LOW"

def risk_cls(p):
    return "high" if p > 0.6 else "medium" if p > 0.35 else "low"

def simulate(area, rainfall, fertilizer, pesticide, state):
    base_map = {"Bihar":0.62,"Odisha":0.55,"Assam":0.60,
                "Jharkhand":0.50,"West Bengal":0.48}
    b = base_map.get(state, 0.28)
    if rainfall < 800:           b += 0.14
    if rainfall > 2200:          b += 0.09
    if fertilizer / area < 0.05: b += 0.11
    if pesticide  / area < 0.01: b += 0.08
    return float(np.clip(b + np.random.normal(0, 0.04), 0.03, 0.96))

# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════
all_states  = sorted(rice_df["State"].unique())
all_seasons = sorted(rice_df["Season"].unique())

with st.sidebar:
    st.markdown("""
    <div class="brand">AGRI<span class="brand-green">INTEL</span></div>
    <div class="brand-sub">Rice Yield Failure System</div>
    """, unsafe_allow_html=True)

    mdl_color = "#7CFC00" if model else "#FFD700"
    mdl_text  = "MODEL ACTIVE" if model else "DEMO MODE"
    st.markdown(
        f'<div style="font-size:10px;letter-spacing:1.5px;color:{mdl_color};margin-bottom:20px;font-weight:600;">'
        f'<span class="status-dot" style="background:{mdl_color};"></span>{mdl_text}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="font-size:10px;letter-spacing:2px;color:#FFD700;font-weight:700;'
        'text-transform:uppercase;margin-bottom:8px;">🗂 Filter Dataset</div>',
        unsafe_allow_html=True
    )

    st.markdown('<p style="color:#FFD700;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:2px;">📍 States</p>', unsafe_allow_html=True)
    sel_states  = st.multiselect("States",  all_states,  default=all_states,  label_visibility="collapsed")
    st.markdown('<p style="color:#FFD700;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:2px;">🗓️ Seasons</p>', unsafe_allow_html=True)
    sel_seasons = st.multiselect("Seasons", all_seasons, default=all_seasons, label_visibility="collapsed")
    st.markdown('<p style="color:#FFD700;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:2px;">📅 Crop Year Range</p>', unsafe_allow_html=True)
    year_range  = st.slider(
        "Crop Year Range",
        int(rice_df["Crop_Year"].min()),
        int(rice_df["Crop_Year"].max()),
        (int(rice_df["Crop_Year"].min()), int(rice_df["Crop_Year"].max())),
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:9px;letter-spacing:1px;color:#A5C880;">'
        'AgriIntel v2.0 · XGBoost Pipeline · India</div>',
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════
#  FILTER
# ══════════════════════════════════════════════
filtered = rice_df[
    (rice_df["State"].isin(sel_states  if sel_states  else all_states)) &
    (rice_df["Season"].isin(sel_seasons if sel_seasons else all_seasons)) &
    (rice_df["Crop_Year"].between(*year_range))
].copy()

# ══════════════════════════════════════════════
#  PAGE HEADER
# ══════════════════════════════════════════════
st.markdown("""
<div class="page-header">
    <div class="page-title">🌾 AGRI<span>INTEL</span></div>
    <div class="page-sub">Rice Yield Failure Prediction Dashboard · India Agricultural Dataset</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  KPI ROW
# ══════════════════════════════════════════════
total    = len(filtered)
failures = int(filtered["Yield_Failure"].sum())
rate     = failures / total * 100 if total else 0
avg_yld  = filtered["Yield"].mean() if total else 0

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card green">
        <div class="kpi-icon" style="color:#7CFC00;">📊</div>
        <div class="kpi-label">Total Records</div>
        <div class="kpi-value">{total:,}</div>
        <div class="kpi-sub">Rice crop entries</div>
    </div>
    <div class="kpi-card amber">
        <div class="kpi-icon" style="color:#FFD700;">⚠️</div>
        <div class="kpi-label">Failure Cases</div>
        <div class="kpi-value">{failures:,}</div>
        <div class="kpi-sub">Yield below 70% avg</div>
    </div>
    <div class="kpi-card red">
        <div class="kpi-icon" style="color:#FF5722;">📉</div>
        <div class="kpi-label">Failure Rate</div>
        <div class="kpi-value">{rate:.1f}%</div>
        <div class="kpi-sub">Of filtered records</div>
    </div>
    <div class="kpi-card blue">
        <div class="kpi-icon" style="color:#4FC3F7;">🌾</div>
        <div class="kpi-label">Avg Yield</div>
        <div class="kpi-value">{avg_yld:.2f}</div>
        <div class="kpi-sub">Tonnes per hectare</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📈  Overview & Trends",
    "🗺️  State Risk Analysis",
    "🎯  Yield Failure Predictor",
])

# ──────────────────────────────────────────────
#  TAB 1 — OVERVIEW
# ──────────────────────────────────────────────
with tab1:
    st.markdown('<div class="sec-head">▸ Yield Trend & Failure Rate Over Time</div>',
                unsafe_allow_html=True)

    trend = filtered.groupby("Crop_Year").agg(
        Avg_Yield=("Yield", "mean"),
        Fail_Rate=("Yield_Failure", "mean"),
    ).reset_index()

    col_a, col_b = st.columns(2)

    with col_a:
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(go.Scatter(
            x=trend["Crop_Year"], y=trend["Avg_Yield"],
            name="Avg Yield (t/ha)", mode="lines+markers",
            line=dict(color="#7CFC00", width=2),
            marker=dict(size=5, color="#FFD700"),
            fill="tozeroy", fillcolor="rgba(124,252,0,0.07)"
        ), secondary_y=False)
        fig1.add_trace(go.Bar(
            x=trend["Crop_Year"], y=trend["Fail_Rate"],
            name="Failure Rate",
            marker_color="rgba(255,87,34,0.45)",
            marker_line_color="#FF5722", marker_line_width=1
        ), secondary_y=True)
        fig1.update_layout(title_text="Yield vs Failure Rate", **PT)
        fig1.update_yaxes(title_text="Yield (t/ha)", secondary_y=False,
                          titlefont=dict(color="#7CFC00"),
                          tickfont=dict(color="#a0c8a0"))
        fig1.update_yaxes(title_text="Failure Rate", secondary_y=True,
                          tickformat=".0%",
                          titlefont=dict(color="#FF5722"),
                          tickfont=dict(color="#a0c8a0"))
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        season_fail = filtered.groupby("Season")["Yield_Failure"].agg(
            Failures="sum", Total="count"
        ).reset_index()
        season_fail["Rate"] = season_fail["Failures"] / season_fail["Total"]
        season_fail = season_fail.sort_values("Rate", ascending=False)

        fig2 = go.Figure(go.Bar(
            x=season_fail["Season"],
            y=season_fail["Rate"],
            marker=dict(
                color=[risk_color(r) for r in season_fail["Rate"]],
                line=dict(color="#1c3020", width=1)
            ),
            text=[f"{r:.0%}" for r in season_fail["Rate"]],
            textposition="outside",
            textfont=dict(color="#F5DEB3", size=11)
        ))
        fig2.update_layout(title_text="Failure Rate by Season", **PT)
        fig2.update_yaxes(tickformat=".0%", gridcolor="#1c3020",
                          tickfont=dict(color="#a0c8a0"))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="sec-head" style="margin-top:8px;">▸ Yield Distribution</div>',
                unsafe_allow_html=True)

    col_c, col_d = st.columns(2)

    with col_c:
        fig3 = go.Figure()
        for val, color, name in [(0, "#7CFC00", "Normal ✓"), (1, "#FF5722", "Failure ✗")]:
            sub = filtered[filtered["Yield_Failure"] == val]["Yield"]
            fig3.add_trace(go.Histogram(
                x=sub, name=name,
                marker_color=color,
                marker_line_color=color, marker_line_width=1,
                nbinsx=35, opacity=0.4
            ))
        fig3.update_layout(
            title_text="Yield Distribution — Failure vs Normal",
            barmode="overlay", **PT
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        fig4 = go.Figure()
        fill_map = {0: "rgba(124,252,0,0.13)", 1: "rgba(255,87,34,0.13)"}
        for val, color, name in [(0, "#7CFC00", "Normal ✓"), (1, "#FF5722", "Failure ✗")]:
            sub = filtered[filtered["Yield_Failure"] == val]["Yield"]
            fig4.add_trace(go.Box(
                y=sub, name=name,
                marker_color=color, line_color=color,
                fillcolor=fill_map[val], boxmean="sd"
            ))
        fig4.update_layout(title_text="Yield Spread by Class", **PT)
        st.plotly_chart(fig4, use_container_width=True)

    with st.expander("📋  View Filtered Dataset"):
        st.dataframe(
            filtered[["State","Season","Crop_Year","Area","Annual_Rainfall",
                       "Fertilizer","Pesticide","Yield","Yield_Failure"]].head(300),
            use_container_width=True, hide_index=True
        )

# ──────────────────────────────────────────────
#  TAB 2 — STATE RISK
# ──────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sec-head">▸ State-Level Risk Index</div>',
                unsafe_allow_html=True)

    state_stats = filtered.groupby("State").agg(
        Records=("Yield", "count"),
        Failures=("Yield_Failure", "sum"),
        Failure_Rate=("Yield_Failure", "mean"),
        Avg_Yield=("Yield", "mean"),
        Avg_Rainfall=("Annual_Rainfall", "mean"),
    ).reset_index().sort_values("Failure_Rate", ascending=False)

    # ── Load India States shapefile from GADM files (same folder as app.py) ─
    @st.cache_data
    def load_india_geojson():
        """
        Loads gadm41_IND_1.shp (and companion files) from the same directory
        as app.py. Returns a GeoJSON-compatible dict ready for px.choropleth.
        Requires: pip install geopandas
        Files needed in same folder: gadm41_IND_1.shp / .dbf / .shx / .prj / .cpg
        """
        import os, json
        import geopandas as gpd

        shp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gadm41_IND_1.shp")
        gdf = gpd.read_file(shp_path)

        # GADM level-1 stores state names in column NAME_1
        # Keep only relevant columns and rename for consistency
        gdf = gdf[["NAME_1", "geometry"]].rename(columns={"NAME_1": "ST_NM"})

        # Simplify geometry for faster rendering (tolerance in degrees ~1km)
        gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.01, preserve_topology=True)

        # Convert to GeoJSON dict
        return json.loads(gdf.to_json())

    # ── Name mapping — GADM names → dataset names ─────────────────────────
    # GADM uses slightly different spellings for some states; map them here
    GADM_NAME_MAP = {
        "Odisha":         "Odisha",       # GADM may say "Orissa" in older versions
        "Orissa":         "Odisha",
        "Uttarakhand":    "Uttarakhand",
        "Uttaranchal":    "Uttarakhand",
    }

    try:
        INDIA_GEOJSON = load_india_geojson()

        # Fix any GADM name mismatches so locations match the dataset
        for feat in INDIA_GEOJSON["features"]:
            name = feat["properties"]["ST_NM"]
            feat["properties"]["ST_NM"] = GADM_NAME_MAP.get(name, name)

        geojson_loaded = True
    except Exception as geo_err:
        geojson_loaded = False
        geo_err_msg = str(geo_err)

    col_e, col_f = st.columns([1.6, 1])

    with col_e:
        # Build map dataframe
        map_df = state_stats[["State","Failure_Rate","Avg_Yield","Records","Avg_Rainfall"]].copy()
        map_df["Failure_Pct"] = (map_df["Failure_Rate"] * 100).round(1)
        map_df["Risk_Label"]  = map_df["Failure_Rate"].apply(risk_label)

        if not geojson_loaded:
            st.warning(
                f"⚠️ Could not load GADM shapefile — make sure **gadm41_IND_1.shp** "
                f"and its companion files (.dbf .shx .prj .cpg) are in the same "
                f"folder as app.py, and that **geopandas** is installed "
                f"(`pip install geopandas`).\n\nError: {geo_err_msg}"
            )
        else:
            # ── Dynamic colour scale — stretch to actual data range ────────
            # This ensures colour contrast is visible regardless of how small
            # the failure rates are (e.g. all states < 5% still show spread)
            r_min = float(map_df["Failure_Rate"].min())
            r_max = float(map_df["Failure_Rate"].max())
            r_mid = r_min + (r_max - r_min) * 0.5   # midpoint = harvest gold
            r_hi  = r_min + (r_max - r_min) * 0.75  # upper quartile = red

            # Normalise breakpoints to [0,1] for color_continuous_scale
            def norm(v): return (v - r_min) / (r_max - r_min) if r_max > r_min else 0.5

            dynamic_scale = [
                [0.0,        "#7CFC00"],  # lowest  state  → seedling green
                [norm(r_mid),"#FFD700"],  # midpoint        → harvest gold
                [norm(r_hi), "#FF5722"],  # upper quartile  → laterite red
                [1.0,        "#CC2200"],  # highest state   → deep red
            ]

            # Colourbar tick labels showing real % values
            n_ticks = 5
            tick_vals = [r_min + i*(r_max - r_min)/(n_ticks-1) for i in range(n_ticks)]
            tick_text = [f"{v*100:.2f}%" for v in tick_vals]

            fig5 = px.choropleth(
                map_df,
                geojson=INDIA_GEOJSON,
                locations="State",
                featureidkey="properties.ST_NM",
                color="Failure_Rate",
                color_continuous_scale=dynamic_scale,
                range_color=[r_min, r_max],   # ← key fix: use actual data range
                hover_name="State",
                hover_data={
                    "Failure_Rate": False,
                    "Failure_Pct":  True,
                    "Avg_Yield":    ":.2f",
                    "Records":      True,
                    "Avg_Rainfall": ":.0f",
                },
                labels={
                    "Failure_Pct":  "Failure Rate (%)",
                    "Avg_Yield":    "Avg Yield (t/ha)",
                    "Records":      "Records",
                    "Avg_Rainfall": "Avg Rainfall (mm)",
                },
                title=f"Rice Yield Failure Risk by State  "
                      f"(range: {r_min*100:.2f}% – {r_max*100:.2f}%)",
            )
            fig5.update_geos(
                fitbounds="locations",
                visible=False,
                bgcolor="#0D1A0A",
                showcoastlines=True,  coastlinecolor="#2D4A20",
                showland=True,        landcolor="#142410",
                showocean=True,       oceancolor="#0D1A0A",
                showlakes=False,
                showrivers=False,
                showframe=False,
            )
            fig5.update_layout(
                **PT,
                height=500,
                margin=dict(t=40, b=10, l=0, r=0),
                coloraxis_colorbar=dict(
                    title=dict(
                        text="Failure Rate",
                        font=dict(color="#FFD700", size=11)
                    ),
                    tickvals=tick_vals,
                    ticktext=tick_text,
                    tickfont=dict(color="#F5DEB3", size=10),
                    bgcolor="#142410",
                    bordercolor="#2D4A20",
                    borderwidth=1,
                    len=0.75,
                    thickness=14,
                ),
            )
            fig5.update_traces(
                marker_line_color="#2D4A20",
                marker_line_width=1.2,
            )
            st.plotly_chart(fig5, use_container_width=True)

    with col_f:
        st.markdown(
            '<div style="font-size:11px;letter-spacing:2px;color:#FFD700;font-weight:800;'
            'text-transform:uppercase;margin-bottom:10px;border-bottom:1px solid #2D4A20;'
            'padding-bottom:6px;">📍 State Risk Index</div>',
            unsafe_allow_html=True
        )
        for _, row in state_stats.iterrows():
            rc = risk_color(row["Failure_Rate"])
            rl = risk_label(row["Failure_Rate"])
            st.markdown(f"""
            <div class="risk-row" style="border-left-color:{rc};">
                <div>
                    <div class="risk-state">{row['State']}</div>
                    <div class="risk-records">{int(row['Records'])} records</div>
                </div>
                <div>
                    <div class="risk-pct" style="color:{rc};">{row['Failure_Rate']:.0%}</div>
                    <div class="risk-lbl" style="color:{rc};">{rl}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="sec-head" style="margin-top:16px;">▸ Rainfall vs Yield</div>',
                unsafe_allow_html=True)

    fig6 = px.scatter(
        state_stats, x="Avg_Rainfall", y="Avg_Yield",
        size="Records", color="Failure_Rate",
        color_continuous_scale=["#7CFC00", "#FFD700", "#FF5722"],
        hover_name="State", size_max=45,
        labels={"Avg_Rainfall": "Avg Rainfall (mm)",
                "Avg_Yield":    "Avg Yield (t/ha)",
                "Failure_Rate": "Failure Rate"},
        title="Rainfall vs Yield  (bubble size = record count)"
    )
    fig6.update_traces(marker=dict(line=dict(color="#1c3020", width=1)))
    theme(fig6)
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("""
    <div style="display:flex;gap:24px;margin-top:6px;">
        <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:14px;height:14px;background:#7CFC00;border-radius:3px;
                        box-shadow:0 0 6px rgba(124,252,0,0.6);"></div>
            <span style="font-size:11px;color:#E8F5D0;letter-spacing:1px;font-weight:600;">
                🌱 LOW  &lt; 35%</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:14px;height:14px;background:#FFD700;border-radius:3px;
                        box-shadow:0 0 6px rgba(255,215,0,0.6);"></div>
            <span style="font-size:11px;color:#E8F5D0;letter-spacing:1px;font-weight:600;">
                🌾 MEDIUM  35–60%</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:14px;height:14px;background:#FF5722;border-radius:3px;
                        box-shadow:0 0 6px rgba(255,87,34,0.6);"></div>
            <span style="font-size:11px;color:#E8F5D0;letter-spacing:1px;font-weight:600;">
                🔥 HIGH  &gt; 60%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  TAB 3 — PREDICTION
# ──────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sec-head">▸ Yield Failure Risk Predictor</div>',
                unsafe_allow_html=True)

    p_left, p_right = st.columns([1.1, 1])

    with p_left:
        st.markdown("""
        <div style="background:var(--surf2);border:1px solid var(--border);
                    border-radius:4px;padding:14px 16px;margin-bottom:14px;">
            <div style="font-size:9px;letter-spacing:2px;color:#FFD700;
                        text-transform:uppercase;margin-bottom:4px;font-weight:700;font-size:11px;">🌾 About This Tool</div>
            <div style="font-size:11px;color:#E8F5D0;line-height:1.8;">
            Enter field-level conditions. The XGBoost model evaluates rainfall
            adequacy, fertilizer efficiency, pesticide coverage, and historical
            state trends to output a failure probability.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div style="font-size:11px;letter-spacing:2px;color:#FFD700;font-weight:800;'
            'text-transform:uppercase;margin-bottom:12px;'
            'border-bottom:1px solid #2D4A20;padding-bottom:8px;">📋 Field Parameters</div>',
            unsafe_allow_html=True
        )

        # Custom CSS to make ALL input labels bright throughout this tab
        st.markdown("""
        <style>
        /* Streamlit widget labels — make all bright */
        div[data-testid="stNumberInput"] label,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSlider"] label,
        div[data-testid="stMultiSelect"] label,
        div[data-testid="stTextInput"] label {
            color: #FFD700 !important;
            font-size: 11px !important;
            font-weight: 700 !important;
            letter-spacing: 1.5px !important;
            text-transform: uppercase !important;
        }
        /* Input box text bright */
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input {
            color: #E8F5D0 !important;
            font-size: 13px !important;
            font-weight: 600 !important;
        }
        /* Selectbox selected value bright */
        div[data-testid="stSelectbox"] div[data-baseweb="select"] span {
            color: #E8F5D0 !important;
            font-size: 13px !important;
            font-weight: 600 !important;
        }
        /* Selectbox dropdown options */
        li[role="option"] span {
            color: #E8F5D0 !important;
            font-size: 12px !important;
        }
        /* Number input +/- step buttons */
        div[data-testid="stNumberInput"] button {
            color: #FFD700 !important;
            border-color: #2D4A20 !important;
        }
        /* Slider label and values */
        div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {
            color: #FFD700 !important;
            font-weight: 700 !important;
        }
        /* Slider thumb label */
        div[data-testid="stSlider"] [data-testid="stThumbValue"] {
            color: #FFD700 !important;
        }
        /* Sidebar labels too */
        section[data-testid="stSidebar"] label {
            color: #FFD700 !important;
            font-weight: 700 !important;
            font-size: 11px !important;
            letter-spacing: 1px !important;
        }
        /* Sidebar multiselect placeholder */
        section[data-testid="stSidebar"] [data-baseweb="select"] span {
            color: #A5C880 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown('<p style="color:#FFD700;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:2px;">🌿 Area (ha)</p>', unsafe_allow_html=True)
            area       = st.number_input("Area (ha)",            1.0,   50000.0, 500.0,  step=50.0,  label_visibility="collapsed")
            st.markdown('<p style="color:#FFD700;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:2px;">🌧️ Annual Rainfall (mm)</p>', unsafe_allow_html=True)
            rainfall   = st.number_input("Annual Rainfall (mm)", 100.0, 4000.0,  1200.0, step=50.0,  label_visibility="collapsed")
        with r2:
            st.markdown('<p style="color:#FFD700;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:2px;">🧪 Fertilizer (kg)</p>', unsafe_allow_html=True)
            fertilizer = st.number_input("Fertilizer (kg)",      1.0,   5000.0,  150.0,  step=10.0,  label_visibility="collapsed")
            st.markdown('<p style="color:#FFD700;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:2px;">🐛 Pesticide (kg)</p>', unsafe_allow_html=True)
            pesticide  = st.number_input("Pesticide (kg)",       0.1,   500.0,   20.0,   step=1.0,   label_visibility="collapsed")

        r3, r4 = st.columns(2)
        with r3:
            st.markdown('<p style="color:#FFD700;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:2px;">📍 State</p>', unsafe_allow_html=True)
            state = st.selectbox("State",     all_states, label_visibility="collapsed")
            st.markdown('<p style="color:#FFD700;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:2px;">📅 Crop Year</p>', unsafe_allow_html=True)
            year  = st.number_input("Crop Year", 2000, 2030, 2023, step=1, label_visibility="collapsed")
        with r4:
            st.markdown('<p style="color:#FFD700;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:2px;">🗓️ Season</p>', unsafe_allow_html=True)
            season = st.selectbox("Season", all_seasons, label_visibility="collapsed")

        rpa = rainfall   / area if area else 0
        fpa = fertilizer / area if area else 0
        ppa = pesticide  / area if area else 0

        st.markdown(
            '<div style="font-size:10px;letter-spacing:2px;color:#F5DEB3;font-weight:700;'
            'text-transform:uppercase;margin:10px 0 6px 0;">Derived Density Features</div>',
            unsafe_allow_html=True
        )
        st.markdown(f"""
        <div class="derived-row">
            <div class="derived-box">
                <div class="derived-label">Rain / ha</div>
                <div class="derived-val">{rpa:.2f}</div>
            </div>
            <div class="derived-box">
                <div class="derived-label">Fert / ha</div>
                <div class="derived-val">{fpa:.2f}</div>
            </div>
            <div class="derived-box">
                <div class="derived-label">Pest / ha</div>
                <div class="derived-val">{ppa:.4f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        run = st.button("▸  RUN PREDICTION")

    with p_right:
        st.markdown(
            '<div style="font-size:10px;letter-spacing:2px;color:#F5DEB3;font-weight:700;'
            'text-transform:uppercase;margin-bottom:8px;">Prediction Result</div>',
            unsafe_allow_html=True
        )

        if run:
            import time
            with st.spinner("Analysing..."):
                time.sleep(0.4)

            if model:
                inp = pd.DataFrame([{
                    "Area": area, "Annual_Rainfall": rainfall,
                    "Fertilizer": fertilizer, "Pesticide": pesticide,
                    "Crop_Year": year,
                    "Rainfall_per_Area":   rpa,
                    "Fertilizer_per_Area": fpa,
                    "Pesticide_per_Area":  ppa,
                    "State": state, "Season": season
                }])
                prob = float(model.predict_proba(inp)[0][1])
            else:
                prob = simulate(area, rainfall, fertilizer, pesticide, state)

            rc  = risk_color(prob)
            rl  = risk_label(prob)
            cls = risk_cls(prob)
            sym = "⚠" if rl == "HIGH" else "△" if rl == "MEDIUM" else "✓"

            st.markdown(f"""
            <div class="pred-wrap {cls}">
                <div style="font-size:9px;letter-spacing:2px;color:#F5DEB3;margin-bottom:6px;font-weight:700;font-size:11px;">
                    YIELD FAILURE PROBABILITY
                </div>
                <div class="pred-pct">{prob*100:.1f}%</div>
                <div style="margin-top:8px;">
                    <span class="pred-badge badge-{cls}">{sym} {rl} RISK</span>
                </div>
                <div class="pred-bar-bg">
                    <div style="height:100%;border-radius:3px;
                                width:{prob*100:.1f}%;
                                background:linear-gradient(90deg,#7CFC00,{rc});"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # driver cards
            st.markdown(
                '<div style="font-size:10px;letter-spacing:2px;color:#F5DEB3;font-weight:700;'
                'text-transform:uppercase;margin:14px 0 8px 0;">Key Risk Drivers</div>',
                unsafe_allow_html=True
            )

            drivers = []
            if rainfall < 800:
                drivers.append(("🌧️ Low Rainfall",
                                 f"{rainfall:.0f} mm — below optimal 800–1500 mm range.", "#FF5722"))
            elif rainfall > 2200:
                drivers.append(("🌧️ Excess Rainfall",
                                 f"{rainfall:.0f} mm — elevated flooding risk.", "#E8A020"))
            else:
                drivers.append(("🌧️ Rainfall Adequate",
                                 f"{rainfall:.0f} mm — within healthy range.", "#7CFC00"))

            if fpa < 0.05:
                drivers.append(("🧪 Low Fertilizer Density",
                                 f"{fpa:.4f} kg/ha — consider increasing inputs.", "#FF5722"))
            else:
                drivers.append(("🧪 Fertilizer Sufficient",
                                 f"{fpa:.4f} kg/ha — adequate nutrient density.", "#7CFC00"))

            if ppa < 0.01:
                drivers.append(("🛡️ Low Pesticide Coverage",
                                 f"{ppa:.5f} kg/ha — pest exposure elevated.", "#E8A020"))
            else:
                drivers.append(("🛡️ Pesticide Coverage OK",
                                 f"{ppa:.5f} kg/ha", "#7CFC00"))

            s_rate = rice_df[rice_df["State"] == state]["Yield_Failure"].mean()
            drivers.append((f"📍 {state} Historical Risk",
                             f"{s_rate:.0%} historical failure rate in dataset.",
                             risk_color(s_rate)))

            for title, desc, col in drivers:
                st.markdown(f"""
                <div class="driver-card" style="border-left-color:{col};">
                    <div class="driver-title" style="color:{col};">{title}</div>
                    <div class="driver-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:var(--surf);border:1px dashed var(--border);
                        border-radius:4px;padding:52px 24px;text-align:center;">
                <div style="font-size:40px;margin-bottom:12px;">🌾</div>
                <div style="font-family:'Syne',sans-serif;font-size:13px;font-weight:700;
                             letter-spacing:2px;text-transform:uppercase;color:#FFD700;font-weight:700;">
                    Configure inputs &amp; run prediction
                </div>
                <div style="font-size:10px;margin-top:8px;letter-spacing:1px;color:#E8F5D0;">
                    Fill in field parameters on the left, then click RUN PREDICTION
                </div>
            </div>
            """, unsafe_allow_html=True)

        # historical sparkline
        hist_state = state if run else all_states[0]
        st.markdown(
            f'<div style="font-size:10px;letter-spacing:2px;color:#F5DEB3;font-weight:700;'
            f'text-transform:uppercase;margin:16px 0 8px 0;">'
            f'Historical Yield — {hist_state}</div>',
            unsafe_allow_html=True
        )

        hist = rice_df[rice_df["State"] == hist_state].groupby("Crop_Year").agg(
            Avg_Yield=("Yield", "mean"),
            Failures=("Yield_Failure", "sum")
        ).reset_index()

        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(
            x=hist["Crop_Year"], y=hist["Avg_Yield"],
            fill="tozeroy", fillcolor="rgba(124,252,0,0.08)",
            line=dict(color="#7CFC00", width=1.5),
            name="Avg Yield"
        ))
        fail_yrs = hist[hist["Failures"] > 0]
        fig7.add_trace(go.Scatter(
            x=fail_yrs["Crop_Year"], y=fail_yrs["Avg_Yield"],
            mode="markers", name="Failure Year",
            marker=dict(color="#FF5722", size=9, symbol="x-thin",
                        line=dict(color="#FF5722", width=2.5))
        ))
        if run:
            fig7.add_vline(x=year, line_dash="dot", line_color="rgba(255,184,48,0.60)",
                           annotation_text=str(year),
                           annotation_font_color="#FFD700")
        fig7.update_layout(height=190, showlegend=True, margin=dict(t=8, b=8, l=8, r=8), **PT)
        st.plotly_chart(fig7, use_container_width=True)

# ══════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════
st.markdown("""
<div class="footer">
    <span>AGRIINTEL v2.0 · XGBoost Pipeline · India Agricultural Dataset</span>
    <span>Built with Streamlit · Plotly · Python</span>
</div>
""", unsafe_allow_html=True)
