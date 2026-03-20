import streamlit as st
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import cv2

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Ahmedabad Greenery Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  NATURE-THEMED CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: linear-gradient(160deg, #0f2d1a 0%, #1a3d1f 35%, #1e4a28 60%, #162b14 100%);
    min-height: 100vh;
    color: #e8f5e0;
}

/* Subtle leaf texture overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(ellipse at 20% 10%, rgba(56,142,60,0.12) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(27,94,32,0.18) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(0,0,0,0.10) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d2410 0%, #143318 60%, #0a1e0d 100%) !important;
    border-right: 1px solid rgba(76,175,80,0.20) !important;
}
[data-testid="stSidebar"] * { color: #c8e6c9 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #a5d6a7 !important;
    font-family: 'Lora', serif !important;
}
[data-testid="stSidebar"] .stSlider label { color: #81c784 !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #81c784 !important; }

/* Slider track */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div {
    background: rgba(76,175,80,0.25) !important;
}

/* ── Headings ── */
h1, h2, h3, h4 {
    font-family: 'Lora', serif !important;
    color: #a5d6a7 !important;
}

/* ── Glass card ── */
.nature-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 20px;
    border: 1px solid rgba(76,175,80,0.22);
    padding: 24px 28px;
    margin-bottom: 18px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.30), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: transform 0.25s ease, border-color 0.25s ease;
}
.nature-card:hover {
    transform: translateY(-2px);
    border-color: rgba(76,175,80,0.40);
}

/* ── Stat box ── */
.stat-box {
    background: rgba(56,142,60,0.12);
    border: 1px solid rgba(76,175,80,0.25);
    border-radius: 16px;
    padding: 18px 14px;
    text-align: center;
    transition: transform 0.22s ease, background 0.22s ease;
    cursor: default;
}
.stat-box:hover {
    transform: translateY(-3px) scale(1.02);
    background: rgba(56,142,60,0.20);
}
.stat-value { font-size: 1.9rem; font-weight: 700; color: #a5d6a7; }
.stat-label { font-size: 0.75rem; color: rgba(200,230,200,0.60); text-transform: uppercase; letter-spacing: 0.8px; margin-top: 3px; font-weight: 600; }

/* ── Progress legend row ── */
.legend-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 10px; border-radius: 10px;
    transition: background 0.18s ease, padding-left 0.18s ease;
    cursor: default;
    border-bottom: 1px solid rgba(76,175,80,0.10);
}
.legend-row:last-child { border-bottom: none; }
.legend-row:hover { background: rgba(76,175,80,0.10); padding-left: 16px; }
.legend-dot { width: 14px; height: 14px; border-radius: 4px; flex-shrink: 0; }

/* ── Primary button ── */
.stButton > button {
    width: 100%;
    border-radius: 50px;
    height: 3.0em;
    background: linear-gradient(135deg, #2e7d32 0%, #43a047 60%, #66bb6a 100%);
    color: #f1f8e9;
    border: none;
    font-weight: 700;
    font-size: 0.97rem;
    letter-spacing: 0.8px;
    box-shadow: 0 4px 18px rgba(56,142,60,0.40);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 28px rgba(102,187,106,0.55);
}
.stButton > button:active { transform: scale(0.97); }

/* ── Slider ── */
.stSlider [data-baseweb="slider"] { padding: 0 4px; }

/* ── Spinner ── */
.stSpinner > div { border-color: #66bb6a transparent transparent transparent !important; }

/* ── Alerts ── */
.stSuccess { background: rgba(56,142,60,0.20) !important; border-color: #4caf50 !important; }
.stSuccess p { color: #c8e6c9 !important; }
.stInfo p { color: #b2dfdb !important; }

/* ── Metric ── */
div[data-testid="stMetricValue"] {
    color: #a5d6a7 !important;
    font-family: 'Lora', serif;
}
div[data-testid="stMetricLabel"] { color: rgba(200,230,200,0.65) !important; }

/* ── Divider ── */
hr { border-color: rgba(76,175,80,0.18) !important; }

/* ── Footer ── */
.footer-text {
    color: rgba(200,230,200,0.35);
    font-size: 0.75rem;
    text-align: center;
    margin-top: 32px;
    padding-bottom: 12px;
}

/* ── Year badge ── */
.year-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    border: 1px solid rgba(102,187,106,0.45);
    border-radius: 50px;
    padding: 4px 18px;
    font-family: 'Lora', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #a5d6a7;
    letter-spacing: 0.5px;
}

/* ── Floating leaves animation ── */
@keyframes floatLeaf {
    0%,100% { transform: translateY(0) rotate(0deg); opacity: 0.6; }
    33%      { transform: translateY(-12px) rotate(8deg); opacity: 0.9; }
    66%      { transform: translateY(-6px) rotate(-5deg); opacity: 0.75; }
}
.leaf1 { animation: floatLeaf 4.0s ease-in-out infinite; display:inline-block; }
.leaf2 { animation: floatLeaf 3.5s ease-in-out 0.8s infinite; display:inline-block; }
.leaf3 { animation: floatLeaf 4.5s ease-in-out 1.5s infinite; display:inline-block; }

/* ── Fade-in ── */
@keyframes fadeUp {
    from { opacity:0; transform:translateY(14px); }
    to   { opacity:1; transform:translateY(0); }
}
.fade-in   { animation: fadeUp 0.60s ease-out both; }
.fade-in-2 { animation: fadeUp 0.60s ease-out 0.12s both; }
.fade-in-3 { animation: fadeUp 0.60s ease-out 0.24s both; }

/* ── Matplotlib figure background fix ── */
.stpyplot { border-radius: 16px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MATPLOTLIB THEME  (dark forest)
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1f10",
    "axes.facecolor":    "#0d1f10",
    "text.color":        "#c8e6c9",
    "axes.titlecolor":   "#a5d6a7",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlepad":     12,
    "font.family":       "sans-serif",
    "savefig.facecolor": "#0d1f10",
    "savefig.edgecolor": "none",
})


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
BASE_YEAR    = 2024
MAP_SIZE     = 722      # final display resolution
MODEL_SIZE   = 64       # model input/output size
SEQ_LEN      = 5

LAND_COLORS  = ['#1565C0', '#607D8B', '#AED581', '#2E7D32']
LAND_LABELS  = ['Water / Wetland', 'Built-up / Urban', 'Sparse Vegetation', 'Dense Greenery']

NDVI_CMAP    = 'RdYlGn'   # red=bare  yellow=sparse  green=dense


# ─────────────────────────────────────────────
#  LOAD ASSETS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    session = ort.InferenceSession('model_flexible.onnx')
    seed    = np.load('seed_data.npy')
    return session, seed

session, seed_data = load_assets()


# ─────────────────────────────────────────────
#  CLASSIFY NDVI → 4 CLASSES
# ─────────────────────────────────────────────
def classify_ndvi(ndvi: np.ndarray) -> np.ndarray:
    out = np.zeros_like(ndvi, dtype=np.uint8)
    out[ndvi < 0]                              = 0   # water
    out[(ndvi >= 0)   & (ndvi < 0.2)]          = 1   # urban
    out[(ndvi >= 0.2) & (ndvi < 0.4)]          = 2   # sparse
    out[ndvi >= 0.4]                           = 3   # dense
    return out


# ─────────────────────────────────────────────
#  COVERAGE STATS
# ─────────────────────────────────────────────
def coverage_pct(classified: np.ndarray):
    total = classified.size
    return {lbl: float((classified == i).sum() / total * 100)
            for i, lbl in enumerate(LAND_LABELS)}


# ─────────────────────────────────────────────
#  PREDICTION ENGINE
# ─────────────────────────────────────────────
def run_prediction(years_ahead: int):
    # Resize seed frames to model size
    frames = []
    for i in range(SEQ_LEN):
        f = cv2.resize(seed_data[i, :, :, 0], (MODEL_SIZE, MODEL_SIZE),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
        frames.append(f)

    current_input = np.array(frames)[np.newaxis, :, :, :, np.newaxis]  # (1,5,64,64,1)
    input_name    = session.get_inputs()[0].name

    pred = None
    for _ in range(years_ahead):
        pred        = session.run(None, {input_name: current_input})[0]
        new_frame   = np.expand_dims(pred, axis=1)
        current_input = np.concatenate((current_input[:, 1:], new_frame), axis=1)

    # Upscale back to display size with cubic interpolation for quality
    raw_64     = pred[0, :, :, 0]
    final_ndvi = cv2.resize(raw_64, (MAP_SIZE, MAP_SIZE),
                            interpolation=cv2.INTER_CUBIC)
    # Smooth slight artefacts
    final_ndvi = cv2.GaussianBlur(final_ndvi, (3, 3), 0)
    return np.clip(final_ndvi, -1, 1)


# ─────────────────────────────────────────────
#  HIGH-QUALITY FIGURE BUILDER
# ─────────────────────────────────────────────
def make_figures(ndvi: np.ndarray, classified: np.ndarray,
                 target_year: int, stats: dict):

    cmap_land = mcolors.ListedColormap(LAND_COLORS)

    # ── Figure 1: NDVI heatmap ─────────────────
    fig1, ax1 = plt.subplots(figsize=(7, 7))
    fig1.patch.set_facecolor("#0d1f10")

    im1 = ax1.imshow(ndvi, cmap=NDVI_CMAP, vmin=-1, vmax=1,
                     interpolation='bicubic', aspect='equal')
    ax1.set_title(f"Vegetation Index (NDVI)  ·  {target_year}",
                  color="#a5d6a7", fontsize=12, fontweight="bold", pad=10)
    ax1.axis("off")

    # colorbar
    cbar1 = fig1.colorbar(im1, ax=ax1, fraction=0.035, pad=0.02,
                          orientation='vertical')
    cbar1.set_label("NDVI  (−1 = water/bare   →   +1 = dense green)",
                    color="#81c784", fontsize=9)
    cbar1.ax.yaxis.set_tick_params(color="#81c784")
    plt.setp(cbar1.ax.yaxis.get_ticklabels(), color="#81c784", fontsize=8)
    cbar1.outline.set_edgecolor("rgba(76,175,80,0.30)")

    fig1.tight_layout(pad=0.5)

    # ── Figure 2: Classified land cover ───────
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    fig2.patch.set_facecolor("#0d1f10")

    im2 = ax2.imshow(classified, cmap=cmap_land, vmin=0, vmax=3,
                     interpolation='nearest', aspect='equal')
    ax2.set_title(f"Land Cover Classification  ·  {target_year}",
                  color="#a5d6a7", fontsize=12, fontweight="bold", pad=10)
    ax2.axis("off")

    # legend patches
    patches = [mpatches.Patch(color=LAND_COLORS[i],
                               label=f"{LAND_LABELS[i]}  ({stats[LAND_LABELS[i]]:.1f}%)")
               for i in range(4)]
    legend = ax2.legend(
        handles=patches,
        loc="lower left",
        fontsize=8.5,
        framealpha=0.35,
        facecolor="#0d2410",
        edgecolor="rgba(76,175,80,0.30)",
        labelcolor="#c8e6c9",
        title="Land Cover",
        title_fontsize=9,
    )
    legend.get_title().set_color("#81c784")

    fig2.tight_layout(pad=0.5)

    # ── Figure 3: Bar chart of coverage ───────
    fig3, ax3 = plt.subplots(figsize=(7, 3.2))
    fig3.patch.set_facecolor("#0d1f10")
    ax3.set_facecolor("#0d1f10")

    vals   = [stats[l] for l in LAND_LABELS]
    bars   = ax3.barh(LAND_LABELS, vals, color=LAND_COLORS,
                      height=0.55, edgecolor="none")
    # value labels
    for bar, val in zip(bars, vals):
        ax3.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", ha="left",
                 color="#c8e6c9", fontsize=8.5, fontweight="600")

    ax3.set_xlim(0, 105)
    ax3.set_xlabel("Coverage (%)", color="#81c784", fontsize=9)
    ax3.set_title(f"Coverage Breakdown  ·  {target_year}",
                  color="#a5d6a7", fontsize=11, fontweight="bold", pad=8)
    ax3.tick_params(colors="#81c784", labelsize=8)
    ax3.spines[:].set_visible(False)
    ax3.xaxis.label.set_color("#81c784")
    for spine in ax3.spines.values():
        spine.set_edgecolor("rgba(76,175,80,0.20)")
    ax3.grid(axis="x", color="rgba(76,175,80,0.12)", linestyle="--", linewidth=0.7)

    fig3.tight_layout(pad=0.6)

    return fig1, fig2, fig3


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class='fade-in' style='text-align:center; padding:28px 0 6px;'>
    <div>
        <span class='leaf1'>🌿</span>&nbsp;
        <span class='leaf2'>🌳</span>&nbsp;
        <span class='leaf3'>🌱</span>
    </div>
    <h1 style='font-family:Lora,serif; font-size:2.6rem; font-weight:700;
               color:#a5d6a7; margin:10px 0 4px; letter-spacing:-0.3px;'>
        Ahmedabad Greenery Predictor
    </h1>
    <p style='color:rgba(168,214,170,0.60); font-size:1.02rem; margin:0;'>
        Urban vegetation forecasting using ConvLSTM deep learning
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='margin:18px 0 22px;'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:18px 0 10px;'>
        <span style='font-size:2.4rem;'>🌳</span>
        <h2 style='font-family:Lora,serif; font-size:1.35rem; font-weight:700;
                   color:#a5d6a7; margin:6px 0 2px;'>Prediction Settings</h2>
        <p style='color:rgba(168,214,170,0.50); font-size:0.82rem; margin:0;'>
            Configure your forecast
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    years_ahead = st.slider(
        "📅  Years into Future",
        min_value=1, max_value=50, value=10, step=1,
        help="Predict greenery up to 50 years from 2024"
    )

    target_year = BASE_YEAR + years_ahead

    st.markdown(f"""
    <div style='text-align:center; margin:14px 0;'>
        <div style='background:rgba(56,142,60,0.18); border:1px solid rgba(102,187,106,0.30);
                    border-radius:14px; padding:12px;'>
            <div style='font-size:0.72rem; color:rgba(168,214,170,0.55);
                        text-transform:uppercase; letter-spacing:1px; font-weight:700;'>
                Target Year
            </div>
            <div style='font-family:Lora,serif; font-size:2.2rem; font-weight:700;
                        color:#a5d6a7; margin-top:2px;'>{target_year}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Time horizon label
    if years_ahead <= 10:
        horizon_label, horizon_color = "Short-Term", "#81c784"
    elif years_ahead <= 25:
        horizon_label, horizon_color = "Mid-Term", "#ffb74d"
    else:
        horizon_label, horizon_color = "Long-Term", "#ef9a9a"

    st.markdown(f"""
    <div style='text-align:center; margin-bottom:16px;'>
        <span style='background:rgba(0,0,0,0.20); border-radius:50px; padding:4px 14px;
                     font-size:0.80rem; font-weight:700; color:{horizon_color};'>
            {horizon_label} Forecast
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    predict_btn = st.button("🔍  Generate Future Map", type="primary")

    st.markdown("""
    <div style='margin-top:22px; padding:14px; background:rgba(56,142,60,0.10);
                border-radius:12px; border:1px solid rgba(76,175,80,0.18);'>
        <p style='color:rgba(168,214,170,0.55); font-size:0.78rem; line-height:1.55; margin:0;'>
            ℹ️ Model auto-iterates year-by-year using ConvLSTM.
            Longer horizons compound uncertainty — treat as indicative trends.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────

# Info strip
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class='stat-box fade-in'>
        <div class='stat-value'>2024</div>
        <div class='stat-label'>Baseline Year</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class='stat-box fade-in-2'>
        <div class='stat-value'>{target_year}</div>
        <div class='stat-label'>Target Year</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class='stat-box fade-in-2'>
        <div class='stat-value'>{years_ahead}</div>
        <div class='stat-label'>Years Ahead</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class='stat-box fade-in-3'>
        <div class='stat-value'>64px</div>
        <div class='stat-label'>Model Resolution</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── PREDICTION ──
if predict_btn:
    progress_bar = st.progress(0, text="Initialising model…")

    with st.spinner(f"🌿 Simulating Ahmedabad in {target_year}…"):

        progress_bar.progress(10, text="Preparing seed frames…")
        ndvi       = run_prediction(years_ahead)
        progress_bar.progress(60, text="Classifying land cover…")
        classified = classify_ndvi(ndvi)
        stats      = coverage_pct(classified)
        progress_bar.progress(80, text="Rendering visualisations…")
        fig1, fig2, fig3 = make_figures(ndvi, classified, target_year, stats)
        progress_bar.progress(100, text="Done!")
        import time; time.sleep(0.3)
        progress_bar.empty()

    # ── Coverage headline stats ──
    st.markdown(f"""
    <div class='nature-card fade-in' style='text-align:center;'>
        <p style='color:rgba(168,214,170,0.55); font-size:0.76rem;
                  text-transform:uppercase; letter-spacing:1.5px; margin-bottom:6px;'>
            Predicted Land Cover · {target_year}
        </p>
        <span class='year-badge'>+{years_ahead} years from baseline</span>
    </div>
    """, unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, label, color in zip(
        [sc1, sc2, sc3, sc4], LAND_LABELS, LAND_COLORS
    ):
        with col:
            st.markdown(f"""
            <div class='stat-box'>
                <div style='width:10px;height:10px;border-radius:3px;
                            background:{color};margin:0 auto 6px;'></div>
                <div class='stat-value' style='color:{color};font-size:1.7rem;'>
                    {stats[label]:.1f}%
                </div>
                <div class='stat-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Maps side by side ──
    map_col1, map_col2 = st.columns(2, gap="medium")

    with map_col1:
        st.markdown(f"""
        <div class='nature-card' style='padding:16px;'>
            <p style='color:#81c784; font-size:0.80rem; text-transform:uppercase;
                      letter-spacing:1px; font-weight:700; margin-bottom:8px;'>
                🌡️ NDVI Heatmap
            </p>""", unsafe_allow_html=True)
        st.pyplot(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with map_col2:
        st.markdown(f"""
        <div class='nature-card' style='padding:16px;'>
            <p style='color:#81c784; font-size:0.80rem; text-transform:uppercase;
                      letter-spacing:1px; font-weight:700; margin-bottom:8px;'>
                🗺️ Classified Land Cover
            </p>""", unsafe_allow_html=True)
        st.pyplot(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Coverage bar chart ──
    st.markdown("<div class='nature-card' style='padding:20px 24px;'>", unsafe_allow_html=True)
    st.markdown("""<p style='color:#81c784; font-size:0.80rem; text-transform:uppercase;
                             letter-spacing:1px; font-weight:700; margin-bottom:10px;'>
                    📊 Coverage Breakdown
                </p>""", unsafe_allow_html=True)
    st.pyplot(fig3, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Legend card ──
    st.markdown("<div class='nature-card'>", unsafe_allow_html=True)
    st.markdown("""<p style='color:#81c784; font-size:0.80rem; text-transform:uppercase;
                             letter-spacing:1px; font-weight:700; margin-bottom:10px;'>
                    📖 Classification Guide
                </p>""", unsafe_allow_html=True)

    legend_info = [
        ("#1565C0", "Water / Wetland",    "NDVI < 0",         "Rivers, lakes, waterlogged areas"),
        ("#607D8B", "Built-up / Urban",   "0 ≤ NDVI < 0.2",   "Roads, buildings, concrete surfaces"),
        ("#AED581", "Sparse Vegetation",  "0.2 ≤ NDVI < 0.4", "Parks, scrubland, thin canopy"),
        ("#2E7D32", "Dense Greenery",     "NDVI ≥ 0.4",        "Forests, thick vegetation, farmland"),
    ]
    for color, label, ndvi_range, desc in legend_info:
        st.markdown(f"""
        <div class='legend-row'>
            <div class='legend-dot' style='background:{color};'></div>
            <div style='flex:1.2; font-weight:700; color:#c8e6c9; font-size:0.88rem;'>{label}</div>
            <div style='flex:1.0; color:rgba(168,214,170,0.60); font-size:0.82rem;
                        font-family:monospace;'>{ndvi_range}</div>
            <div style='flex:2.5; color:rgba(168,214,170,0.52); font-size:0.82rem;'>{desc}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Disclaimer ──
    st.markdown(f"""
    <div style='background:rgba(255,193,7,0.06); border-left:3px solid rgba(255,193,7,0.40);
                border-radius:10px; padding:13px 16px; margin-top:4px;'>
        <p style='color:rgba(255,224,130,0.70); font-size:0.82rem; margin:0; line-height:1.55;'>
            ⚠️ <strong style='color:rgba(255,224,130,0.90);'>Model Note:</strong>
            Predictions beyond 15 years compound iterative uncertainty. Treat long-range forecasts
            as indicative trajectories, not precise values. The model was trained on Ahmedabad
            satellite data (2019–2024) and extrapolates vegetation patterns under stable conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.success(f"✅  Simulation complete — Ahmedabad predicted for {target_year}!")

else:
    # ── Idle state placeholder ──
    st.markdown("""
    <div class='nature-card fade-in' style='text-align:center; padding:52px 36px;'>
        <div style='font-size:3.5rem; margin-bottom:14px;'>🌿</div>
        <h3 style='font-family:Lora,serif; color:#a5d6a7; margin-bottom:8px;'>
            Ready to Forecast
        </h3>
        <p style='color:rgba(168,214,170,0.55); font-size:0.96rem; max-width:420px; margin:0 auto;'>
            Set your target year using the slider on the left,
            then click <strong style='color:#81c784;'>Generate Future Map</strong>
            to see Ahmedabad's predicted greenery.
        </p>
        <div style='margin-top:24px; display:flex; justify-content:center; gap:28px;'>
            <div style='text-align:center;'>
                <div style='font-size:1.6rem;'>📡</div>
                <div style='font-size:0.76rem; color:rgba(168,214,170,0.45); margin-top:4px;'>ConvLSTM Model</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:1.6rem;'>🗓️</div>
                <div style='font-size:0.76rem; color:rgba(168,214,170,0.45); margin-top:4px;'>Up to 50 Years</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:1.6rem;'>🗺️</div>
                <div style='font-size:0.76rem; color:rgba(168,214,170,0.45); margin-top:4px;'>722×722 Output</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div class='footer-text'>
    🌳 &nbsp;Ahmedabad Greenery Predictor &nbsp;|&nbsp;
    ConvLSTM + ONNX &nbsp;|&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
