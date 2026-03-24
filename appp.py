import streamlit as st
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import cv2

st.set_page_config(
    page_title="Ahmedabad Greenery Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp { font-family: 'DM Sans', sans-serif; }

/* CHANGED: dark navy/indigo instead of dark green */
.stApp {
    background: linear-gradient(160deg, #0a0e1a 0%, #111827 35%, #0f172a 60%, #080c18 100%);
    min-height: 100vh;
    color: #e2e8f0;
}

/* CHANGED: overlay uses indigo/purple tones */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(ellipse at 20% 10%, rgba(99,102,241,0.10) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(67,56,202,0.14) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(0,0,0,0.12) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* CHANGED: sidebar navy/slate */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060912 0%, #0d1117 60%, #050810 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.22) !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #a5b4fc !important;
    font-family: 'Lora', serif !important;
}
[data-testid="stSidebar"] .stSlider label { color: #818cf8 !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #818cf8 !important; }

[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div {
    background: rgba(99,102,241,0.25) !important;
}

/* CHANGED: headings to lavender */
h1, h2, h3, h4 {
    font-family: 'Lora', serif !important;
    color: #a5b4fc !important;
}

/* CHANGED: glass card uses slate/indigo border */
.nature-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 20px;
    border: 1px solid rgba(99,102,241,0.20);
    padding: 24px 28px;
    margin-bottom: 18px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.40), inset 0 1px 0 rgba(255,255,255,0.05);
    transition: transform 0.25s ease, border-color 0.25s ease;
}
.nature-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99,102,241,0.38);
}

/* CHANGED: stat box slate/indigo */
.stat-box {
    background: rgba(99,102,241,0.10);
    border: 1px solid rgba(99,102,241,0.22);
    border-radius: 16px;
    padding: 18px 14px;
    text-align: center;
    transition: transform 0.22s ease, background 0.22s ease;
    cursor: default;
}
.stat-box:hover {
    transform: translateY(-3px) scale(1.02);
    background: rgba(99,102,241,0.18);
}
.stat-value { font-size: 1.9rem; font-weight: 700; color: #a5b4fc; }
.stat-label { font-size: 0.75rem; color: rgba(203,213,225,0.55); text-transform: uppercase; letter-spacing: 0.8px; margin-top: 3px; font-weight: 600; }

/* CHANGED: legend row uses indigo hover */
.legend-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 10px; border-radius: 10px;
    transition: background 0.18s ease, padding-left 0.18s ease;
    cursor: default;
    border-bottom: 1px solid rgba(99,102,241,0.10);
}
.legend-row:last-child { border-bottom: none; }
.legend-row:hover { background: rgba(99,102,241,0.10); padding-left: 16px; }
.legend-dot { width: 14px; height: 14px; border-radius: 4px; flex-shrink: 0; }

/* CHANGED: button indigo */
.stButton > button {
    width: 100%;
    border-radius: 50px;
    height: 3.0em;
    background: linear-gradient(135deg, #3730a3 0%, #4f46e5 60%, #6366f1 100%);
    color: #eef2ff;
    border: none;
    font-weight: 700;
    font-size: 0.97rem;
    letter-spacing: 0.8px;
    box-shadow: 0 4px 18px rgba(79,70,229,0.42);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 28px rgba(99,102,241,0.58);
}
.stButton > button:active { transform: scale(0.97); }

.stSlider [data-baseweb="slider"] { padding: 0 4px; }
.stSpinner > div { border-color: #6366f1 transparent transparent transparent !important; }

.stSuccess { background: rgba(56,142,60,0.18) !important; border-color: #4caf50 !important; }
.stSuccess p { color: #bbf7d0 !important; }
.stInfo p { color: #bfdbfe !important; }

div[data-testid="stMetricValue"] {
    color: #a5b4fc !important;
    font-family: 'Lora', serif;
}
div[data-testid="stMetricLabel"] { color: rgba(203,213,225,0.60) !important; }

hr { border-color: rgba(99,102,241,0.18) !important; }

.footer-text {
    color: rgba(203,213,225,0.30);
    font-size: 0.75rem;
    text-align: center;
    margin-top: 32px;
    padding-bottom: 12px;
}

/* CHANGED: year badge indigo */
.year-badge {
    display: inline-block;
    background: linear-gradient(135deg, #312e81, #3730a3);
    border: 1px solid rgba(129,140,248,0.45);
    border-radius: 50px;
    padding: 4px 18px;
    font-family: 'Lora', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #c7d2fe;
    letter-spacing: 0.5px;
}

/* Animations — identical */
@keyframes floatLeaf {
    0%,100% { transform: translateY(0) rotate(0deg); opacity: 0.6; }
    33%      { transform: translateY(-12px) rotate(8deg); opacity: 0.9; }
    66%      { transform: translateY(-6px) rotate(-5deg); opacity: 0.75; }
}
.leaf1 { animation: floatLeaf 4.0s ease-in-out infinite; display:inline-block; }
.leaf2 { animation: floatLeaf 3.5s ease-in-out 0.8s infinite; display:inline-block; }
.leaf3 { animation: floatLeaf 4.5s ease-in-out 1.5s infinite; display:inline-block; }

@keyframes fadeUp {
    from { opacity:0; transform:translateY(14px); }
    to   { opacity:1; transform:translateY(0); }
}
.fade-in   { animation: fadeUp 0.60s ease-out both; }
.fade-in-2 { animation: fadeUp 0.60s ease-out 0.12s both; }
.fade-in-3 { animation: fadeUp 0.60s ease-out 0.24s both; }

.stpyplot { border-radius: 16px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# CHANGED: matplotlib theme — deep navy instead of dark forest green
plt.rcParams.update({
    "figure.facecolor":  "#0a0e1a",
    "axes.facecolor":    "#0a0e1a",
    "text.color":        "#cbd5e1",
    "axes.titlecolor":   "#a5b4fc",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlepad":     12,
    "font.family":       "sans-serif",
    "savefig.facecolor": "#0a0e1a",
    "savefig.edgecolor": "none",
})


BASE_YEAR    = 2024
MAP_SIZE     = 722
MODEL_SIZE   = 64
SEQ_LEN      = 5

LAND_COLORS  = ['#1565C0', '#607D8B', '#AED581', '#2E7D32']
LAND_LABELS  = ['Water / Wetland', 'Built-up / Urban', 'Sparse Vegetation', 'Dense Greenery']

NDVI_CMAP    = 'RdYlGn'


@st.cache_resource(show_spinner=False)
def load_assets():
    session = ort.InferenceSession('model_flexible.onnx')
    seed    = np.load('seed_data.npy')
    return session, seed

session, seed_data = load_assets()


def classify_ndvi(ndvi: np.ndarray) -> np.ndarray:
    out = np.zeros_like(ndvi, dtype=np.uint8)
    out[ndvi < 0]                              = 0
    out[(ndvi >= 0)   & (ndvi < 0.2)]          = 1
    out[(ndvi >= 0.2) & (ndvi < 0.4)]          = 2
    out[ndvi >= 0.4]                           = 3
    return out


def coverage_pct(classified: np.ndarray):
    total = classified.size
    return {lbl: float((classified == i).sum() / total * 100)
            for i, lbl in enumerate(LAND_LABELS)}


def run_prediction(years_ahead: int):
    frames = []
    for i in range(SEQ_LEN):
        f = cv2.resize(seed_data[i, :, :, 0], (MODEL_SIZE, MODEL_SIZE),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
        frames.append(f)

    current_input = np.array(frames)[np.newaxis, :, :, :, np.newaxis]
    input_name    = session.get_inputs()[0].name

    pred = None
    for _ in range(years_ahead):
        pred        = session.run(None, {input_name: current_input})[0]
        new_frame   = np.expand_dims(pred, axis=1)
        current_input = np.concatenate((current_input[:, 1:], new_frame), axis=1)

    raw_64     = pred[0, :, :, 0]
    final_ndvi = cv2.resize(raw_64, (MAP_SIZE, MAP_SIZE),
                            interpolation=cv2.INTER_CUBIC)
    final_ndvi = cv2.GaussianBlur(final_ndvi, (3, 3), 0)
    return np.clip(final_ndvi, -1, 1)


def make_figures(ndvi: np.ndarray, classified: np.ndarray,
                 target_year: int, stats: dict):

    cmap_land = mcolors.ListedColormap(LAND_COLORS)

    # CHANGED: figure bg navy
    fig1, ax1 = plt.subplots(figsize=(7, 7))
    fig1.patch.set_facecolor("#0a0e1a")

    im1 = ax1.imshow(ndvi, cmap=NDVI_CMAP, vmin=-1, vmax=1,
                     interpolation='bicubic', aspect='equal')
    ax1.set_title(f"Vegetation Index (NDVI)  ·  {target_year}",
                  color="#a5b4fc", fontsize=12, fontweight="bold", pad=10)
    ax1.axis("off")

    cbar1 = fig1.colorbar(im1, ax=ax1, fraction=0.035, pad=0.02,
                          orientation='vertical')
    cbar1.set_label("NDVI  (−1 = water/bare   →   +1 = dense green)",
                    color="#818cf8", fontsize=9)
    cbar1.ax.yaxis.set_tick_params(color="#818cf8")
    plt.setp(cbar1.ax.yaxis.get_ticklabels(), color="#818cf8", fontsize=8)
    # CHANGED: indigo tuple instead of green
    cbar1.outline.set_edgecolor((0.388, 0.400, 0.945, 0.30))

    fig1.tight_layout(pad=0.5)

    # CHANGED: figure bg navy
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    fig2.patch.set_facecolor("#0a0e1a")

    im2 = ax2.imshow(classified, cmap=cmap_land, vmin=0, vmax=3,
                     interpolation='nearest', aspect='equal')
    ax2.set_title(f"Land Cover Classification  ·  {target_year}",
                  color="#a5b4fc", fontsize=12, fontweight="bold", pad=10)
    ax2.axis("off")

    patches = [mpatches.Patch(color=LAND_COLORS[i],
                               label=f"{LAND_LABELS[i]}  ({stats[LAND_LABELS[i]]:.1f}%)")
               for i in range(4)]
    legend = ax2.legend(
        handles=patches,
        loc="lower left",
        fontsize=8.5,
        framealpha=0.40,
        facecolor="#0d1117",
        # CHANGED: indigo border tuple
        edgecolor=(0.388, 0.400, 0.945, 0.35),
        labelcolor="#cbd5e1",
        title="Land Cover",
        title_fontsize=9,
    )
    legend.get_title().set_color("#818cf8")

    fig2.tight_layout(pad=0.5)

    # CHANGED: figure bg navy
    fig3, ax3 = plt.subplots(figsize=(7, 3.2))
    fig3.patch.set_facecolor("#0a0e1a")
    ax3.set_facecolor("#0a0e1a")

    vals   = [stats[l] for l in LAND_LABELS]
    bars   = ax3.barh(LAND_LABELS, vals, color=LAND_COLORS,
                      height=0.55, edgecolor="none")
    for bar, val in zip(bars, vals):
        ax3.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", ha="left",
                 color="#cbd5e1", fontsize=8.5, fontweight="600")

    ax3.set_xlim(0, 105)
    ax3.set_xlabel("Coverage (%)", color="#818cf8", fontsize=9)
    ax3.set_title(f"Coverage Breakdown  ·  {target_year}",
                  color="#a5b4fc", fontsize=11, fontweight="bold", pad=8)
    ax3.tick_params(colors="#818cf8", labelsize=8)
    ax3.spines[:].set_visible(False)
    ax3.xaxis.label.set_color("#818cf8")
    for spine in ax3.spines.values():
        # CHANGED: indigo tuple
        spine.set_edgecolor((0.388, 0.400, 0.945, 0.20))
    # CHANGED: indigo tuple
    ax3.grid(axis="x", color=(0.388, 0.400, 0.945, 0.12), linestyle="--", linewidth=0.7)

    fig3.tight_layout(pad=0.6)

    return fig1, fig2, fig3


# HEADER — identical, only accent colour in inline style changed to lavender
st.markdown("""
<div class='fade-in' style='text-align:center; padding:28px 0 6px;'>
    <div>
        <span class='leaf1'>🌿</span>&nbsp;
        <span class='leaf2'>🌳</span>&nbsp;
        <span class='leaf3'>🌱</span>
    </div>
    <h1 style='font-family:Lora,serif; font-size:2.6rem; font-weight:700;
               color:#a5b4fc; margin:10px 0 4px; letter-spacing:-0.3px;'>
        Ahmedabad Greenery Predictor
    </h1>
    <p style='color:rgba(165,180,252,0.60); font-size:1.02rem; margin:0;'>
        Urban vegetation forecasting using ConvLSTM deep learning
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='margin:18px 0 22px;'>", unsafe_allow_html=True)


# SIDEBAR — identical structure, accent colours updated
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:18px 0 10px;'>
        <span style='font-size:2.4rem;'>🌳</span>
        <h2 style='font-family:Lora,serif; font-size:1.35rem; font-weight:700;
                   color:#a5b4fc; margin:6px 0 2px;'>Prediction Settings</h2>
        <p style='color:rgba(165,180,252,0.50); font-size:0.82rem; margin:0;'>
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
        <div style='background:rgba(99,102,241,0.16); border:1px solid rgba(129,140,248,0.30);
                    border-radius:14px; padding:12px;'>
            <div style='font-size:0.72rem; color:rgba(165,180,252,0.55);
                        text-transform:uppercase; letter-spacing:1px; font-weight:700;'>
                Target Year
            </div>
            <div style='font-family:Lora,serif; font-size:2.2rem; font-weight:700;
                        color:#a5b4fc; margin-top:2px;'>{target_year}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if years_ahead <= 10:
        horizon_label, horizon_color = "Short-Term", "#86efac"
    elif years_ahead <= 25:
        horizon_label, horizon_color = "Mid-Term",   "#fcd34d"
    else:
        horizon_label, horizon_color = "Long-Term",  "#fca5a5"

    st.markdown(f"""
    <div style='text-align:center; margin-bottom:16px;'>
        <span style='background:rgba(0,0,0,0.25); border-radius:50px; padding:4px 14px;
                     font-size:0.80rem; font-weight:700; color:{horizon_color};'>
            {horizon_label} Forecast
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    predict_btn = st.button("🔍  Generate Future Map", type="primary")

    st.markdown("""
    <div style='margin-top:22px; padding:14px; background:rgba(99,102,241,0.08);
                border-radius:12px; border:1px solid rgba(99,102,241,0.18);'>
        <p style='color:rgba(165,180,252,0.55); font-size:0.78rem; line-height:1.55; margin:0;'>
            ℹ️ Model auto-iterates year-by-year using ConvLSTM.
            Longer horizons compound uncertainty — treat as indicative trends.
        </p>
    </div>
    """, unsafe_allow_html=True)


# MAIN CONTENT — identical structure, accent colours updated
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

    st.markdown(f"""
    <div class='nature-card fade-in' style='text-align:center;'>
        <p style='color:rgba(165,180,252,0.55); font-size:0.76rem;
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

    map_col1, map_col2 = st.columns(2, gap="medium")

    with map_col1:
        st.markdown("""
        <div class='nature-card' style='padding:16px;'>
            <p style='color:#818cf8; font-size:0.80rem; text-transform:uppercase;
                      letter-spacing:1px; font-weight:700; margin-bottom:8px;'>
                🌡️ NDVI Heatmap
            </p>""", unsafe_allow_html=True)
        st.pyplot(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with map_col2:
        st.markdown("""
        <div class='nature-card' style='padding:16px;'>
            <p style='color:#818cf8; font-size:0.80rem; text-transform:uppercase;
                      letter-spacing:1px; font-weight:700; margin-bottom:8px;'>
                🗺️ Classified Land Cover
            </p>""", unsafe_allow_html=True)
        st.pyplot(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='nature-card' style='padding:20px 24px;'>", unsafe_allow_html=True)
    st.markdown("""<p style='color:#818cf8; font-size:0.80rem; text-transform:uppercase;
                             letter-spacing:1px; font-weight:700; margin-bottom:10px;'>
                    📊 Coverage Breakdown
                </p>""", unsafe_allow_html=True)
    st.pyplot(fig3, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='nature-card'>", unsafe_allow_html=True)
    st.markdown("""<p style='color:#818cf8; font-size:0.80rem; text-transform:uppercase;
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
            <div style='flex:1.2; font-weight:700; color:#cbd5e1; font-size:0.88rem;'>{label}</div>
            <div style='flex:1.0; color:rgba(165,180,252,0.60); font-size:0.82rem;
                        font-family:monospace;'>{ndvi_range}</div>
            <div style='flex:2.5; color:rgba(203,213,225,0.50); font-size:0.82rem;'>{desc}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:rgba(255,193,7,0.06); border-left:3px solid rgba(255,193,7,0.40);
                border-radius:10px; padding:13px 16px; margin-top:4px;'>
        <p style='color:rgba(255,224,130,0.70); font-size:0.82rem; margin:0; line-height:1.55;'>
            ⚠️ <strong style='color:rgba(255,224,130,0.90);'>Model Note:</strong>
            Predictions beyond 15 years compound iterative uncertainty. Treat long-range forecasts
            as indicative trajectories, not precise values. The model was trained on Ahmedabad
            satellite data (2001–2024) and extrapolates vegetation patterns under stable conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.success(f"✅  Simulation complete — Ahmedabad predicted for {target_year}!")

else:
    st.markdown("""
    <div class='nature-card fade-in' style='text-align:center; padding:52px 36px;'>
        <div style='font-size:3.5rem; margin-bottom:14px;'>🌿</div>
        <h3 style='font-family:Lora,serif; color:#a5b4fc; margin-bottom:8px;'>
            Ready to Forecast
        </h3>
        <p style='color:rgba(165,180,252,0.55); font-size:0.96rem; max-width:420px; margin:0 auto;'>
            Set your target year using the slider on the left,
            then click <strong style='color:#818cf8;'>Generate Future Map</strong>
            to see Ahmedabad's predicted greenery.
        </p>
        <div style='margin-top:24px; display:flex; justify-content:center; gap:28px;'>
            <div style='text-align:center;'>
                <div style='font-size:1.6rem;'>📡</div>
                <div style='font-size:0.76rem; color:rgba(165,180,252,0.45); margin-top:4px;'>ConvLSTM Model</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:1.6rem;'>🗓️</div>
                <div style='font-size:0.76rem; color:rgba(165,180,252,0.45); margin-top:4px;'>Up to 50 Years</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:1.6rem;'>🗺️</div>
                <div style='font-size:0.76rem; color:rgba(165,180,252,0.45); margin-top:4px;'>722×722 Output</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class='footer-text'>
    🌳 &nbsp;Ahmedabad Greenery Predictor &nbsp;|&nbsp;
    ConvLSTM + ONNX &nbsp;|&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
