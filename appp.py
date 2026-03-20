import streamlit as st
import numpy as np
import onnxruntime as ort  # Removed TensorFlow, using lightweight ONNX
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- UI CONFIG ---
st.set_page_config(page_title="Ahmedabad Greenery Predictor", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    h1 { color: #2e7d32; font-family: 'Segoe UI', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌳 Ahmedabad 2035: Greenery Forecaster")
st.subheader("Predicting urban vegetation using ConvLSTM Deep Learning (Optimized)")

# --- LOAD DATA & MODEL ---
@st.cache_resource
def load_assets():
    # Load the compressed ONNX model instead of the massive .h5 file
    session = ort.InferenceSession('model_flexible.onnx')
    seed = np.load('seed_data.npy')
    return session, seed

session, seed_data = load_assets()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Prediction Settings")
    years_to_jump = st.slider("Select Years into Future", 1, 15, 10)
    predict_btn = st.button("Generate Future Map", type="primary")

# --- PREDICTION LOGIC ---
import cv2 # Add this at the top

# ... (UI and Load Assets code stay the same) ...

if predict_btn:
    with st.spinner(f"Simulating Ahmedabad in {2024 + years_to_jump}..."):
        
        # --- STEP 1: RESIZE INPUT TO 64x64 ---
        # We resize each of the 5 years in the seed data
        resized_seed = []
        for i in range(5):
            # Resize from 722x722 down to 64x64
            img_64 = cv2.resize(seed_data[i, :, :, 0], (64, 64))
            resized_seed.append(img_64)
        
        current_input = np.array(resized_seed).astype(np.float32)
        current_input = np.expand_dims(current_input, axis=(0, -1)) # Shape: (1, 5, 64, 64, 1)
        
        input_name = session.get_inputs()[0].name

        # --- STEP 2: RUN PREDICTION ---
        for _ in range(years_to_jump):
            model_inputs = {input_name: current_input}
            pred = session.run(None, model_inputs)[0]
            new_frame = np.expand_dims(pred, axis=1)
            current_input = np.concatenate((current_input[:, 1:], new_frame), axis=1)
        
        # --- STEP 3: RESIZE OUTPUT BACK TO ORIGINAL ---
        # Get the 64x64 result
        prediction_64 = pred[0, :, :, 0]
        # Scale it back up to 722x722 so it looks detailed in the UI
        final_ndvi = cv2.resize(prediction_64, (722, 722), interpolation=cv2.INTER_CUBIC)

        # ... (rest of your classification and plotting code stays the same) ...

        # Classification
        classified = np.zeros_like(final_ndvi)
        classified[final_ndvi < 0] = 0
        classified[(final_ndvi >= 0) & (final_ndvi < 0.2)] = 1
        classified[(final_ndvi >= 0.2) & (final_ndvi < 0.4)] = 2
        classified[final_ndvi >= 0.4] = 3

        # Plotting
        col1, col2 = st.columns(2)
        colors = ['#0000FF', '#808080', '#90EE90', '#006400']
        cmap_custom = mcolors.ListedColormap(colors)

        with col1:
            st.write(f"### Heatmap ({2024 + years_to_jump})")
            fig1, ax1 = plt.subplots()
            ax1.imshow(final_ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
            ax1.axis('off')
            st.pyplot(fig1)

        with col2:
            st.write(f"### Classified Map ({2024 + years_to_jump})")
            fig2, ax2 = plt.subplots()
            ax2.imshow(classified, cmap=cmap_custom)
            ax2.axis('off')
            st.pyplot(fig2)
            
        st.success(f"Prediction for {2024 + years_to_jump} complete!")
