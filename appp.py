import streamlit as st
import numpy as np
import tensorflow as tf
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
st.subheader("Predicting urban vegetation using ConvLSTM Deep Learning")

# --- LOAD DATA & MODEL ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('convlstm_greenery_model.h5')
    seed = np.load('seed_data.npy')
    return model, seed

model, seed_data = load_assets()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Prediction Settings")
    years_to_jump = st.slider("Select Years into Future", 1, 15, 10)
    predict_btn = st.button("Generate Future Map", type="primary")

# --- PREDICTION LOGIC ---
if predict_btn:
    with st.spinner(f"Simulating Ahmedabad in {2024 + years_to_jump}..."):
        # This is a simplified version of your stitching logic
        # For the app, we use the seed_data to predict the future frames
        
        current_input = np.expand_dims(seed_data, axis=0) # (1, 5, H, W, 1)
        
        # Prediction loop
        for _ in range(years_to_jump):
            pred = model.predict(current_input, verbose=0)
            new_frame = np.expand_dims(pred, axis=1)
            current_input = np.concatenate((current_input[:, 1:], new_frame), axis=1)
        
        final_ndvi = pred[0, :, :, 0]

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