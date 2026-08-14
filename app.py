# pyrefly: ignore [missing-import]
import streamlit as st
import lightkurve as lk
import numpy as np
import pandas as pd
import pywt
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import json
import os
import pickle

# Set page configuration
st.set_page_config(
    page_title="ISRO Hackathon: AI Exoplanet Detection",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 ISRO Hackathon: AI-Exoplanet Detection Portal")
st.write("Upload a TESS or Kepler FITS light curve file to execute the pipeline, detect planets, and generate a scientific report.")

# Custom Attention Layer (same as notebook for compatibility)
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)
    def call(self, x):
        score = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(x * attention_weights, axis=1)
        return context_vector, attention_weights
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1], 1)]

# Preprocessing helpers
def preprocess_lightcurve(lc):
    lc = lc.remove_nans().normalize()
    flat_lc = lc.flatten(window_length=301)
    flux = flat_lc.flux.value
    coeffs = pywt.wavedec(flux, 'db4', level=4)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(flux)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    denoised_flux = pywt.waverec(coeffs, 'db4')
    if len(denoised_flux) > len(flux):
        denoised_flux = denoised_flux[:len(flux)]
    elif len(denoised_flux) < len(flux):
        denoised_flux = np.pad(denoised_flux, (0, len(flux) - len(denoised_flux)), 'edge')
    return flat_lc, denoised_flux

# Load model weights on the fly
@st.cache_resource
def load_models():
    model_path = 'best_advanced_model.keras'
    rf_path = 'rf_baseline.pkl'
    
    dl_model = None
    rf_model = None
    
    if os.path.exists(model_path):
        try:
            dl_model = tf.keras.models.load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
        except Exception as e:
            st.error(f"Error loading Deep Learning model: {e}")
            
    if os.path.exists(rf_path):
        try:
            with open(rf_path, 'rb') as f:
                rf_model = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading Random Forest baseline: {e}")
            
    return dl_model, rf_model

dl_model, rf_model = load_models()

# File Upload Panel
uploaded_file = st.file_uploader("Upload MAST FITS File (.fits)", type=["fits"])

if uploaded_file is not None:
    # Save uploaded file to disk temporarily
    temp_path = uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    try:
        with st.spinner("Reading FITS file..."):
            lc = lk.read(temp_path)
            
        st.success("Successfully loaded FITS file!")
        
        # Display meta-information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Target Name", getattr(lc, 'targetid', 'Unknown') or getattr(lc, 'label', 'Unknown'))
        with col2:
            st.metric("Mission", getattr(lc, 'meta', {}).get('MISSION', 'TESS/Kepler'))
        with col3:
            st.metric("Total Data Cadences", len(lc))

        # Preprocessing Step
        with st.spinner("Executing Detrending, Flattening, and Wavelet Denoising..."):
            flat_lc, denoised_flux = preprocess_lightcurve(lc)
            
        # Plot raw vs denoised
        st.subheader("📈 Light Curve Preprocessing Visualizations")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(flat_lc.time.value, flat_lc.flux.value, label='Normalized Flat Flux', alpha=0.4, color='gray')
        ax.plot(flat_lc.time.value, denoised_flux, label='Wavelet Denoised Flux (db4)', color='blue', linewidth=1.5)
        ax.set_xlabel("Time (Days)")
        ax.set_ylabel("Normalized Flux")
        ax.legend()
        st.pyplot(fig)
        
        # Transit parameter extraction
        with st.spinner("Running Box Least Squares (BLS) Periodogram..."):
            periods = np.linspace(0.5, 45, 1000)
            durations = np.linspace(0.01, 0.4, 5)
            bls = flat_lc.to_periodogram(method='bls', period=periods, duration=durations)
            
            estimated_period = bls.period_at_max_power.value
            transit_time = bls.transit_time_at_max_power.value
            transit_depth = bls.depth_at_max_power.value
            transit_duration = bls.duration_at_max_power.value
            max_power_idx = np.argmax(bls.power)
            signal_to_noise_ratio = bls.snr[max_power_idx].value
            
        st.subheader("🪐 Extracted Transit Physical Parameters")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Estimated Period", f"{estimated_period:.3f} days")
        with col2:
            st.metric("Transit Depth", f"{transit_depth * 100:.4f}%")
        with col3:
            st.metric("Transit Duration", f"{transit_duration * 24:.2f} hours")
        with col4:
            st.metric("Signal to Noise Ratio (SNR)", f"{signal_to_noise_ratio:.2f}")
            
        # Model predictions
        with st.spinner("Running AI Model Inference..."):
            # Slice windows
            X_target = []
            sequence_length = 256
            step = 64
            for start_idx in range(0, len(denoised_flux) - sequence_length, step):
                window = denoised_flux[start_idx : start_idx + sequence_length]
                window = (window - np.mean(window)) / np.std(window)
                X_target.append(window)
            X_target = np.array(X_target)
            
            if len(X_target) > 0:
                X_target_dl = X_target.reshape(X_target.shape[0], X_target.shape[1], 1)
                
                # Predict
                if dl_model is not None:
                    predictions = dl_model.predict(X_target_dl, verbose=0).flatten()
                    avg_confidence = np.mean(predictions) * 100
                else:
                    avg_confidence = 50.0  # Placeholder if not loaded
                    
                # Predict using RF baseline if loaded
                if rf_model is not None:
                    features = np.array([[transit_depth, transit_duration, estimated_period, signal_to_noise_ratio]])
                    rf_confidence = rf_model.predict_proba(features)[0, 1] * 100
                else:
                    rf_confidence = 50.0
                    
                # Decision rules
                planet_detected = bool(avg_confidence > 50.0 and signal_to_noise_ratio > 5.0)
            else:
                avg_confidence = 0.0
                rf_confidence = 0.0
                planet_detected = False
                
        # Predictions Panel
        st.subheader("🤖 AI Classification Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Planet Detection Decision", "✅ Planet Detected" if planet_detected else "❌ No Planet Detected")
        with col2:
            st.metric("Deep Learning Confidence", f"{avg_confidence:.1f}%")
            
        # Explainability panel (Attention heatmap)
        if dl_model is not None and len(X_target) > 0:
            st.subheader("🔍 Attention Explainability Map")
            # Find the window with highest probability
            max_prob_idx = np.argmax(predictions)
            test_seq = X_target_dl[max_prob_idx : max_prob_idx + 1]
            
            # Setup attention extractor
            attention_extractor = tf.keras.Model(inputs=dl_model.inputs, outputs=[dl_model.outputs, dl_model.layers[-4].output])
            prob, att_w = attention_extractor.predict(test_seq, verbose=0)
            
            # Interpolate attention weights
            att_weights_1d = att_w.flatten()
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(att_weights_1d))
            x_new = np.linspace(0, 1, 256)
            f_interp = interp1d(x_old, att_weights_1d, kind='linear')
            att_weights_256 = f_interp(x_new)
            
            # Plot
            fig, ax1 = plt.subplots(figsize=(10, 3.5))
            ax1.plot(test_seq.flatten(), color='blue', label='Light Curve')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Normalized Flux', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            ax2 = ax1.twinx()
            ax2.fill_between(range(256), att_weights_256, color='red', alpha=0.3, label='Attention Weights')
            ax2.set_ylabel('Attention Weight', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            plt.title('Attention Heatmap Overlay (Focussing on the Transit Dip)')
            st.pyplot(fig)

        # Scientific JSON report
        st.subheader("📄 Generated Scientific Report (JSON)")
        report = {
            "target_id": str(getattr(lc, 'targetid', 'Unknown') or getattr(lc, 'label', 'Unknown')),
            "planet_detected": planet_detected,
            "confidence": round(avg_confidence, 1),
            "transit_depth": round(transit_depth * 100, 3),
            "transit_duration_hours": round(transit_duration * 24, 2),
            "estimated_period_days": round(estimated_period, 2),
            "signal_to_noise_ratio": round(signal_to_noise_ratio, 2)
        }
        st.json(report)
        st.download_button(
            label="Download Scientific JSON Report",
            data=json.dumps(report, indent=2),
            file_name=f"scientific_report_{report['target_id']}.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Error executing analysis: {e}")
        
    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
