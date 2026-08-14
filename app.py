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
    page_title="AI Exoplanet Detection",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 AI-Exoplanet Detection Portal")
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
            st.metric("Total Da
