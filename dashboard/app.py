"""
Acoustic Reef - Main Streamlit Dashboard
AI-powered stethoscope for the ocean
"""

import streamlit as st
import numpy as np
import pandas as pd
# import librosa  # Commented out for now due to installation issues
# import matplotlib.pyplot as plt  # Commented out for now due to installation issues
# import seaborn as sns  # Commented out for now due to installation issues
from pathlib import Path
import tempfile
import os
import wave
import contextlib

from src.models.surfperch_integration import SurfPerchModel
from src.utils.config import SURFPERCH_SETTINGS, EMBEDDINGS_CSV, MASTER_DATASET_CSV, RF_MODEL_PATH
from src.models.reef_classifier import (
    load_embeddings_from_csv,
    load_master_dataset,
    align_embeddings_and_labels,
    load_trained_rf_model,
    predict_with_model,
)

# Page configuration
st.set_page_config(
    page_title="Acoustic Reef",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-degraded {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🌊 Acoustic Reef</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-powered stethoscope for the ocean</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        st.markdown("---")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Audio File",
            type=['wav', 'mp3', 'flac'],
            help="Upload a .wav file from your hydrophone recording"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Settings")
        
        # Analysis parameters
        sample_rate = st.selectbox(
            "Target Sample Rate",
            [22050, 44100, 48000],
            index=0,
            help="Sample rate for audio processing"
        )
        
        duration_limit = st.slider(
            "Max Duration (seconds)",
            min_value=5,
            max_value=300,
            value=60,
            help="Maximum duration to analyze"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Acoustic Reef** analyzes underwater soundscapes to assess coral reef health.
        
        **How it works:**
        1. Upload your hydrophone recording
        2. AI analyzes the audio using Google SurfPerch
        3. Get instant reef health assessment
        """)
    
    # Main content area with tabs
    tabs = st.tabs(["Upload & Analyze", "Batch Predictions"])
    with tabs[0]:
        if uploaded_file is not None:
            analyze_audio(uploaded_file, sample_rate, duration_limit)
        else:
            show_landing_page()

    with tabs[1]:
        show_batch_predictions()

def show_landing_page():
    """Display the landing page when no file is uploaded"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">Welcome to Acoustic Reef</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        **Acoustic Reef** is your AI-powered tool for monitoring coral reef health through underwater sound analysis.
        
        ### 🎯 What We Analyze
        - **Reef Health Status**: Healthy vs. Degraded
        - **Anthrophony Detection**: Human-made noise presence
        - **Biodiversity Indicators**: Sound signature analysis
        
        ### 🔬 How It Works
        1. **Upload**: Record underwater audio with a hydrophone
        2. **Process**: Our AI extracts acoustic features using Google SurfPerch
        3. **Classify**: Multi-output model provides health assessment
        4. **Report**: Get detailed reef vital signs report
        
        ### 🌊 Why Sound Matters
        Healthy coral reefs are noisy ecosystems with diverse marine life sounds.
        Degraded reefs fall silent as biodiversity decreases.
        """)
        
        st.markdown("### 📁 Supported Formats")
        st.markdown("- WAV files (recommended)")
        st.markdown("- MP3 files")
        st.markdown("- FLAC files")
        
    with col2:
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. **Get a hydrophone** - Underwater microphone
        2. **Record audio** - 30-60 seconds near coral reef
        3. **Upload here** - Use the sidebar file uploader
        4. **Get results** - Instant health assessment
        """)
        
        st.markdown("### 📊 Sample Results")
        
        # Mock results for demonstration
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Health Status", "Healthy", "85% confidence")
        with col_b:
            st.metric("Anthrophony", "Low", "12% detected")
        
        st.markdown("### 🎧 Audio Quality Tips")
        st.markdown("""
        - Record in calm conditions
        - Avoid boat traffic
        - 30-60 second duration
        - Clear water visibility
        """)

def analyze_audio(uploaded_file, sample_rate, duration_limit):
    """Analyze the uploaded audio file"""
    
    st.markdown('<h2 class="sub-header">🔍 Audio Analysis</h2>', unsafe_allow_html=True)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Basic WAV metadata without external deps
        with contextlib.closing(wave.open(tmp_path, 'rb')) as wf:
            n_channels = wf.getnchannels()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            duration_sec = n_frames / float(sr) if sr else 0.0

        # Display audio info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Duration", f"{duration_sec:.1f}s")
        with col2:
            st.metric("Sample Rate", f"{sr:,} Hz")
        with col3:
            st.metric("Channels", "Mono" if n_channels == 1 else f"{n_channels} ch")
        with col4:
            st.metric("RMS Level", "—")

        # Read PCM samples to numpy for embedding
        with contextlib.closing(wave.open(tmp_path, 'rb')) as wf:
            raw_bytes = wf.readframes(n_frames)
            sample_width = wf.getsampwidth()
            dtype = {1: np.int8, 2: np.int16, 3: np.int32, 4: np.int32}.get(sample_width, np.int16)
            audio_np = np.frombuffer(raw_bytes, dtype=dtype)
            if n_channels > 1:
                audio_np = audio_np.reshape(-1, n_channels).mean(axis=1)
            # Normalize to float32 -1..1
            max_val = np.max(np.abs(audio_np)) or 1
            audio_np = (audio_np.astype(np.float32) / max_val).astype(np.float32)

        # Initialize SurfPerch
        sp_model_path = str(SURFPERCH_SETTINGS["model_path"])  # default location
        surfperch = SurfPerchModel(model_path=sp_model_path)

        # Preprocess and embed
        processed = surfperch.preprocess_audio(audio_np, sr)
        embeddings = surfperch.generate_embeddings(processed, 22050)

        st.success(f"Generated embeddings shape: {embeddings.shape}")

        # Demo: derive mock metrics from embeddings statistics
        emb_mean = float(np.mean(embeddings))
        emb_var = float(np.var(embeddings))

        st.markdown("### 🤖 AI Analysis Results")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🏥 Reef Health Assessment")
            health_score = 0.5 + 0.5 * (np.tanh(emb_mean))
            health_status = "Healthy" if health_score > 0.7 else "Degraded"
            health_color = "status-healthy" if health_status == "Healthy" else "status-degraded"
            st.markdown(f'<p class="{health_color}">Status: {health_status}</p>', unsafe_allow_html=True)
            st.metric("Confidence", f"{health_score:.1%}")
        with col2:
            st.markdown("#### 🔊 Anthrophony Detection")
            anthro_score = min(0.95, max(0.05, emb_var / (emb_var + 1)))
            anthro_status = "High" if anthro_score > 0.3 else "Low"
            anthro_color = "status-degraded" if anthro_status == "High" else "status-healthy"
            st.markdown(f'<p class="{anthro_color}">Human Noise: {anthro_status}</p>', unsafe_allow_html=True)
            st.metric("Detection Level", f"{anthro_score:.1%}")

        # Summary table
        st.markdown("### 📋 Reef Vital Signs Report")
        df = pd.DataFrame({
            'Metric': ['Overall Health', 'Embedding Mean', 'Embedding Variance', 'Human Impact'],
            'Score': [f"{health_score:.1%}", f"{emb_mean:.3f}", f"{emb_var:.3f}", f"{anthro_score:.1%}"],
            'Status': [health_status, '—', '—', anthro_status]
        })
        st.dataframe(df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def show_batch_predictions():
    """Load embeddings/dataset, align, run RF predictions, and display with filters/export."""
    st.markdown('<h2 class="sub-header">📦 Batch Predictions</h2>', unsafe_allow_html=True)

    # Paths
    st.caption(f"Embeddings: {EMBEDDINGS_CSV}")
    st.caption(f"Dataset: {MASTER_DATASET_CSV}")
    st.caption(f"Model: {RF_MODEL_PATH}")

    try:
        with st.spinner("Loading data and model..."):
            X_emb, emb_df = load_embeddings_from_csv()
            dataset_df = load_master_dataset()
            X, y, merged = align_embeddings_and_labels(emb_df, dataset_df)
            model = load_trained_rf_model()
            preds, probs = predict_with_model(model, X)

        # Build results dataframe
        results_df = merged.copy()
        results_df["prediction"] = preds

        # If probabilities available, add max prob
        if isinstance(probs, list):
            try:
                # multi-output: use first head's max prob as preview
                first_head = probs[0]
                if first_head is not None:
                    results_df["prob_max"] = np.max(first_head, axis=1)
            except Exception:
                pass
        elif probs is not None:
            try:
                results_df["prob_max"] = np.max(probs, axis=1)
            except Exception:
                pass

        # Sidebar-like filters in expander
        with st.expander("Filters", expanded=False):
            filter_cols = [c for c in results_df.columns if results_df[c].dtype == object and c != "prediction"]
            selections = {}
            cols = st.columns(min(3, max(1, len(filter_cols)))) if filter_cols else []
            for i, c in enumerate(filter_cols):
                unique_vals = ["(all)"] + sorted([str(v) for v in results_df[c].dropna().unique()])
                with cols[i % max(1, len(cols))]:
                    selections[c] = st.selectbox(f"{c}", unique_vals, index=0)

        # Apply filters
        filtered = results_df
        for c, v in (selections or {}).items():
            if v != "(all)":
                filtered = filtered[filtered[c].astype(str) == v]

        st.markdown("### Results")
        st.dataframe(filtered, use_container_width=True, height=480)

        # Download button
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="batch_predictions.csv", mime="text/csv")

        # Metrics if ground-truth present
        label_candidates = ["health_label", "reef_health", "label"]
        gt_col = next((c for c in label_candidates if c in filtered.columns), None)
        if gt_col is not None:
            from sklearn.metrics import classification_report
            st.markdown("### Quick Metrics")
            try:
                report = classification_report(filtered[gt_col], filtered["prediction"], output_dict=False)
                st.text(report)
            except Exception as e:
                st.info(f"Could not compute metrics: {e}")

        # Small summary
        st.caption(f"Total records: {len(results_df)} | After filters: {len(filtered)} | Feature dim: {X.shape[1] if 'X' in locals() else '—'}")

    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
    except Exception as e:
        st.error(f"Error running batch predictions: {e}")

if __name__ == "__main__":
    main()
