"""
Acoustic Reef - Main Streamlit Dashboard
AI-powered stethoscope for the ocean
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
# import librosa  # Commented out for now due to installation issues
# import matplotlib.pyplot as plt  # Commented out for now due to installation issues
# import seaborn as sns  # Commented out for now due to installation issues
from pathlib import Path
import tempfile
import os
import wave
import contextlib
import hashlib

from src.models.surfperch_integration import SurfPerchModel
from src.utils.config import SURFPERCH_SETTINGS, EMBEDDINGS_CSV, MASTER_DATASET_CSV, RF_MODEL_PATH
from src.inference import (
    resolve_features_for_file,
    predict_vital_signs,
    load_umap_coordinates,
    transform_with_umap,
)
from src.utils.acoustic_map_enhancements import (
    identify_acoustic_clusters,
    create_enhanced_scatter_plot,
    analyze_nearest_neighbors,
    create_trajectory_plot,
    generate_acoustic_insights
)
from src.simple_inference import predict_simple
from src.mock_classifier import predict_with_mock_classifier
from src.force_real_classifier import predict_with_force_classifier
from src.real_model_loader import predict_with_real_model

# Caching functions for performance optimization
@st.cache_data
def load_umap_data_cached():
    """Cached function to load UMAP coordinates"""
    return load_umap_coordinates()

@st.cache_data
def load_cluster_data_cached(base_df):
    """Cached function to identify acoustic clusters"""
    if base_df is not None and not base_df.empty:
        return identify_acoustic_clusters(base_df, method='kmeans', n_clusters=3)
    return None, None

@st.cache_data
def compute_spectrogram_cached(audio_data, sample_rate, duration):
    """Cached function to compute spectrogram"""
    import numpy as _np
    
    win = 2048
    hop = 512
    num_frames = max(1, (len(audio_data) - win) // hop + 1)
    window = _np.hanning(win)
    
    spec = []
    for i in range(num_frames):
        start = i * hop
        frame = audio_data[start:start+win]
        if len(frame) < win:
            frame = _np.pad(frame, (0, win - len(frame)))
        windowed_frame = frame * window
        fft_result = _np.fft.rfft(windowed_frame)
        mag = _np.abs(fft_result)
        spec.append(mag)
    
    spec = _np.array(spec).T
    spec_db = 20 * _np.log10(spec + 1e-10)
    time_axis = _np.linspace(0, duration, spec.shape[1])
    freq_axis = _np.linspace(0, sample_rate/2, spec.shape[0])
    
    return spec_db, time_axis, freq_axis

@st.cache_resource
def load_model_cached():
    """Cached function to load the trained model"""
    try:
        import joblib
        from src.utils.config import RF_MODEL_PATH
        if os.path.exists(RF_MODEL_PATH):
            return joblib.load(RF_MODEL_PATH)
        return None
    except Exception:
        return None

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
        
        # File upload with enhanced guidance
        st.markdown("#### 🎤 Upload Your Recording")
        
        # Pre-upload guidance
        with st.expander("📋 Recording Guidelines", expanded=True):
            st.markdown("""
            **For best results, follow these guidelines:**
            
            **⏱️ Duration**: 5-60 seconds (30-60 seconds ideal)
            **🔊 Quality**: Clear audio with minimal handling noise
            **🌊 Conditions**: Record during typical reef activity (daytime often best)
            **📱 Equipment**: Use a hydrophone or waterproof microphone
            **🌊 Environment**: Calm water, avoid boat traffic during recording
            
            **What to avoid:**
            - Very short recordings (<5 seconds)
            - Excessive background noise or clipping
            - Recording during storms or heavy boat traffic
            - Poor quality microphones or damaged equipment
            """)
        
        uploaded_file = st.file_uploader(
            "Choose Audio File",
            type=['wav', 'mp3', 'flac'],
            help="Upload a hydrophone recording of your reef (WAV format recommended)"
        )

        # Batch upload (multiple files)
        with st.expander("Batch Upload (analyze multiple files)"):
            batch_files = st.file_uploader(
                "Upload multiple audio files",
                type=['wav', 'mp3', 'flac'],
                accept_multiple_files=True,
                help="Drag & drop multiple files to analyze together"
            )
            if batch_files:
                run_batch = st.button("Run Batch Analysis")
            else:
                run_batch = False
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Settings")
        
        # Analysis parameters
        sample_rate = st.selectbox(
            "Target Sample Rate",
            [32000, 44100, 48000],
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
    tabs = st.tabs(["🎤 Single Analysis", "📊 Batch Analysis", "🗺️ Acoustic Map & Diagnostics"])
    with tabs[0]:
        if uploaded_file is not None:
            analyze_audio(uploaded_file, sample_rate, duration_limit)
        else:
            show_landing_page()

    with tabs[1]:
        # Prefer new in-app batch upload if files provided; otherwise show dataset-based batch predictions
        try:
            if 'batch_files' in locals() and batch_files and 'run_batch' in locals() and run_batch:
                run_batch_upload(batch_files)
            else:
                show_batch_predictions()
        except Exception as e:
            st.error(f"Batch analysis failed: {e}")
    with tabs[2]:
        show_acoustic_map()

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
        with st.spinner("Processing audio and generating embeddings..."):
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

            # Enhanced data quality validation with specific feedback
            quality_issues = []
            quality_warnings = []
            quality_info = []
            
            # Duration validation
            if duration_sec < 5:
                quality_issues.append("❌ **Too short**: Recording is less than 5 seconds")
                quality_warnings.append("Results may be unreliable due to insufficient data")
            elif duration_sec < 10:
                quality_warnings.append("⚠️ **Short recording**: Consider recording for 30-60 seconds for better accuracy")
            elif duration_sec > 300:
                quality_warnings.append("⚠️ **Very long**: Only the first 5 minutes will be analyzed")
            elif duration_sec > duration_limit:
                quality_info.append("ℹ️ Only the first segment up to the Max Duration was analyzed")
            
            # Sample rate validation
            if sr < 16000:
                quality_issues.append("❌ **Low sample rate**: Below 16kHz may miss important frequencies")
            elif sr < 22050:
                quality_warnings.append("⚠️ **Low sample rate**: Consider recording at 32kHz or higher")
            
            # Channel validation
            if n_channels > 2:
                quality_warnings.append("⚠️ **Multiple channels**: Only the first channel will be used")
            
            # Display quality feedback
            if quality_issues:
                st.error("**Recording Quality Issues:**")
                for issue in quality_issues:
                    st.markdown(issue)
                st.markdown("**Recommendation**: Please record a new audio file following the guidelines above.")
                return
            
            if quality_warnings:
                st.warning("**Recording Quality Warnings:**")
                for warning in quality_warnings:
                    st.markdown(warning)
            
            if quality_info:
                for info in quality_info:
                    st.info(info)

            # Inline audio playback
            st.audio(tmp_path, format='audio/wav')

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

            # Enhanced spectrogram visualization with modern styling
            try:
                import plotly.express as px
                import plotly.graph_objects as go
                
                # Use cached spectrogram computation
                spec_db, time_axis, freq_axis = compute_spectrogram_cached(audio_np, sr, duration_sec)
                
                # Create enhanced spectrogram with modern styling
                fig_s = go.Figure(data=go.Heatmap(
                    z=spec_db,
                    x=time_axis,
                    y=freq_axis,
                    colorscale='Viridis',
                    colorbar=dict(
                        title=dict(
                            text="Power (dB)",
                            font=dict(size=12, color='#2c3e50')
                        ),
                        tickfont=dict(size=10, color='#2c3e50'),
                        len=0.8,
                        y=0.5,
                        yanchor='middle'
                    ),
                    hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Frequency:</b> %{y:.0f} Hz<br><b>Power:</b> %{z:.1f} dB<extra></extra>',
                    hoverongaps=False
                ))
                
                # Enhanced layout with modern styling
                fig_s.update_layout(
                    title=dict(
                        text="🎵 Audio Spectrogram Analysis",
                        font=dict(size=18, color='#1f77b4', family='Arial, sans-serif'),
                        x=0.5,
                        xanchor='center'
                    ),
                    xaxis=dict(
                        title=dict(
                            text="Time (seconds)",
                            font=dict(size=14, color='#2c3e50')
                        ),
                        tickfont=dict(size=12, color='#2c3e50'),
                        gridcolor='rgba(128,128,128,0.2)',
                        showgrid=True,
                        zeroline=False
                    ),
                    yaxis=dict(
                        title=dict(
                            text="Frequency (Hz)",
                            font=dict(size=14, color='#2c3e50')
                        ),
                        tickfont=dict(size=12, color='#2c3e50'),
                        gridcolor='rgba(128,128,128,0.2)',
                        showgrid=True,
                        zeroline=False
                    ),
                    height=400,
                    margin=dict(l=60, r=20, t=60, b=60),
                    plot_bgcolor='rgba(248,249,250,0.8)',
                    paper_bgcolor='rgba(255,255,255,0.9)',
                    font=dict(family='Arial, sans-serif')
                )
                
                # Add frequency band annotations for marine acoustics
                max_freq = sr/2
                if max_freq >= 2000:
                    # Add frequency band annotations
                    bands = [
                        (0, 20, "Infrasound", "#ff6b6b"),
                        (20, 200, "Low Frequency", "#4ecdc4"),
                        (200, 2000, "Mid Frequency", "#45b7d1"),
                        (2000, max_freq, "High Frequency", "#96ceb4")
                    ]
                    
                    for low, high, label, color in bands:
                        if high <= max_freq:
                            fig_s.add_annotation(
                                x=0.02, y=(low + high) / 2,
                                text=label,
                                showarrow=False,
                                font=dict(size=10, color=color),
                                bgcolor="rgba(255,255,255,0.8)",
                                bordercolor=color,
                                borderwidth=1,
                                xref="paper", yref="y"
                            )
                
                st.markdown("### 🎵 Audio Spectrogram Analysis")
                st.plotly_chart(fig_s, use_container_width=True)
                
                # Add informative text about the spectrogram
                with st.expander("ℹ️ About this spectrogram", expanded=False):
                    st.markdown("""
                    **What you're seeing:**
                    - **X-axis (Time)**: Shows how the audio changes over time
                    - **Y-axis (Frequency)**: Shows different sound frequencies (Hz)
                    - **Color intensity**: Shows sound power (darker = louder)
                    
                    **Marine acoustics context:**
                    - **0-20 Hz**: Infrasound (whale calls, seismic activity)
                    - **20-200 Hz**: Low frequency (fish sounds, boat engines)
                    - **200-2000 Hz**: Mid frequency (coral reef sounds, marine life)
                    - **2000+ Hz**: High frequency (dolphin clicks, anthropogenic noise)
                    
                    **Healthy reef indicators:**
                    - Rich, diverse frequency content
                    - Consistent sound patterns
                    - Natural marine life sounds
                    """)
                    
            except Exception as e:
                st.warning(f"Could not generate spectrogram: {e}")
                # Fallback to simple visualization
                try:
                    import plotly.express as px
                    import numpy as _np
                    # Simple fallback spectrogram
                    win = 1024
                    hop = 512
                    num_frames = max(1, (len(audio_np) - win) // hop + 1)
                    spec = []
                    for i in range(num_frames):
                        start = i * hop
                        frame = audio_np[start:start+win]
                        if len(frame) < win:
                            frame = _np.pad(frame, (0, win - len(frame)))
                        mag = _np.abs(_np.fft.rfft(frame))
                        spec.append(mag)
                    spec = _np.array(spec).T
                    spec_db = 20 * _np.log10(spec + 1e-6)
                    fig_s = px.imshow(spec_db, origin='lower', color_continuous_scale='Viridis')
                    fig_s.update_layout(height=300, title="Basic Spectrogram")
                    st.plotly_chart(fig_s, use_container_width=True)
                except Exception:
                    st.info("Spectrogram visualization not available")

            # Resolve feature vector (precomputed vs runtime)
            file_basename = os.path.basename(getattr(uploaded_file, 'name', ''))
            try:
                feature_vals, source = resolve_features_for_file(file_basename, audio_np, sr)
            except Exception as e:
                st.error("Analysis failed while preparing features. Please try a different audio file.")
                return

            st.success(f"Using {source}; features shape: {feature_vals.shape}")

            # Predict vital signs (health + noise) with simplified fallback system
            result = None
            model_source = "Unknown"
            model_warning = None
            
            # Primary: Try REAL trained model with robust error handling
            try:
                # Check if model file exists before attempting to load
                from src.utils.config import RF_MODEL_PATH
                if os.path.exists(RF_MODEL_PATH):
                    result = predict_with_real_model(feature_vals)
                    model_source = "Real Trained Model"
                    st.success("✅ Using trained AI model - highest accuracy")
                else:
                    raise FileNotFoundError("Trained model file not found")
            except Exception as e_real:
                # Fallback: Mock classifier with clear explanation
                try:
                    result = predict_with_mock_classifier(feature_vals)
                    model_source = "Heuristic Analysis"
                    model_warning = "⚠️ Using heuristic analysis due to missing trained model. Results are estimates based on audio characteristics."
                    st.warning(model_warning)
                except Exception as e_mock:
                    st.error("❌ Analysis failed. Please check your audio file and try again.")
                    st.error(f"Technical details: {str(e_mock)}")
                    return

            # Vital Signs UI with enhanced presentation
            st.markdown("### 🩺 Vital Signs")
            
            # Model source indicator
            if model_warning:
                st.info(f"**Analysis Method**: {model_source}")
            else:
                st.success(f"**Analysis Method**: {model_source}")
            
            # Helper function for confidence interpretation
            def get_confidence_level(conf):
                if conf is None:
                    return "Unknown", "❓"
                elif conf >= 0.8:
                    return "High", "🟢"
                elif conf >= 0.5:
                    return "Medium", "🟡"
                else:
                    return "Low", "🔴"
            
            col1, col2 = st.columns(2)
            with col1:
                color = "status-healthy" if result.health_label == "Healthy" else "status-degraded"
                st.markdown("#### 🏥 Reef Health")
                st.markdown(f'<p class="{color}">{result.health_label}</p>', unsafe_allow_html=True)
                if result.health_conf is not None:
                    conf_level, conf_icon = get_confidence_level(result.health_conf)
                    st.metric("Confidence", f"{result.health_conf:.0%} ({conf_level})", 
                             help=f"{conf_icon} {conf_level} confidence - {'Very reliable' if conf_level == 'High' else 'Moderately reliable' if conf_level == 'Medium' else 'Low reliability, consider retaking recording'}")
            with col2:
                st.markdown("#### 🔊 Noise Pollution")
                color_n = "status-degraded" if result.noise_label == "High" else "status-healthy"
                st.markdown(f'<p class="{color_n}">{result.noise_label}</p>', unsafe_allow_html=True)
                if result.noise_conf is not None:
                    conf_level, conf_icon = get_confidence_level(result.noise_conf)
                    st.metric("Confidence", f"{result.noise_conf:.0%} ({conf_level})", 
                             help=f"{conf_icon} {conf_level} confidence - {'Very reliable' if conf_level == 'High' else 'Moderately reliable' if conf_level == 'Medium' else 'Low reliability, consider retaking recording'}")

            # Annotated Spectrogram based on predictions
            st.markdown("### 🎵 Annotated Spectrogram Analysis")
            try:
                # Use cached spectrogram computation
                import plotly.graph_objects as go
                spec_db, time_axis, freq_axis = compute_spectrogram_cached(audio_np, sr, duration_sec)
                
                # Create annotated spectrogram
                fig_annotated = go.Figure()
                
                # Add the spectrogram
                fig_annotated.add_trace(go.Heatmap(
                    z=spec_db,
                    x=time_axis,
                    y=freq_axis,
                    colorscale='Viridis',
                    colorbar=dict(
                        title=dict(text="Power (dB)", font=dict(size=12)),
                        len=0.8
                    ),
                    hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Frequency:</b> %{y:.0f} Hz<br><b>Power:</b> %{z:.1f} dB<extra></extra>',
                    name="Spectrogram"
                ))
                
                # Add frequency band annotations based on predictions
                annotations = []
                shapes = []
                
                # Define frequency bands
                low_freq_max = 200
                mid_freq_max = 2000
                
                # Add frequency band highlights
                if result.noise_label == "High":
                    # Highlight low frequency bands for boat noise
                    shapes.append(dict(
                        type="rect",
                        x0=0, x1=duration_sec,
                        y0=0, y1=low_freq_max,
                        fillcolor="red",
                        opacity=0.2,
                        line=dict(width=2, color="red"),
                        name="Boat Noise Zone"
                    ))
                    annotations.append(dict(
                        x=duration_sec/2, y=low_freq_max/2,
                        text="🚢 Boat Noise<br>Detected",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="red",
                        font=dict(color="red", size=12),
                        bgcolor="white",
                        bordercolor="red"
                    ))
                
                if result.health_label == "Healthy":
                    # Highlight mid frequency bands for healthy reef sounds
                    shapes.append(dict(
                        type="rect",
                        x0=0, x1=duration_sec,
                        y0=low_freq_max, y1=mid_freq_max,
                        fillcolor="green",
                        opacity=0.2,
                        line=dict(width=2, color="green"),
                        name="Healthy Reef Zone"
                    ))
                    annotations.append(dict(
                        x=duration_sec/2, y=(low_freq_max + mid_freq_max)/2,
                        text="🐠 Healthy Reef<br>Activity",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="green",
                        font=dict(color="green", size=12),
                        bgcolor="white",
                        bordercolor="green"
                    ))
                elif result.health_label == "Degraded":
                    # Highlight high frequency bands for degraded reef
                    shapes.append(dict(
                        type="rect",
                        x0=0, x1=duration_sec,
                        y0=mid_freq_max, y1=sr/2,
                        fillcolor="orange",
                        opacity=0.2,
                        line=dict(width=2, color="orange"),
                        name="Degraded Zone"
                    ))
                    annotations.append(dict(
                        x=duration_sec/2, y=(mid_freq_max + sr/2)/2,
                        text="⚠️ Degraded<br>Reef Sounds",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="orange",
                        font=dict(color="orange", size=12),
                        bgcolor="white",
                        bordercolor="orange"
                    ))
                
                # Update layout with annotations
                fig_annotated.update_layout(
                    title=dict(
                        text="🎵 AI-Annotated Spectrogram",
                        font=dict(size=16, color='#2c3e50'),
                        x=0.5
                    ),
                    xaxis=dict(title="Time (seconds)", showgrid=True),
                    yaxis=dict(title="Frequency (Hz)", showgrid=True),
                    height=500,
                    shapes=shapes,
                    annotations=annotations
                )
                
                st.plotly_chart(fig_annotated, use_container_width=True)
                
                # Add explanation
                with st.expander("🔍 What do the highlighted areas mean?", expanded=False):
                    if result.noise_label == "High":
                        st.markdown("**🔴 Red highlighted area (0-200 Hz)**: Boat engine noise detected in low frequencies")
                    if result.health_label == "Healthy":
                        st.markdown("**🟢 Green highlighted area (200-2000 Hz)**: Rich marine life sounds indicating healthy reef")
                    elif result.health_label == "Degraded":
                        st.markdown("**🟠 Orange highlighted area (2000+ Hz)**: Reduced biodiversity sounds, indicating reef stress")
                    
                    st.markdown("""
                    **Understanding the colors:**
                    - **Darker areas**: Quieter sounds
                    - **Brighter areas**: Louder sounds
                    - **Highlighted zones**: Key frequency bands that influenced the AI's decision
                    """)
                    
            except Exception as e:
                st.warning(f"Could not generate annotated spectrogram: {e}")

            # Simplified Model Analysis - Focus on key insights
            try:
                import joblib
                import plotly.graph_objects as go
                from src.utils.config import RF_MODEL_PATH
                
                # Only show detailed analysis if using real model
                model = load_model_cached()
                if model is not None and hasattr(model, "predict_proba") and hasattr(model, "classes_"):
                    probs = model.predict_proba(feature_vals)[0]
                    cls_to_prob = {str(c): float(p) for c, p in zip(model.classes_, probs)}
                    
                    st.markdown("### 📊 AI Analysis Details")
                    
                    # Simplified single chart focusing on key information
                    class_names = list(cls_to_prob.keys())
                    probabilities = list(cls_to_prob.values())
                    
                    # Define colors for each class
                    color_map = {
                        'healthy': '#2E8B57',      # Sea Green
                        'degraded': '#DC143C',     # Crimson  
                        'anthrophony': '#FF8C00'   # Dark Orange
                    }
                    colors = [color_map.get(cls, '#6A5ACD') for cls in class_names]
                    
                    # Create simple, clear bar chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=class_names,
                            y=probabilities,
                            marker=dict(
                                color=colors,
                                line=dict(width=2, color='white'),
                                opacity=0.8
                            ),
                            text=[f"{p:.1%}" for p in probabilities],
                            textposition='auto',
                            textfont=dict(size=14, color='white', family='Arial Black'),
                            name="AI Confidence"
                        )
                    ])
                    
                    # Update layout for clarity
                    fig.update_layout(
                        title=dict(
                            text="🎯 AI Confidence Breakdown",
                            font=dict(size=16, color='#2c3e50'),
                            x=0.5,
                            xanchor='center'
                        ),
                        height=400,
                        showlegend=False,
                        plot_bgcolor='rgba(248,249,250,0.8)',
                        paper_bgcolor='rgba(255,255,255,0.9)',
                        font=dict(family='Arial, sans-serif', size=12),
                        xaxis=dict(title="Prediction Categories"),
                        yaxis=dict(title="Confidence Level", range=[0, 1])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Key metrics in simple format
                    max_prob = max(probabilities)
                    second_best = sorted(probabilities, reverse=True)[1]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Highest Confidence", f"{max_prob:.1%}", 
                                delta=f"{max_prob - 0.5:.1%}" if max_prob > 0.5 else None)
                    with col2:
                        st.metric("Second Best", f"{second_best:.1%}")
                    with col3:
                        uncertainty = 1 - max_prob
                        st.metric("Uncertainty", f"{uncertainty:.1%}")
                    
                    # Simple interpretation
                    with st.expander("ℹ️ What do these results mean?", expanded=False):
                        st.markdown("""
                        **Understanding the AI analysis:**
                        
                        - **High confidence (>80%)**: Very reliable results
                        - **Medium confidence (50-80%)**: Good results, some uncertainty
                        - **Low confidence (<50%)**: Consider retaking the recording
                        
                        **What each category means:**
                        - **Healthy**: Reef shows good biodiversity and natural sounds
                        - **Degraded**: Reef shows signs of stress or damage  
                        - **Anthrophony**: Human-made noise detected (boats, engines)
                        """)
                    
                    # Download option
                    enhanced_data = {
                        **cls_to_prob,
                        'max_confidence': max_prob,
                        'uncertainty': 1 - max_prob
                    }
                    csv_bytes = pd.DataFrame([enhanced_data]).to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Analysis Results (CSV)", 
                        data=csv_bytes, 
                        file_name="reef_analysis_results.csv", 
                        mime="text/csv"
                    )
                else:
                    st.info("📊 Detailed AI analysis available with trained model")
                    
            except Exception as e:
                st.warning(f"Enhanced probability visualization not available: {e}")
                # Fallback to simple display
                try:
                    import joblib
                    from src.utils.config import RF_MODEL_PATH
                    model = joblib.load(RF_MODEL_PATH)
                    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
                        probs = model.predict_proba(feature_vals)[0]
                        cls_to_prob = {str(c): float(p) for c, p in zip(model.classes_, probs)}
                        st.markdown("#### Class Probabilities")
                        st.json(cls_to_prob)
                except Exception:
                    pass

            # Confidence explanation
            with st.expander("What does confidence mean?"):
                st.write("Confidence reflects the model's estimated probability for the predicted class based on SurfPerch embeddings. Recording length, background noise, and signal quality can affect confidence.")

        # Enhanced Take Action section with context-aware recommendations
        if result.health_label in ("Degraded", "Stressed") or result.noise_label == "High":
            if st.button("🎯 Get Action Recommendations"):
                st.markdown("### 🛟 Take Action")
                
                # Get acoustic map insights if possible
                acoustic_insights = []
                try:
                    # Load UMAP data and get insights using cached functions
                    base_df = load_umap_data_cached()
                    if base_df is not None and not base_df.empty:
                        # Get UMAP coordinates for this recording
                        coord = transform_with_umap(feature_vals)
                        if coord is not None:
                            # Identify clusters using cached function
                            cluster_labels, cluster_info = load_cluster_data_cached(base_df)
                            
                            # Analyze nearest neighbors
                            neighbor_analysis = analyze_nearest_neighbors(base_df, coord, n_neighbors=5)
                            
                            # Generate insights
                            acoustic_insights = generate_acoustic_insights(coord, cluster_labels, cluster_info, neighbor_analysis)
                except Exception as e:
                    st.warning(f"Could not generate acoustic insights: {e}")
                
                # Display acoustic map insights if available
                if acoustic_insights:
                    st.markdown("#### 💡 Insights from the Acoustic Map:")
                    for insight in acoustic_insights:
                        st.markdown(insight)
                    st.markdown("---")
                
                # Context-aware recommendations based on prediction combinations
                st.markdown("#### 🎯 Recommended Actions:")
                
                # Determine the primary concern and urgency
                health_degraded = result.health_label in ("Degraded", "Stressed")
                noise_high = result.noise_label == "High"
                
                if health_degraded and noise_high:
                    # Most critical case: Both health and noise issues
                    st.error("🚨 **CRITICAL SITUATION**: Both reef health and noise pollution detected")
                    st.markdown("""
                    **Immediate Actions (Priority 1):**
                    - **Report vessel activity** to local authorities immediately
                    - **Document noise sources** (boat registration, time, location)
                    - **Establish emergency quiet zones** around the reef
                    - **Contact marine patrol** for enforcement
                    
                    **Follow-up Actions (Priority 2):**
                    - **Water quality testing** for pollution sources
                    - **Habitat assessment** for physical damage
                    - **Community engagement** to reduce local noise
                    - **Long-term monitoring** plan implementation
                    """)
                    
                elif health_degraded and not noise_high:
                    # Health issue without noise - focus on environmental factors
                    st.warning("⚠️ **ENVIRONMENTAL CONCERN**: Reef health issues detected (low noise)")
                    st.markdown("""
                    **Primary Actions:**
                    - **Water quality testing** (nutrients, temperature, pH, turbidity)
                    - **Bleaching assessment** and heat stress monitoring
                    - **Pollution source investigation** (runoff, sewage, agricultural)
                    - **Habitat restoration** planning and implementation
                    
                    **Monitoring Actions:**
                    - **Regular acoustic monitoring** to track recovery
                    - **Water quality baseline** establishment
                    - **Community education** on reef protection
                    - **Scientific collaboration** for detailed assessment
                    """)
                    
                elif not health_degraded and noise_high:
                    # Noise issue without health problems - preventive action
                    st.info("🔊 **NOISE CONCERN**: High noise detected on healthy reef")
                    st.markdown("""
                    **Preventive Actions:**
                    - **Monitor noise impact** on reef ecosystem
                    - **Report excessive vessel traffic** to authorities
                    - **Advocate for speed limits** and quiet zones
                    - **Document noise patterns** and sources
                    
                    **Protection Actions:**
                    - **Community awareness** about noise impacts
                    - **Seasonal restrictions** during sensitive periods
                    - **Alternative routing** for vessels
                    - **Regular monitoring** to prevent health decline
                    """)
                    
                else:
                    # Healthy reef, low noise - maintenance and monitoring
                    st.success("✅ **HEALTHY REEF**: Good health and low noise levels")
                    st.markdown("""
                    **Maintenance Actions:**
                    - **Continue regular monitoring** to maintain health
                    - **Document baseline conditions** for future comparison
                    - **Community education** on reef conservation
                    - **Support local conservation** efforts
                    
                    **Prevention Actions:**
                    - **Monitor for early warning signs** of degradation
                    - **Maintain water quality** standards
                    - **Prevent pollution** and overfishing
                    - **Support marine protected areas**
                    """)
                
                # Additional recommendations based on acoustic insights
                if acoustic_insights:
                    st.markdown("#### 🔍 Additional Insights:")
                    if any("High Noise Zone" in insight for insight in acoustic_insights):
                        st.warning("🚨 **Acoustic Analysis**: Your recording is in a high-noise acoustic zone. Immediate noise reduction measures are needed.")
                    elif any("Degraded Zone" in insight for insight in acoustic_insights):
                        st.warning("⚠️ **Acoustic Analysis**: Your recording falls in a degraded acoustic zone. Focus on pollution reduction and habitat restoration.")
                    elif any("trajectory" in insight.lower() for insight in acoustic_insights):
                        st.info("📈 **Trend Analysis**: Consider monitoring this location over time to track acoustic changes.")

                with st.form("send_report_form"):
                    reporter = st.text_input("Your name (optional)")
                    email = st.text_input("Contact email (optional)")
                    notes = st.text_area("Notes / location details")
                    submitted = st.form_submit_button("Send Report")
                    if submitted:
                        st.success("Report submitted. Thank you for taking action!")
        
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
        from src.models.reef_classifier import load_embeddings_from_csv, load_master_dataset, align_embeddings_and_labels, load_trained_rf_model, predict_with_model
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
        st.dataframe(filtered, width='stretch', height=480)

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


def run_batch_upload(batch_files):
    """Analyze multiple uploaded files and summarize results with UMAP visualization."""
    st.markdown("### 📊 Batch Analysis Results")
    
    # Initialize data collection
    rows = []
    healthy_count = 0
    degraded_count = 0
    error_count = 0
    user_embeddings = []
    user_coordinates = []
    user_labels = []
    user_filenames = []

    # Process each file
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, f in enumerate(batch_files):
        try:
            status_text.text(f"Processing {f.name}... ({i+1}/{len(batch_files)})")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(f.getvalue())
                tmp_path = tmp_file.name
            # Minimal read
            with contextlib.closing(wave.open(tmp_path, 'rb')) as wf:
                n_frames = wf.getnframes()
                sr = wf.getframerate()
                raw_bytes = wf.readframes(n_frames)
                sample_width = wf.getsampwidth()
            dtype = {1: np.int8, 2: np.int16, 3: np.int32, 4: np.int32}.get(sample_width, np.int16)
            audio_np = np.frombuffer(raw_bytes, dtype=dtype)
            # Mono
            try:
                with contextlib.closing(wave.open(tmp_path, 'rb')) as wf:
                    n_channels = wf.getnchannels()
                if n_channels > 1:
                    audio_np = audio_np.reshape(-1, n_channels).mean(axis=1)
            except Exception:
                pass
            max_val = np.max(np.abs(audio_np)) or 1
            audio_np = (audio_np.astype(np.float32) / max_val).astype(np.float32)

            # Resolve features and get embeddings
            feature_vals, _ = resolve_features_for_file(f.name, audio_np, sr)
            
            # Store embeddings for UMAP
            user_embeddings.append(feature_vals.flatten())
            user_filenames.append(f.name)
            
            # Predict
            result = predict_with_real_model(feature_vals)
            status = 'success'
            health = result.health_label
            conf = result.health_conf if result.health_conf is not None else 0.0
            
            # Store prediction results
            user_labels.append(health)
            
            if health == 'Healthy':
                healthy_count += 1
            else:
                degraded_count += 1
            rows.append({
                'filename': f.name,
                'health': health,
                'confidence': f"{conf:.0%}",
                'noise': result.noise_label,
                'status': status,
            })
        except Exception as e:
            error_count += 1
            rows.append({
                'filename': getattr(f, 'name', 'unknown'),
                'health': '—',
                'confidence': '—',
                'noise': '—',
                'status': f"error: {e}",
            })
        finally:
            try:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
        progress_bar.progress((i + 1) / len(batch_files))

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    # Display results table
    df = pd.DataFrame(rows)
    st.markdown("### 📋 Analysis Results")
    st.dataframe(df, use_container_width=True)
    
    # Summary metrics
    total = len(rows)
    if total > 0:
        pct_healthy = healthy_count / total
        pct_degraded = degraded_count / total
        st.markdown("### 📊 Summary Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Files", f"{total}")
        c2.metric("Healthy", f"{healthy_count} ({pct_healthy:.0%})")
        c3.metric("Degraded", f"{degraded_count} ({pct_degraded:.0%})")
        c4.metric("Errors", f"{error_count}")

    # Enhanced UMAP Visualization with Trajectory Analysis
    if len(user_embeddings) > 0:
        st.markdown("### 🗺️ Enhanced Acoustic Map - Your Recordings")
        st.markdown("See how your recordings cluster in acoustic space with trajectory analysis:")
        
        try:
            # Load base UMAP data for context
            base_df = load_umap_coordinates()
            if base_df is not None and not base_df.empty:
                # Identify clusters in base data
                cluster_labels, cluster_info = identify_acoustic_clusters(base_df, method='kmeans', n_clusters=3)
                
            # Transform user embeddings to UMAP coordinates
            from src.inference import transform_with_umap
            
            # Get UMAP coordinates for user data
            user_coords = []
            for embedding in user_embeddings:
                coord = transform_with_umap(embedding.reshape(1, -1))
                if coord is not None:
                    user_coords.append(coord[0])
                else:
                    user_coords.append([0, 0])  # Fallback
            
            # Create user data DataFrame
            user_df = pd.DataFrame({
                'x': [coord[0] for coord in user_coords],
                'y': [coord[1] for coord in user_coords],
                'filename': user_filenames,
                'health_status': user_labels
            })
            
            # Check if filenames contain date information for trajectory analysis
            trajectory_data = []
            has_temporal_data = False
            
            # Try to extract dates from filenames
            import re
            from datetime import datetime
            
            for i, filename in enumerate(user_filenames):
                # Look for common date patterns in filenames
                date_patterns = [
                    r'(\d{8})',  # YYYYMMDD
                    r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
                    r'(\d{2}/\d{2}/\d{4})',  # MM/DD/YYYY
                    r'(\d{4}\d{2}\d{2})',  # YYYYMMDD
                ]
                
                date_found = None
                for pattern in date_patterns:
                    match = re.search(pattern, filename)
                    if match:
                        try:
                            date_str = match.group(1)
                            if len(date_str) == 8:  # YYYYMMDD
                                date_found = datetime.strptime(date_str, '%Y%m%d')
                            elif '-' in date_str:  # YYYY-MM-DD
                                date_found = datetime.strptime(date_str, '%Y-%m-%d')
                            elif '/' in date_str:  # MM/DD/YYYY
                                date_found = datetime.strptime(date_str, '%m/%d/%Y')
                            break
                        except ValueError:
                            continue
                
                if date_found:
                    has_temporal_data = True
                    trajectory_data.append({
                        'x': user_coords[i][0],
                        'y': user_coords[i][1],
                        'filename': filename,
                        'date': date_found,
                        'label': user_labels[i]
                    })
            
            # Create trajectory plot if temporal data is available
            if has_temporal_data and len(trajectory_data) > 1:
                st.markdown("#### 📈 Acoustic Trajectory Analysis")
                st.markdown("Your recordings show temporal progression. The red line shows the acoustic trajectory over time:")
                
                trajectory_fig = create_trajectory_plot(base_df, trajectory_data, cluster_labels, cluster_info)
                st.plotly_chart(trajectory_fig, use_container_width=True)
                
                # Analyze trajectory trends
                trajectory_df = pd.DataFrame(trajectory_data)
                trajectory_df = trajectory_df.sort_values('date')
                
                # Calculate trajectory metrics
                start_point = trajectory_df.iloc[0]
                end_point = trajectory_df.iloc[-1]
                distance_moved = np.sqrt((end_point['x'] - start_point['x'])**2 + (end_point['y'] - start_point['y'])**2)
                
                # Determine trend direction
                x_trend = end_point['x'] - start_point['x']
                y_trend = end_point['y'] - start_point['y']
                
                st.markdown("#### 📊 Trajectory Insights")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Distance Moved", f"{distance_moved:.2f}")
                with col2:
                    st.metric("Time Span", f"{(end_point['date'] - start_point['date']).days} days")
                with col3:
                    if distance_moved > 3:
                        trend_text = "Significant Change"
                        trend_color = "🔴"
                    elif distance_moved > 1:
                        trend_text = "Moderate Change"
                        trend_color = "🟡"
                    else:
                        trend_text = "Stable"
                        trend_color = "🟢"
                    st.metric("Acoustic Stability", f"{trend_color} {trend_text}")
                
                # Health trend analysis
                health_trend = trajectory_df['label'].tolist()
                if len(set(health_trend)) > 1:
                    st.warning("⚠️ **Health Status Changed**: Your recordings show different health classifications over time. This may indicate environmental changes or measurement variations.")
                else:
                    st.success("✅ **Consistent Health Status**: All recordings show the same health classification.")
            
            else:
                # Regular scatter plot without trajectory
                    st.markdown("#### 🎯 Acoustic Distribution")
                    st.markdown("Your recordings plotted against the training data acoustic landscape:")
                    
                    # Create enhanced scatter plot with base data and user points
                    fig = create_enhanced_scatter_plot(base_df, cluster_labels, cluster_info)
                    
                    # Add user points
                    for i, row in user_df.iterrows():
                        fig.add_trace(go.Scatter(
                            x=[row['x']],
                            y=[row['y']],
                            mode='markers',
                marker=dict(
                                symbol="star",
                                size=15,
                                color="red" if row['health_status'] == 'Degraded' else "blue",
                                line=dict(width=2, color="white")
                            ),
                            name=f"Your Recording: {row['filename']}",
                            hovertemplate=f"<b>{row['filename']}</b><br>" +
                                         f"Health: {row['health_status']}<br>" +
                                         "Position: (%{x:.2f}, %{y:.2f})<br>" +
                                         "<extra></extra>"
                        ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show clustering insights
            st.markdown("#### 🔍 Acoustic Clustering Insights")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Clustering Analysis:**")
                if len(user_df) > 1:
                    # Calculate spread
                    x_spread = user_df['x'].max() - user_df['x'].min()
                    y_spread = user_df['y'].max() - user_df['y'].min()
                    st.write(f"• **Acoustic spread**: {x_spread:.2f} × {y_spread:.2f}")
                    
                    # Check for tight clusters
                    if x_spread < 2 and y_spread < 2:
                        st.write("• **Tight cluster**: Your recordings have very similar acoustic signatures")
                    elif x_spread < 5 and y_spread < 5:
                        st.write("• **Moderate spread**: Your recordings show some acoustic diversity")
                    else:
                        st.write("• **Wide spread**: Your recordings show diverse acoustic patterns")
                else:
                    st.write("• Upload more files to see clustering patterns")
            
            with col2:
                st.markdown("**Health Distribution:**")
                health_counts = user_df['health_status'].value_counts()
                for health, count in health_counts.items():
                    percentage = (count / len(user_df)) * 100
                    st.write(f"• **{health}**: {count} files ({percentage:.0f}%)")
            
            # Fallback to simple visualization without base data
            if base_df is None or base_df.empty:
                st.warning("Base UMAP data not available. Showing simplified visualization.")
                
                # Simple scatter plot
                fig = px.scatter(
                    user_df,
                    x='x',
                    y='y',
                    color='health_status',
                    hover_data=['filename'],
                    title="Your Recordings in Acoustic Space"
                )
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not generate enhanced acoustic map: {e}")
            st.info("This feature requires the UMAP model to be available.")

    # Download results
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download batch results (CSV)", data=csv_bytes, file_name="batch_results.csv", mime="text/csv")


def show_acoustic_map():
    """Enhanced Acoustic Map visualization with diagnostic capabilities."""
    st.markdown('<h2 class="sub-header">🗺️ Acoustic Map - Diagnostic Tool</h2>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    **Transform your reef recordings into actionable insights!** This enhanced acoustic map uses advanced clustering 
    and neighbor analysis to provide diagnostic information about reef health patterns and conservation priorities.
    """)
    
    # Help section
    with st.expander("ℹ️ How to use the Acoustic Map", expanded=False):
        st.markdown("""
        **What is an Acoustic Map?**
        - Each point represents a reef recording positioned by its acoustic characteristics
        - Similar-sounding reefs cluster together in acoustic space
        - Colors show health status: 🟢 Healthy, 🔴 Degraded, 🟠 High Noise
        
        **Diagnostic Zones:**
        - The map automatically identifies distinct acoustic zones
        - Each zone has a dominant health pattern
        - Your recording's position reveals its acoustic "neighborhood"
        
        **Neighbor Analysis:**
        - Shows the 5 most similar recordings from our database
        - Helps understand what your recording sounds like compared to known samples
        - Provides context for conservation decisions
        
        **Trajectory Analysis (Batch Uploads):**
        - If you upload multiple files with dates in filenames, see how acoustic patterns change over time
        - Red lines show the progression of acoustic health
        - Helps identify trends and environmental changes
        """)
    
    # Load data
    try:
        base_df = load_umap_coordinates()
        if base_df is None or base_df.empty:
            st.info("🔄 UMAP coordinates not available. Generating them now...")
            st.info("Please run `python generate_umap.py` to create the acoustic map visualization.")
            return
    except Exception as e:
        st.error(f"❌ UMAP coordinates not available: {e}")
        return

    # Identify acoustic clusters
    with st.spinner("🔍 Analyzing acoustic patterns..."):
        cluster_labels, cluster_info = identify_acoustic_clusters(base_df, method='kmeans', n_clusters=3)
    
    # Display cluster information
    st.markdown("### 🌊 Acoustic Diagnostic Zones")
    st.markdown("The map below shows automatically identified acoustic zones based on sound pattern similarity:")
    
    # Create enhanced scatter plot
    fig = create_enhanced_scatter_plot(base_df, cluster_labels, cluster_info)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display cluster statistics
    st.markdown("#### 📊 Zone Analysis")
    col1, col2, col3 = st.columns(3)
    
    for i, (cluster_id, info) in enumerate(cluster_info.items()):
        if cluster_id != -1:  # Skip noise points
            with [col1, col2, col3][i % 3]:
                st.metric(
                    f"Zone {cluster_id + 1}",
                    f"{info['dominant_label'].title()} ({info['dominant_percentage']:.0f}%)",
                    f"{info['size']} recordings"
                )
    
    # Overall statistics
    st.markdown("#### 📈 Overall Dataset Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        healthy_count = len(base_df[base_df['label'] == 'healthy'])
        st.metric("🌿 Healthy Reefs", healthy_count)
    with col2:
        degraded_count = len(base_df[base_df['label'] == 'degraded'])
        st.metric("⚠️ Degraded Reefs", degraded_count)
    with col3:
        anthro_count = len(base_df[base_df['label'] == 'anthrophony'])
        st.metric("🔊 Noisy Areas", anthro_count)

    # Interactive Upload Section
    st.markdown("---")
    st.markdown("### 🎯 Analyze Your Recording")
    st.markdown("Upload an audio file to see where it falls in the acoustic landscape and get diagnostic insights:")
    
    uploaded = st.file_uploader(
        "Upload a reef recording", 
        type=["wav", "mp3", "flac"], 
        key="umap_uploader",
        help="Upload a WAV, MP3, or FLAC file to visualize its position in acoustic space"
    )
    
    if uploaded is None:
        st.info("👆 Upload a file above to see where it appears on the acoustic map and get diagnostic insights")
        return

    # Process uploaded file
    with st.spinner("🔍 Analyzing your recording..."):
        # Quick WAV read (same as analyze)
        import wave, contextlib, tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded.getvalue())
            tmp_path = tmp_file.name
        try:
            with contextlib.closing(wave.open(tmp_path, 'rb')) as wf:
                n_channels = wf.getnchannels()
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                raw_bytes = wf.readframes(n_frames)
                sample_width = wf.getsampwidth()
            dtype = {1: np.int8, 2: np.int16, 3: np.int32, 4: np.int32}.get(sample_width, np.int16)
            audio_np = np.frombuffer(raw_bytes, dtype=dtype)
            if n_channels > 1:
                audio_np = audio_np.reshape(-1, n_channels).mean(axis=1)
            max_val = np.max(np.abs(audio_np)) or 1
            audio_np = (audio_np.astype(np.float32) / max_val).astype(np.float32)

            # Compute/resolve features and transform
            feature_vals, _ = resolve_features_for_file(uploaded.name, audio_np, sr)
            coord = transform_with_umap(feature_vals)
            if coord is None:
                st.warning("⚠️ Could not transform point with UMAP model.")
                return

            # Create enhanced overlay plot with uploaded point
            st.markdown("### 🎯 Your Recording on the Map")
            fig2 = create_enhanced_scatter_plot(base_df, cluster_labels, cluster_info, coord, uploaded.name)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Enhanced nearest neighbor analysis
            st.markdown("### 🔍 Detailed Neighbor Analysis")
            neighbor_analysis = analyze_nearest_neighbors(base_df, coord, n_neighbors=5)
            
            # Display summary
            st.info(f"**{neighbor_analysis['summary']}**")
            
            # Show detailed neighbor information
            st.markdown("#### 📋 Nearest Neighbors Details")
            neighbor_df = neighbor_analysis['neighbors'].copy()
            neighbor_df['Distance'] = neighbor_analysis['distances']
            neighbor_df = neighbor_df[['filename', 'label', 'Distance']]
            neighbor_df.columns = ['Filename', 'Health Status', 'Similarity Distance']
            st.dataframe(neighbor_df, use_container_width=True)
            
            # Generate acoustic insights
            st.markdown("### 💡 Diagnostic Insights")
            insights = generate_acoustic_insights(coord, cluster_labels, cluster_info, neighbor_analysis)
            
            for insight in insights:
                st.markdown(insight)
            
            # Position information
            st.markdown("#### 📍 Position Details")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("X Position", f"{coord[0,0]:.2f}")
            with col2:
                st.metric("Y Position", f"{coord[0,1]:.2f}")
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

if __name__ == "__main__":
    main()
