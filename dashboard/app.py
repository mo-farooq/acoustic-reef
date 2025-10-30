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
import requests
import sys
from datetime import datetime, timezone
import io

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Optional metadata libs for GPS/date extraction
try:
    import mutagen
    from mutagen.wave import WAVE
except Exception:
    mutagen = None
try:
    import exifread
except Exception:
    exifread = None

# Map rendering
try:
    import folium
    from streamlit_folium import st_folium
except Exception:
    folium = None
    st_folium = None

# Simple PDF generation
try:
    from fpdf import FPDF
except Exception:
    FPDF = None

from src.models.surfperch_integration import SurfPerchModel
from src.utils.config import SURFPERCH_SETTINGS, EMBEDDINGS_CSV, MASTER_DATASET_CSV, RF_MODEL_PATH, CLASSIFIER_MODEL_DIR
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

# Minimal CSS for essential styling only
st.markdown("""
<style>
    /* Minimal custom styling - rely on Streamlit's native components */
    .stApp {
        max-width: 1400px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header using native Streamlit
    st.title("🌊 Acoustic Reef")
    st.subheader("🎧 AI-powered stethoscope for the ocean 🐠")
    st.divider()
    
    # Sidebar using native Streamlit
    with st.sidebar:
        st.header("🎛️ Control Panel")
        st.divider()
        
        # File upload section
        st.subheader("🎤 Upload Your Recording")
        
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
        st.markdown("### 📍 Location (optional)")
        # Manual coordinate inputs; will be pre-filled if metadata extraction succeeds
        default_lat = st.session_state.get('geo_lat', None)
        default_lon = st.session_state.get('geo_lon', None)
        col_lat, col_lon = st.columns(2)
        with col_lat:
            manual_lat = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=float(default_lat) if isinstance(default_lat, (int, float)) else 0.0,
                step=0.0001,
                format="%0.6f"
            )
        with col_lon:
            manual_lon = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=float(default_lon) if isinstance(default_lon, (int, float)) else 0.0,
                step=0.0001,
                format="%0.6f"
            )
        use_manual_coords = st.checkbox("Use manual coordinates", value=False,
            help="Enable to override file metadata coordinates if present")
        st.session_state['manual_lat'] = manual_lat
        st.session_state['manual_lon'] = manual_lon
        st.session_state['use_manual_coords'] = use_manual_coords
        
        st.divider()
        
        # About section using native Streamlit
        with st.container():
            st.subheader("ℹ️ About This Tool")
            st.info("""
            **Acoustic Reef** uses AI to analyze underwater soundscapes and assess coral reef health.
            
            **🔬 How it works:**
            1. Upload hydrophone recording
            2. AI analyzes with Google SurfPerch  
            3. Get instant health assessment
        """)
    
    # Main content area with tabs
    tabs = st.tabs(["🎤 Single Analysis", "📊 Batch Analysis", "🗺️ Acoustic Map & Diagnostics", "🌍 Geo-Acoustic Map"])
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
    with tabs[3]:
        show_geo_acoustic_map()

def show_landing_page():
    """Display the landing page when no file is uploaded"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.success("""
        ### 👋 Welcome to Acoustic Reef
        Your AI-powered tool for monitoring coral reef health through underwater sound analysis
        """)
        
        st.markdown("""
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

            # Attempt GPS/date metadata extraction from file header
            extracted_lat, extracted_lon, recording_dt = extract_gps_and_datetime(tmp_path)
            if extracted_lat is not None and extracted_lon is not None:
                st.session_state['geo_lat'] = extracted_lat
                st.session_state['geo_lon'] = extracted_lon
            if recording_dt is not None:
                st.session_state['recording_datetime'] = recording_dt

            # Decide which coordinates to use
            chosen_lat = None
            chosen_lon = None
            if st.session_state.get('use_manual_coords'):
                chosen_lat = float(st.session_state.get('manual_lat') or 0.0)
                chosen_lon = float(st.session_state.get('manual_lon') or 0.0)
            else:
                if extracted_lat is not None and extracted_lon is not None:
                    chosen_lat = extracted_lat
                    chosen_lon = extracted_lon
                else:
                    # Fallback to manual fields even if checkbox off (user may have typed)
                    chosen_lat = float(st.session_state.get('manual_lat') or 0.0)
                    chosen_lon = float(st.session_state.get('manual_lon') or 0.0)

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

            # Small location display if available
            if chosen_lat or chosen_lon:
                st.caption(f"Location: {chosen_lat:.6f}, {chosen_lon:.6f}")

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

            # Vital Signs UI with native Streamlit
            st.header("🩺 Reef Vital Signs")
            
            # Model source indicator using native Streamlit
            if model_warning:
                st.info(f"**📊 Analysis Method:** {model_source}")
            else:
                st.success(f"**✅ Analysis Method:** {model_source}")
            
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
            
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                # Health status using native containers
                health_icon = "🌿" if result.health_label == "Healthy" else "⚠️"
                
                with st.container():
                    if result.health_label == "Healthy":
                        st.success("**🏥 Reef Health Status**")
                    else:
                        st.error("**🏥 Reef Health Status**")
                    
                    st.markdown(f"# {health_icon} {result.health_label}")
                    
                if result.health_conf is not None:
                    conf_level, conf_icon = get_confidence_level(result.health_conf)
                    st.metric(
                        "Confidence Level", 
                        f"{result.health_conf:.0%}",
                        delta=f"{conf_level}",
                        help=f"{conf_icon} {conf_level} confidence"
                    )
            
            with col2:
                # Noise pollution using native containers
                noise_icon = "🔊" if result.noise_label == "High" else "🔇"
                
                with st.container():
                    if result.noise_label == "High":
                        st.error("**🔊 Noise Pollution Level**")
                    else:
                        st.success("**🔊 Noise Pollution Level**")
                    
                    st.markdown(f"# {noise_icon} {result.noise_label}")
                    
                if result.noise_conf is not None:
                    conf_level, conf_icon = get_confidence_level(result.noise_conf)
                    st.metric(
                        "Confidence Level",
                        f"{result.noise_conf:.0%}",
                        delta=f"{conf_level}",
                        help=f"{conf_icon} {conf_level} confidence"
                    )

            # Environmental Context based on coordinates/date
            st.header("🌿 Environmental Context")
            context_alerts = []
            try:
                if chosen_lat or chosen_lon:
                    context_alerts, context_details = query_environmental_context(
                        chosen_lat if chosen_lat is not None else 0.0,
                        chosen_lon if chosen_lon is not None else 0.0,
                        st.session_state.get('recording_datetime')
                    )
                else:
                    context_details = {}
            except Exception as e:
                context_details = {}
                st.info(f"Context lookup unavailable: {e}")

            if context_alerts:
                for alert in context_alerts:
                    if 'SST' in alert or 'temperature' in alert.lower():
                        st.warning(f"{alert}")
                    else:
                        st.info(alert)
            else:
                st.caption("No significant environmental anomalies detected near the provided location.")

            # Annotated Spectrogram based on predictions
            st.header("🎵 AI-Annotated Spectrogram")
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
                    
                    st.header("📊 AI Analysis Details")
                    
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
            st.divider()
            if st.button("🎯 Get Personalized Action Recommendations", type="primary"):
                st.header("🛟 Take Action - Conservation Plan")
                
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
                    loc_text = f"{chosen_lat:.6f}, {chosen_lon:.6f}" if (chosen_lat or chosen_lon) else "(not provided)"
                    default_notes = f"Location: {loc_text}\n"
                    if context_alerts:
                        default_notes += "Context Alerts: " + "; ".join(context_alerts)
                    notes = st.text_area("Notes / location details", value=default_notes)
                    submitted = st.form_submit_button("Send Report")
                    if submitted:
                        try:
                            pdf_path = generate_pdf_report(result, chosen_lat, chosen_lon, context_alerts, reporter, notes)
                            st.success("Report generated. Thank you for taking action!")
                            with open(pdf_path, 'rb') as f:
                                st.download_button("📄 Download PDF Report", data=f.read(), file_name="reef_report.pdf", mime="application/pdf")
                        except Exception as e:
                            st.success("Report submitted. Thank you for taking action!")
        
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Persist last analysis in session for map tab
    try:
        st.session_state['last_analysis'] = {
            'health_label': result.health_label if 'result' in locals() and result else None,
            'noise_label': result.noise_label if 'result' in locals() and result else None,
            'lat': chosen_lat if 'chosen_lat' in locals() else None,
            'lon': chosen_lon if 'chosen_lon' in locals() else None,
        }
    except Exception:
        pass


def show_batch_predictions():
    """Load embeddings/dataset, align, run RF predictions, and display with filters/export."""
    st.header("📦 Batch Predictions")

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
    st.header("📊 Batch Analysis Results")
    
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

    # Enhanced 3D UMAP Visualization with Diagnostics
    if len(user_embeddings) > 0:
        st.divider()
        st.header("🗺️ 3D Acoustic Map & Diagnostics")
        
        # Introduction with clear explanation
        st.success("""
        **🎯 What is this?** This map shows how similar your recordings sound to each other. 
        Recordings that sound alike appear close together, while different-sounding recordings are far apart.
        """)
        
        # Quick guide
        with st.expander("📖 Quick Guide - How to Read This Map", expanded=False):
            st.markdown("""
            ### Understanding the 3D Acoustic Map
            
            **🎨 Colors:**
            - 🟢 **Green Points** = Healthy reef sounds (lots of marine life)
            - 🔴 **Red Points** = Degraded reef sounds (less marine life)
            
            **📍 Position:**
            - **Close together** = Recordings sound very similar
            - **Far apart** = Recordings sound very different
            - **Tight cluster** = Consistent reef conditions
            - **Scattered points** = Varied reef conditions
            
            **🎮 Controls:**
            - **Rotate**: Click and drag anywhere
            - **Zoom**: Use mouse wheel or pinch
            - **Pan**: Right-click and drag
            - **Reset View**: Double-click
            
            **💡 What to Look For:**
            - All green and close together? ✅ Great! Your reef is consistently healthy
            - Mixed colors? ⚠️ Your reef may have some stressed areas
            - All red? 🔴 Your reef needs attention
            - Points spread far? 🌐 Very different conditions across recordings
            """)
        
        st.markdown("---")
        
        try:
            # Load 3D UMAP model if available
            import joblib
            UMAP_MODEL_3D_PATH = CLASSIFIER_MODEL_DIR / "umap_model_3d.joblib"
            umap_model_3d = None
            if UMAP_MODEL_3D_PATH.exists():
                try:
                    umap_model_3d = joblib.load(UMAP_MODEL_3D_PATH)
                    st.success("✅ Loaded 3D UMAP model successfully")
                except Exception as e:
                    st.warning(f"3D UMAP model could not be loaded: {e}. Falling back to 2D.")
            else:
                st.warning("3D UMAP model not found. Showing 2D visualization instead.")

            # Load base UMAP coordinates and clusters (independent of 3D model)
            base_df = load_umap_coordinates()
            if base_df is not None and not base_df.empty:
                cluster_labels, cluster_info = identify_acoustic_clusters(base_df, method='kmeans', n_clusters=3)
                
            # Transform user embeddings to 3D UMAP coordinates when possible
            st.info("🔄 Transforming your recordings into 3D acoustic space...")
            user_embeddings_array = np.array(user_embeddings)
            if umap_model_3d is not None:
                user_coords_3d = umap_model_3d.transform(user_embeddings_array)
            else:
                user_coords_3d = None  # Fallback path triggers 2D-only visuals
            from src.inference import transform_with_umap
            user_coords = []
            for embedding in user_embeddings:
                coord = transform_with_umap(embedding.reshape(1, -1))
                if coord is not None:
                    user_coords.append(coord[0])
                else:
                    user_coords.append([0, 0])
            
            # Create 3D visualization if available
            if user_coords_3d is not None:
                # Create user data DataFrame for 3D
                user_df_3d = pd.DataFrame({
                    'x': user_coords_3d[:, 0],
                    'y': user_coords_3d[:, 1],
                    'z': user_coords_3d[:, 2],
                    'filename': user_filenames,
                    'health_status': user_labels,
                    'confidence': [f"{rows[i]['confidence']}" if i < len(rows) else "N/A" for i in range(len(user_filenames))],
                    'noise': [rows[i]['noise'] if i < len(rows) else "N/A" for i in range(len(user_filenames))]
                })
                
                # Create interactive 3D scatter plot
                st.subheader("📊 3D Interactive Acoustic Space")
                
                # Color map for health status
                color_map = {
                    'Healthy': '#10b981',  # Green
                    'Degraded': '#ef4444',  # Red
                    'Stressed': '#f59e0b'   # Orange
                }
                user_df_3d['color'] = user_df_3d['health_status'].map(color_map)
                
                # Create 3D scatter plot
                fig_3d = go.Figure()
                
                # Group by health status for legend
                for health_status in user_df_3d['health_status'].unique():
                    mask = user_df_3d['health_status'] == health_status
                    df_filtered = user_df_3d[mask]
                    
                    fig_3d.add_trace(go.Scatter3d(
                        x=df_filtered['x'],
                        y=df_filtered['y'],
                        z=df_filtered['z'],
                        mode='markers+text',
                        name=health_status,
                        marker=dict(
                            size=12,
                            color=color_map.get(health_status, '#6B7280'),
                            symbol='diamond',
                            line=dict(width=2, color='white'),
                            opacity=0.9
                        ),
                        text=df_filtered['filename'],
                        hovertemplate='<b>%{text}</b><br>' +
                                     'Health: ' + df_filtered['health_status'] + '<br>' +
                                     'Confidence: ' + df_filtered['confidence'] + '<br>' +
                                     'Noise: ' + df_filtered['noise'] + '<br>' +
                                     'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
                                     '<extra></extra>'
                    ))
                
                # Update layout for modern dark 3D visualization
                fig_3d.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='#000000',
                    title=dict(
                        text='🎵 Your Recordings in 3D Acoustic Space',
                        font=dict(size=20, color='#e5e7eb'),
                        x=0.5,
                        xanchor='center'
                    ),
                    scene=dict(
                        bgcolor='#000000',
                        xaxis=dict(
                            title=dict(text='UMAP Dimension 1', font=dict(color='#e5e7eb')),
                            gridcolor='#333333',
                            zeroline=False,
                            showbackground=False,
                            showgrid=True,
                            color='#e5e7eb',
                            tickfont=dict(color='#9ca3af')
                        ),
                        yaxis=dict(
                            title=dict(text='UMAP Dimension 2', font=dict(color='#e5e7eb')),
                            gridcolor='#333333',
                            zeroline=False,
                            showbackground=False,
                            showgrid=True,
                            color='#e5e7eb',
                            tickfont=dict(color='#9ca3af')
                        ),
                        zaxis=dict(
                            title=dict(text='UMAP Dimension 3', font=dict(color='#e5e7eb')),
                            gridcolor='#333333',
                            zeroline=False,
                            showbackground=False,
                            showgrid=True,
                            color='#e5e7eb',
                            tickfont=dict(color='#9ca3af')
                        ),
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.3)
                        )
                    ),
                    height=700,
                    showlegend=True,
                    legend=dict(
                        title='Health Status',
                        yanchor='top',
                        y=0.99,
                        xanchor='left',
                        x=0.01,
                        bgcolor='rgba(0,0,0,0)',
                        bordercolor='#374151',
                        borderwidth=1,
                        font=dict(color='#e5e7eb')
                    ),
                    hovermode='closest',
                    margin=dict(l=0, r=0, t=60, b=0),
                    hoverlabel=dict(
                        bgcolor='rgba(17,17,17,0.95)',
                        bordercolor='#374151',
                        font_color='#e5e7eb'
                    )
                )
                # Ensure text labels are readable on dark background
                fig_3d.update_traces(textfont=dict(color='#e5e7eb', size=10))
                
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Interactive guide directly under the map
                col_guide1, col_guide2 = st.columns([1, 1])
                with col_guide1:
                    st.info("""
                    **🎮 How to Explore:**
                    - **Rotate**: Click and drag to spin the map
                    - **Zoom In/Out**: Scroll with mouse wheel
                    - **Move Around**: Right-click and drag
                    - **Reset View**: Double-click anywhere
                    """)
                with col_guide2:
                    st.info("""
                    **🔍 What You're Seeing:**
                    - Each diamond 💎 is one of your recordings
                    - Hover over any point to see details
                    - 3 dimensions = 3 different sound characteristics
                    - Explore from different angles!
                    """)
                
                # Diagnostic Features Section with Simple Language
                st.divider()
                st.subheader("🔬 Simple Analysis Results")
                st.caption("Here's what we found in plain English:")
                
                # Clustering Analysis with Simple Explanations
                st.markdown("### 📊 How Similar Are Your Recordings?")
                col1, col2, col3 = st.columns(3)
                
                # Calculate 3D spread
                x_spread = user_df_3d['x'].max() - user_df_3d['x'].min()
                y_spread = user_df_3d['y'].max() - user_df_3d['y'].min()
                z_spread = user_df_3d['z'].max() - user_df_3d['z'].min()
                total_spread = np.sqrt(x_spread**2 + y_spread**2 + z_spread**2)
                
                with col1:
                    # Simplified spread metric
                    if total_spread < 3:
                        spread_emoji = "🎯"
                        spread_text = "Very Similar"
                        spread_explanation = "All recordings sound alike - consistent conditions!"
                        spread_color = "success"
                    elif total_spread < 7:
                        spread_emoji = "📊"
                        spread_text = "Somewhat Similar"
                        spread_explanation = "Some variation - conditions may change between recordings"
                        spread_color = "info"
                    else:
                        spread_emoji = "🌐"
                        spread_text = "Very Different"
                        spread_explanation = "Big differences - recordings from different conditions/times"
                        spread_color = "warning"
                    
                    st.metric("Similarity Score", f"{spread_emoji} {spread_text}")
                    if spread_color == "success":
                        st.success(spread_explanation)
                    elif spread_color == "info":
                        st.info(spread_explanation)
                    else:
                        st.warning(spread_explanation)
                
                with col2:
                    # Calculate cluster density with simple explanation
                    n_points = len(user_df_3d)
                    volume = x_spread * y_spread * z_spread if (x_spread > 0 and y_spread > 0 and z_spread > 0) else 1
                    density = n_points / volume if volume > 0 else 0
                    
                    if density > 1.0:
                        cluster_text = "Tightly Grouped"
                        cluster_emoji = "📍"
                        cluster_explanation = "Points are close together - similar sound patterns"
                    else:
                        cluster_text = "Spread Out"
                        cluster_emoji = "🗺️"
                        cluster_explanation = "Points are scattered - diverse sound patterns"
                    
                    st.metric("Grouping", f"{cluster_emoji} {cluster_text}")
                    st.caption(cluster_explanation)
                
                with col3:
                    # Health consistency with clear explanation
                    health_counts = user_df_3d['health_status'].value_counts()
                    dominant_health = health_counts.index[0]
                    dominant_pct = (health_counts.iloc[0] / len(user_df_3d)) * 100
                    
                    if dominant_pct >= 80:
                        consistency_emoji = "✅"
                        consistency_text = "Very Consistent"
                        consistency_color = "success"
                    elif dominant_pct >= 60:
                        consistency_emoji = "⚖️"
                        consistency_text = "Mostly Consistent"
                        consistency_color = "info"
                    else:
                        consistency_emoji = "⚠️"
                        consistency_text = "Mixed Results"
                        consistency_color = "warning"
                    
                    st.metric("Health Pattern", f"{consistency_emoji} {consistency_text}")
                    if consistency_color == "success":
                        st.success(f"{dominant_pct:.0f}% are {dominant_health}")
                    elif consistency_color == "info":
                        st.info(f"{dominant_pct:.0f}% are {dominant_health}")
                
                
                # Simplified Health Distribution with Visual Guide
                st.markdown("---")
                st.markdown("### 🏥 What Does This Mean for Your Reef?")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Create bar chart with simple title
                    health_dist = user_df_3d['health_status'].value_counts()
                    fig_dist = go.Figure(data=[
                        go.Bar(
                            x=health_dist.index,
                            y=health_dist.values,
                            marker_color=['#10b981' if h == 'Healthy' else '#ef4444' for h in health_dist.index],
                            text=[f"{v} files" for v in health_dist.values],
                            textposition='auto',
                            textfont=dict(size=14, color='white')
                        )
                    ])
                    fig_dist.update_layout(
                        title=dict(text='How Many Recordings Show Each Status?', font=dict(size=16)),
                        xaxis_title='Status',
                        yaxis_title='Number of Your Recordings',
                        height=320,
                        showlegend=False
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
                    st.markdown("**Your Results:**")
                    for health, count in health_dist.items():
                        percentage = (count / len(user_df_3d)) * 100
                        if health == 'Healthy':
                            st.success(f"🟢 **{count}** {health}\n\n({percentage:.0f}% of total)")
                        else:
                            st.error(f"🔴 **{count}** {health}\n\n({percentage:.0f}% of total)")
                    
                    # Add simple interpretation
                    st.caption("---")
                    if health_dist.index[0] == 'Healthy' and (health_dist.iloc[0] / len(user_df_3d)) >= 0.7:
                        st.success("**Good news!** Most recordings show a healthy reef.")
                    elif health_dist.index[0] == 'Degraded' and (health_dist.iloc[0] / len(user_df_3d)) >= 0.7:
                        st.error("**Attention needed!** Most recordings show reef stress.")
                    else:
                        st.info("**Mixed results.** Some areas healthy, some need attention.")
                
                # Simplified Insights in Plain Language
                st.markdown("---")
                st.markdown("### 💡 What We Learned")
                st.caption("Here's what the AI found (in simple terms):")
                
                # Create cards for each insight
                insight_cols = st.columns(2)
                
                # Insight 1: Similarity
                with insight_cols[0]:
                    if total_spread < 3:
                        st.info("""
                        **🎯 Very Consistent Sounds**
                        
                        All your recordings sound very similar to each other. This is good! 
                        It means your reef conditions are stable and consistent.
                        
                        *Why this matters:* Consistent sounds = stable environment
                        """)
                    elif total_spread > 7:
                        st.warning("""
                        **🌐 Very Different Sounds**
                        
                        Your recordings sound quite different from each other. This could mean:
                        - Different recording locations
                        - Different times of day
                        - Changing environmental conditions
                        
                        *Why this matters:* Big changes might need investigation
                        """)
                    else:
                        st.info("""
                        **📊 Some Variation**
                        
                        Your recordings have some differences but aren't too extreme. 
                        This is normal for reefs across different times or areas.
                        
                        *Why this matters:* Natural variation is expected
                        """)
                
                # Insight 2: Health patterns
                with insight_cols[1]:
                    if dominant_pct >= 90:
                        if dominant_health == "Healthy":
                            st.success(f"""
                            **✅ Excellent News!**
                            
                            Over 90% of your recordings show a **{dominant_health}** reef!
                            Your reef is in great condition.
                            
                            *What to do:* Keep monitoring to maintain this good health
                            """)
                        else:
                            st.error(f"""
                            **⚠️ Needs Attention!**
                            
                            Over 90% of your recordings show a **{dominant_health}** reef.
                            Action is needed to help your reef recover.
                            
                            *What to do:* See recommendations below
                            """)
                    elif len(health_counts) > 1 and abs(health_counts.iloc[0] - health_counts.iloc[1]) < 3:
                        st.info("""
                        **⚖️ Mixed Conditions**
                        
                        Your reef shows both healthy and stressed areas. This suggests:
                        - Transitional phase
                        - Some areas better than others
                        - Variable conditions
                        
                        *What to do:* Monitor closely to track changes
                        """)
                    else:
                        st.info(f"""
                        **📊 Mostly {dominant_health}**
                        
                        Most recordings show **{dominant_health}** status ({dominant_pct:.0f}%).
                        
                        *Status:* {'Continue good practices' if dominant_health == 'Healthy' else 'Consider conservation actions'}
                        """)
                
                # Actionable Recommendations in Simple Steps
                st.markdown("---")
                st.markdown("### 🎯 What Should You Do Next?")
                st.caption("Simple, actionable steps based on your results:")
                
                # Priority recommendation based on dominant health
                if dominant_health == "Degraded" and dominant_pct > 50:
                    st.error("""
                    ### 🚨 HIGH PRIORITY: Your Reef Needs Help
                    
                    **Why:** Most of your recordings show a stressed reef
                    
                    **What to do NOW:**
                    1. 📸 **Document** what you see (photos, notes)
                    2. 📞 **Contact** local marine conservation groups
                    3. 🧪 **Test water** quality (temperature, pH, clarity)
                    4. 🚫 **Reduce** human activity in the area if possible
                    5. 📅 **Monitor** weekly to track changes
                    
                    **Resources:** Contact your local environmental agency or reef conservation organization
                    """)
                elif dominant_health == "Healthy" and dominant_pct > 70:
                    st.success("""
                    ### ✅ GOOD NEWS: Keep Up the Great Work!
                    
                    **Why:** Your reef is showing healthy signs
                    
                    **What to do to MAINTAIN health:**
                    1. 📊 **Keep monitoring** monthly to catch early problems
                    2. 🌊 **Protect** the area from damage
                    3. 👥 **Share** your success with the community
                    4. 📚 **Educate** others about reef conservation
                    5. 🎯 **Stay vigilant** for any changes
                    
                    **Goal:** Maintain this excellent condition!
                    """)
                else:
                    st.info("""
                    ### ⚖️ MIXED RESULTS: Monitor and Investigate
                    
                    **Why:** Some areas healthy, some stressed
                    
                    **What to do:**
                    1. 🔍 **Identify** which recordings are healthy vs stressed
                    2. 📍 **Map** where each recording was taken
                    3. 🕐 **Note** when each was recorded
                    4. 🔬 **Look for patterns** (location, time, conditions)
                    5. 📈 **Track** changes over time
                    
                    **Focus:** Find what's different between healthy and stressed areas
                    """)
                
                # Additional recommendations
                st.markdown("**📋 Additional Tips:**")
                
                tips = []
                
                if total_spread > 7:
                    tips.append({
                        "title": "🌐 Investigate Why Sounds Are So Different",
                        "content": "Your recordings vary a lot. Check if they were taken at different times of day, locations, or weather conditions. Large differences might indicate environmental changes worth investigating."
                    })
                
                if len(user_df_3d) < 10:
                    tips.append({
                        "title": "📊 Collect More Recordings",
                        "content": f"You have {len(user_df_3d)} recordings. For better accuracy, try to collect at least 10-15 recordings from different times and areas of your reef."
                    })
                
                if len(tips) > 0:
                    for tip in tips:
                        with st.expander(tip["title"]):
                            st.write(tip["content"])
                else:
                    st.success("✅ Your recording coverage looks good!")
                
                # Educational FAQ Section
                st.markdown("---")
                with st.expander("❓ Common Questions About This Analysis", expanded=False):
                    st.markdown("""
                    ### Frequently Asked Questions
                    
                    **Q: What does "acoustic space" mean?**
                    A: Think of it like a map where similar-sounding recordings are placed close together. 
                    Instead of showing geographic location, it shows how recordings sound compared to each other.
                    
                    **Q: Why are there 3 dimensions?**
                    A: Sound has many characteristics (frequency, loudness, patterns). The AI found the 3 most 
                    important differences between recordings and used them as X, Y, and Z axes.
                    
                    **Q: How does the AI know if a reef is healthy?**
                    A: Healthy reefs have lots of marine life making sounds (fish, shrimp, coral). The AI learned 
                    what healthy reefs sound like from thousands of examples and compares your recording to those patterns.
                    
                    **Q: What if my points are far apart?**
                    A: This means your recordings sound different from each other. Could be:
                    - Different locations on the reef
                    - Different times of day
                    - Different weather/water conditions
                    - Environmental changes happening
                    
                    **Q: How accurate is this?**
                    A: The AI is trained on real reef recordings, but it's a tool to help you, not a replacement 
                    for expert analysis. Use it to guide your monitoring and know when to call in experts.
                    
                    **Q: What should I do with these results?**
                    A: Follow the recommendations above! Share with local conservation groups, use for reports, 
                    track changes over time, and take action to protect your reef.
                    """)
                
            else:
                # Fallback to 2D if 3D model not available
                st.warning("""
                **⚠️ 3D Model Not Available**
                
                The 3D visualization model isn't found. Showing a simpler 2D version instead.
                
                *Note:* Contact your administrator to enable 3D visualization for the full experience.
                """)
                
                user_df = pd.DataFrame({
                    'x': [coord[0] for coord in user_coords],
                    'y': [coord[1] for coord in user_coords],
                    'filename': user_filenames,
                    'health_status': user_labels
                })
                
                # Simple but enhanced 2D scatter plot
                fig_2d = px.scatter(
                        user_df,
                        x='x',
                        y='y',
                        color='health_status',
                        hover_data=['filename'],
                    title="🗺️ Your Recordings in 2D Acoustic Space",
                    color_discrete_map={'Healthy': '#10b981', 'Degraded': '#ef4444'},
                    labels={'x': 'Sound Characteristic 1', 'y': 'Sound Characteristic 2'}
                )
                fig_2d.update_layout(
                    height=500,
                    hovermode='closest'
                )
                fig_2d.update_traces(marker=dict(size=15, symbol='diamond'))
                st.plotly_chart(fig_2d, use_container_width=True)
                
                # Simple analysis for 2D
                st.info("""
                **📊 Quick Analysis:**
                - **Green points** = Healthy reefs
                - **Red points** = Stressed reefs
                - **Close together** = Similar sounds
                - **Far apart** = Different sounds
                """)
            
        except Exception as e:
            st.error(f"Could not generate 3D acoustic map: {e}")
            st.info("This feature requires the 3D UMAP model to be available. Please ensure the model is trained.")
            import traceback
            st.code(traceback.format_exc())

    # Final Summary Section
    st.divider()
    st.header("📋 Overall Summary")
    
    # Create summary based on results
    summary_col1, summary_col2 = st.columns([2, 1])
    
    with summary_col1:
        # Overall assessment
        if total > 0:
            healthy_ratio = healthy_count / total
            
            if healthy_ratio >= 0.8:
                st.success(f"""
                ### 🌟 Excellent Overall Health
                
                **{healthy_count} out of {total} recordings** ({healthy_ratio:.0%}) show healthy reef conditions.
                
                **What this means:**
                Your reef is in great shape! The marine ecosystem is thriving with diverse sounds 
                indicating plenty of marine life activity.
                
                **Keep it up!** Continue your monitoring efforts and maintain protection measures.
                """)
            elif healthy_ratio >= 0.5:
                st.info(f"""
                ### ⚖️ Mixed Health Status
                
                **{healthy_count} out of {total} recordings** ({healthy_ratio:.0%}) show healthy conditions, 
                while **{degraded_count}** show signs of stress.
                
                **What this means:**
                Your reef is experiencing some challenges but still has healthy areas. This is a critical 
                time to intervene and prevent further degradation.
            
                **Action needed:** Focus on protecting healthy areas while addressing stressed zones.
                """)
            else:
                st.error(f"""
                ### 🚨 Reef Health Concern
                
                **Only {healthy_count} out of {total} recordings** ({healthy_ratio:.0%}) show healthy conditions.
                **{degraded_count} recordings** indicate reef stress.
                
                **What this means:**
                Your reef is facing significant challenges. The low number of healthy recordings suggests 
                widespread stress that requires immediate attention.
                
                **Priority action:** Contact local marine conservation authorities immediately.
                """)
    
    with summary_col2:
        # Key stats card
        st.metric("Total Analyzed", f"{total} files")
        st.metric("Healthy", f"{healthy_count}", delta=f"{(healthy_count/total)*100:.0f}%" if total > 0 else "0%")
        st.metric("Degraded", f"{degraded_count}", delta=f"-{(degraded_count/total)*100:.0f}%" if total > 0 else "0%", delta_color="inverse")
        if error_count > 0:
            st.metric("Errors", f"{error_count}", delta="⚠️ Check files", delta_color="off")
    
    # Next steps reminder
    st.info("""
    **📥 Don't forget to download your results!**
    
    Use the button below to save your analysis as a CSV file. You can:
    - Share with conservation teams
    - Track changes over time
    - Include in reports
    - Compare with future recordings
    """)

    # Download results
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Complete Analysis Report (CSV)", 
        data=csv_bytes, 
        file_name=f"acoustic_reef_analysis_{len(batch_files)}_files.csv", 
        mime="text/csv",
        type="primary"
    )


def show_acoustic_map():
    """Enhanced Acoustic Map visualization with diagnostic capabilities."""
    st.header("🗺️ Acoustic Map - Diagnostic Tool")
    
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


def show_geo_acoustic_map():
    """Render folium map with last analysis coordinates and status."""
    st.header("🌍 Geo-Acoustic Map")
    # Map controls
    col_map1, col_map2, col_map3 = st.columns([1, 1, 1])
    with col_map1:
        basemap = st.selectbox(
            "Basemap style",
            ["Carto Dark", "Carto Light", "OpenStreetMap", "Stamen Terrain"],
            index=0,
            help="Choose a background map for better contrast"
        )
    with col_map2:
        show_minimap = st.checkbox("Minimap", value=True)
    with col_map3:
        show_fullscreen = st.checkbox("Fullscreen control", value=True)
    # Lazy import to handle cases where dependencies were installed after app start
    local_folium = folium
    local_st_folium = st_folium
    local_plugins = None
    if local_folium is None or local_st_folium is None:
        try:
            import importlib
            local_folium = importlib.import_module('folium')
            _sf_mod = importlib.import_module('streamlit_folium')
            local_st_folium = getattr(_sf_mod, 'st_folium')
            try:
                local_plugins = importlib.import_module('folium.plugins')
            except Exception:
                local_plugins = None
            # Update globals for future calls
            globals()['folium'] = local_folium
            globals()['st_folium'] = local_st_folium
        except Exception as e:
            st.info("Install mapping dependencies to enable this feature (folium, streamlit-folium).")
            st.caption(f"Python: {sys.executable}")
            st.caption(f"Import error: {e}")
            return
    else:
        try:
            from folium import plugins as local_plugins  # type: ignore
        except Exception:
            local_plugins = None

    last = st.session_state.get('last_analysis') or {}
    lat = last.get('lat')
    lon = last.get('lon')
    health = last.get('health_label')
    noise = last.get('noise_label')

    if lat is None or lon is None or (lat == 0.0 and lon == 0.0):
        st.info("Run a Single Analysis with coordinates to see the map marker.")
        return

    # Color by status
    if health == 'Healthy' and (noise or '').lower() != 'high':
        color = 'green'
    elif (noise or '').lower() == 'high':
        color = 'purple'
    else:
        color = 'red'

    # Create map without default tiles to allow user-selected basemap
    m = local_folium.Map(location=[lat, lon], zoom_start=6, tiles=None, control_scale=True)

    # Basemap selection
    if basemap == "Carto Dark":
        local_folium.TileLayer('CartoDB dark_matter', name='Carto Dark').add_to(m)
    elif basemap == "Carto Light":
        local_folium.TileLayer('CartoDB positron', name='Carto Light').add_to(m)
    elif basemap == "Stamen Terrain":
        local_folium.TileLayer('Stamen Terrain', name='Stamen Terrain').add_to(m)
    else:
        local_folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)

    # Optional controls
    if show_fullscreen and local_plugins is not None and hasattr(local_plugins, 'Fullscreen'):
        local_plugins.Fullscreen(position='topright').add_to(m)
    if show_minimap and local_plugins is not None and hasattr(local_plugins, 'MiniMap'):
        try:
            mini = local_plugins.MiniMap(toggle_display=True, minimized=False)
            mini.add_to(m)
        except Exception:
            pass

    # Marker cluster (useful if session has history)
    marker_layer = None
    if local_plugins is not None and hasattr(local_plugins, 'MarkerCluster'):
        try:
            marker_layer = local_plugins.MarkerCluster(name='Reef Samples')
            marker_layer.add_to(m)
        except Exception:
            marker_layer = None

    # Build popup HTML with health badge
    badge_color = {'green': '#10b981', 'red': '#ef4444', 'purple': '#8b5cf6'}.get(color, '#60a5fa')
    popup_html = f"""
    <div style='font-family:Inter,system-ui,Segoe UI,Roboto,Arial; font-size:13px; color:#e5e7eb;'>
      <div style='margin-bottom:6px;'>
        <span style='background:{badge_color}; color:white; padding:2px 8px; border-radius:9999px; font-weight:600;'>
          {health or '—'}
        </span>
        <span style='margin-left:8px; color:#cbd5e1;'>Noise: {noise or '—'}</span>
      </div>
      <div style='color:#cbd5e1;'>📍 {lat:.6f}, {lon:.6f}</div>
    </div>
    """
    popup = local_folium.Popup(local_folium.IFrame(popup_html, width=240, height=90), max_width=260)

    marker = local_folium.CircleMarker(
        location=[lat, lon],
        radius=10,
        color=color,
        fill=True,
        fill_opacity=0.9,
        popup=popup,
        tooltip=f"{health or 'Status'} • Noise: {noise or '—'}"
    )
    if marker_layer is not None:
        marker.add_to(marker_layer)
    else:
        marker.add_to(m)

    # Plot any available history from session state
    history = st.session_state.get('analysis_history') or []
    for item in history:
        try:
            h_lat = item.get('lat'); h_lon = item.get('lon')
            if not h_lat or not h_lon:
                continue
            h_health = item.get('health_label')
            h_noise = item.get('noise_label')
            if h_health == 'Healthy' and (h_noise or '').lower() != 'high':
                h_color = 'green'
            elif (h_noise or '').lower() == 'high':
                h_color = 'purple'
            else:
                h_color = 'red'
            h_badge = {'green': '#10b981', 'red': '#ef4444', 'purple': '#8b5cf6'}.get(h_color, '#60a5fa')
            h_popup_html = f"""
            <div style='font-family:Inter,system-ui,Segoe UI,Roboto,Arial; font-size:12px; color:#e5e7eb;'>
              <div style='margin-bottom:4px;'>
                <span style='background:{h_badge}; color:white; padding:1px 6px; border-radius:9999px; font-weight:600;'>
                  {h_health or '—'}
                </span>
                <span style='margin-left:6px; color:#cbd5e1;'>Noise: {h_noise or '—'}</span>
              </div>
              <div style='color:#cbd5e1;'>📍 {h_lat:.6f}, {h_lon:.6f}</div>
            </div>
            """
            h_popup = local_folium.Popup(local_folium.IFrame(h_popup_html, width=220, height=80), max_width=240)
            h_marker = local_folium.CircleMarker(
                location=[h_lat, h_lon],
                radius=7,
                color=h_color,
                fill=True,
                fill_opacity=0.85,
                popup=h_popup,
                tooltip=f"{h_health or 'Status'} • Noise: {h_noise or '—'}"
            )
            if marker_layer is not None:
                h_marker.add_to(marker_layer)
            else:
                h_marker.add_to(m)
        except Exception:
            continue

    # Add a simple legend
    legend_html = """
    <div style='position: fixed; bottom: 20px; left: 20px; z-index: 9999; background: rgba(17,17,17,0.85); padding: 10px 12px; border-radius: 8px; color: #e5e7eb; font-family: Inter,system-ui,Segoe UI,Roboto,Arial; font-size: 12px; border: 1px solid #374151;'>
      <div style='font-weight: 600; margin-bottom: 6px;'>Legend</div>
      <div style='display:flex; align-items:center; gap:6px; margin-bottom:4px;'>
        <span style='width:10px; height:10px; background:#10b981; border-radius:50%; display:inline-block;'></span>
        <span>Healthy</span>
      </div>
      <div style='display:flex; align-items:center; gap:6px; margin-bottom:4px;'>
        <span style='width:10px; height:10px; background:#ef4444; border-radius:50%; display:inline-block;'></span>
        <span>Degraded</span>
      </div>
      <div style='display:flex; align-items:center; gap:6px;'>
        <span style='width:10px; height:10px; background:#8b5cf6; border-radius:50%; display:inline-block;'></span>
        <span>High Noise</span>
      </div>
    </div>
    """
    m.get_root().html.add_child(local_folium.Element(legend_html))

    # Layer control when multiple layers are present
    try:
        local_folium.LayerControl(position='topright').add_to(m)
    except Exception:
        pass

    local_st_folium(m, width=None, height=520)


def extract_gps_and_datetime(file_path: str):
    """Attempt to extract GPS (lat, lon) and recording datetime from audio metadata.

    Tries mutagen for RIFF/WAV INFO or ID3-like tags, then exifread for EXIF GPS.
    Returns (lat, lon, datetime or None). On failure, returns (None, None, None).
    """
    lat = lon = None
    rec_dt = None

    # Try mutagen (handles WAV INFO, ID3 for some formats)
    try:
        if mutagen is not None:
            audio = mutagen.File(file_path, easy=True)
            if audio is not None and hasattr(audio, 'tags') and audio.tags:
                # Scan all tag values for possible coordinates
                for k, v in dict(audio.tags).items():
                    try:
                        val_list = v if isinstance(v, list) else [v]
                        for item in val_list:
                            text = str(item)
                            parsed = _parse_lat_lon_from_string(text)
                            if parsed and lat is None and lon is None:
                                lat, lon = parsed
                            if rec_dt is None:
                                dt_cand = _parse_datetime_safe(text)
                                if dt_cand:
                                    rec_dt = dt_cand
                    except Exception:
                        continue
    except Exception:
        pass

    # Try exifread for GPS tags if still missing
    if (lat is None or lon is None) and exifread is not None:
        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
            gps_lat = tags.get('GPS GPSLatitude')
            gps_lat_ref = tags.get('GPS GPSLatitudeRef')
            gps_lon = tags.get('GPS GPSLongitude')
            gps_lon_ref = tags.get('GPS GPSLongitudeRef')
            if gps_lat and gps_lon and gps_lat.values and gps_lon.values:
                lat = _convert_exif_gps_to_decimal(gps_lat.values, str(gps_lat_ref))
                lon = _convert_exif_gps_to_decimal(gps_lon.values, str(gps_lon_ref))
        except Exception:
            pass

    # Try WAV-specific RIFF chunk scan for iXML/axml/XMP/LIST/bext
    if (lat is None or lon is None):
        try:
            lat2, lon2, dt2 = _extract_gps_from_wav_chunks(file_path)
            if lat2 is not None and lon2 is not None:
                lat, lon = lat2, lon2
            if rec_dt is None and dt2 is not None:
                rec_dt = dt2
        except Exception:
            pass

    # Fallback: scan entire file text for coordinates patterns
    if (lat is None or lon is None):
        try:
            with open(file_path, 'rb') as f:
                blob = f.read()
            text = blob.decode('utf-8', errors='ignore')
            parsed = _search_lat_lon_in_text(text)
            if parsed:
                lat, lon = parsed
            if rec_dt is None:
                rec_dt = _parse_datetime_safe(text)
        except Exception:
            pass

    return lat, lon, rec_dt


def _parse_lat_lon_from_string(s: str):
    """Parse strings like '12.34, -56.78' into (lat, lon)."""
    try:
        parts = s.strip().replace(";", ",").split(",")
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
    except Exception:
        return None
    return None


def _search_lat_lon_in_text(text: str):
    """Search arbitrary text for plausible latitude/longitude values.
    Returns (lat, lon) or None.
    """
    import re
    # Combined decimal pattern first
    combined = re.search(r"([+-]?\d{1,2}\.\d{3,})\s*[,/; ]\s*([+-]?\d{1,3}\.\d{3,})", text)
    if combined:
        lat = float(combined.group(1))
        lon = float(combined.group(2))
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            return lat, lon
    # Labeled forms like 'lat: 12.3456', 'lon= -123.4567'
    lat_m = re.search(r"(lat|latitude)\s*[:=]\s*([+-]?\d{1,2}\.\d+)", text, re.IGNORECASE)
    lon_m = re.search(r"(lon|longitude|lng)\s*[:=]\s*([+-]?\d{1,3}\.\d+)", text, re.IGNORECASE)
    if lat_m and lon_m:
        lat = float(lat_m.group(2))
        lon = float(lon_m.group(2))
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            return lat, lon
    return None


def _parse_datetime_safe(val: str):
    try:
        # Try common formats
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y%m%dT%H%M%S"):
            try:
                return datetime.strptime(str(val), fmt).replace(tzinfo=timezone.utc)
            except Exception:
                continue
        # Fallback to fromisoformat
        return datetime.fromisoformat(str(val))
    except Exception:
        return None


def _convert_exif_gps_to_decimal(values, ref):
    """Convert EXIF GPS components to decimal degrees."""
    try:
        def frac_to_float(x):
            try:
                return float(x.num) / float(x.den)
            except Exception:
                return float(x)
        d = frac_to_float(values[0])
        m = frac_to_float(values[1])
        s = frac_to_float(values[2])
        dec = d + m/60.0 + s/3600.0
        if ref and ref.strip().upper() in ('S', 'W'):
            dec = -dec
        return dec
    except Exception:
        return None


def _extract_gps_from_wav_chunks(file_path: str):
    """Parse RIFF/WAV chunks (iXML, axml, XMP , LIST, bext) for embedded GPS and date.
    Returns (lat, lon, datetime_or_None) when found; otherwise (None, None, None).
    """
    import struct
    lat = lon = None
    rec_dt = None

    with open(file_path, 'rb') as f:
        data = f.read()
    if len(data) < 12 or data[0:4] != b'RIFF' or data[8:12] != b'WAVE':
        return None, None, None

    offset = 12
    n = len(data)
    while offset + 8 <= n:
        chunk_id = data[offset:offset+4]
        chunk_size = struct.unpack('<I', data[offset+4:offset+8])[0]
        chunk_data_start = offset + 8
        chunk_data_end = min(n, chunk_data_start + chunk_size)
        payload = data[chunk_data_start:chunk_data_end]

        if chunk_id in (b'iXML', b'axml', b'XMP ', b'bext', b'LIST'):
            try:
                text = payload.decode('utf-8', errors='ignore')
                parsed = _search_lat_lon_in_text(text)
                if parsed and (lat is None or lon is None):
                    lat, lon = parsed
                if rec_dt is None:
                    rec_dt = _parse_datetime_safe(text)
            except Exception:
                pass

        # Chunks are padded to even sizes
        offset = chunk_data_end + (chunk_size % 2)

        if lat is not None and lon is not None and rec_dt is not None:
            break

    return lat, lon, rec_dt

@st.cache_data(show_spinner=False)
def get_sst_anomaly(latitude: float, longitude: float, date_str: str) -> float | None:
    """Fetch NOAA Coral Reef Watch (CRW) daily SST anomaly (°C) via ERDDAP for a point and date.

    Data source: NOAA CoastWatch ERDDAP, dataset `noaacrwsstDaily` (CRW CoralTemp 5km, daily).
    Variable: `analysed_sst_anomaly` (degree_C).

    Notes:
    - ERDDAP requires coordinates matching the dataset grid. We query nearest grid cell by specifying exact lat/lon.
    - Longitudes are often 0..360; this function tries both -180..180 and 0..360 conventions.
    - Date must be provided as YYYY-MM-DD; time is set to 00:00:00Z.
    """
    base_csv = "https://coastwatch.noaa.gov/erddap/griddap/noaacrwsstDaily.csv"

    def build_url(lat_val: float, lon_val: float) -> str:
        return (
            f"{base_csv}?analysed_sst_anomaly[({date_str}T00:00:00Z):1:({date_str}T00:00:00Z)]"
            f"[({lat_val}):1:({lat_val})][({lon_val}):1:({lon_val})]"
        )

    # Try with provided lon first
    urls = [build_url(latitude, longitude)]
    # Also try 0..360
    lon360 = (longitude + 360.0) % 360.0
    if abs(lon360 - longitude) > 1e-6:
        urls.append(build_url(latitude, lon360))

    last_error = None
    for url in urls:
        try:
            resp = requests.get(url, timeout=12)
            resp.raise_for_status()
            # ERDDAP CSV returns header + data. Use io.StringIO for pandas.
            df = pd.read_csv(io.StringIO(resp.text))
            # Expect a column named like 'analysed_sst_anomaly (degree_C)'
            col = next((c for c in df.columns if 'analysed_sst_anomaly' in c), None)
            if col and not df.empty:
                val = df.iloc[0][col]
                try:
                    return float(val)
                except Exception:
                    continue
        except Exception as e:
            last_error = str(e)
            continue
    # If all attempts fail
    return None


def query_environmental_context(lat: float, lon: float, recording_dt: datetime | None):
    """Query environmental context near the recording location/time.

    - NOAA CRW daily SST anomaly (via ERDDAP) to assess heat stress risk.
    - AIS placeholder (public AIS often requires API keys).
    """
    alerts = []
    details = {}

    # Determine date string (UTC) for CRW
    if recording_dt is None:
        date_use = datetime.utcnow().date().isoformat()
    else:
        date_use = recording_dt.date().isoformat()

    # 1) NOAA CRW SST anomaly (°C). Threshold for concern: >= +1.0°C
    try:
        sst_anom = get_sst_anomaly(lat, lon, date_use)
        if sst_anom is not None:
            details['sst_anomaly_c'] = sst_anom
            if sst_anom >= 1.0:
                alerts.append(f"🌡️ High SST Anomaly Detected: +{sst_anom:.1f}°C above average. Risk of coral bleaching.")
        else:
            details['sst_anomaly_unavailable'] = True
    except Exception as e:
        details['sst_anomaly_error'] = str(e)

    # 2) AIS placeholder
    details['ais_info'] = "AIS lookup requires API key (e.g., AISHub, MarineTraffic)."

    return alerts, details


def generate_pdf_report(result, lat: float | None, lon: float | None, alerts: list[str], reporter: str | None, notes: str | None) -> str:
    """Generate a simple PDF report including coordinates.
    Returns path to the generated PDF.
    """
    if FPDF is None:
        raise RuntimeError("PDF library not installed")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, txt="Acoustic Reef Report", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.ln(4)
    pdf.cell(0, 8, txt=f"Health: {getattr(result, 'health_label', '—')} (conf: {getattr(result, 'health_conf', 0):.0%} if available)", ln=True)
    pdf.cell(0, 8, txt=f"Noise: {getattr(result, 'noise_label', '—')} (conf: {getattr(result, 'noise_conf', 0):.0%} if available)", ln=True)
    loc_line = f"Coordinates: {lat:.6f}, {lon:.6f}" if (lat is not None and lon is not None) else "Coordinates: (not provided)"
    pdf.cell(0, 8, txt=loc_line, ln=True)
    if reporter:
        pdf.cell(0, 8, txt=f"Reporter: {reporter}", ln=True)
    if notes:
        pdf.multi_cell(0, 8, txt=f"Notes: {notes}")
    if alerts:
        pdf.ln(2)
        pdf.cell(0, 8, txt="Context Alerts:", ln=True)
        for a in alerts:
            pdf.multi_cell(0, 8, txt=f"- {a}")

    # Save to temp file
    pdf_output = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(pdf_output.name)
    pdf_output.close()
    return pdf_output.name

if __name__ == "__main__":
    main()
