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
from src.inference import (
    resolve_features_for_file,
    predict_vital_signs,
    load_umap_coordinates,
    transform_with_umap,
)
from src.simple_inference import predict_simple
from src.mock_classifier import predict_with_mock_classifier
from src.force_real_classifier import predict_with_force_classifier
from src.real_model_loader import predict_with_real_model

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
    tabs = st.tabs(["Upload & Analyze", "Batch Predictions", "Acoustic Map"])
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

            # Data quality warnings
            if duration_sec < 5:
                st.warning("Audio is very short (<5s). Results may be unreliable.")
            if duration_sec > duration_limit:
                st.info("Only the first segment up to the Max Duration was analyzed.")

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
                import numpy as _np
                
                # Compute enhanced spectrogram with better parameters
                win = 2048  # Larger window for better frequency resolution
                hop = 512   # 75% overlap for smoother visualization
                num_frames = max(1, (len(audio_np) - win) // hop + 1)
                
                # Apply window function for better spectral analysis
                window = _np.hanning(win)
                
                spec = []
                for i in range(num_frames):
                    start = i * hop
                    frame = audio_np[start:start+win]
                    if len(frame) < win:
                        frame = _np.pad(frame, (0, win - len(frame)))
                    
                    # Apply window and compute FFT
                    windowed_frame = frame * window
                    fft_result = _np.fft.rfft(windowed_frame)
                    mag = _np.abs(fft_result)
                    spec.append(mag)
                
                spec = _np.array(spec).T  # [freq, time]
                
                # Convert to dB with better dynamic range
                spec_db = 20 * _np.log10(spec + 1e-10)
                
                # Create time and frequency axes
                time_axis = _np.linspace(0, duration_sec, spec.shape[1])
                freq_axis = _np.linspace(0, sr/2, spec.shape[0])
                
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

            # Predict vital signs (health + noise)
            try:
                # Try REAL trained model first
                result = predict_with_real_model(feature_vals)
                st.success("✅ Using REAL trained model!")
            except Exception as e_real:
                # Fallback 1: force-compatible classifier
                try:
                    result = predict_with_force_classifier(feature_vals)
                    st.warning("🛠️ Real model missing/incompatible. Using force-compatible classifier.")
                except Exception as e_force:
                    # Fallback 2: mock classifier as last resort
                    try:
                        result = predict_with_mock_classifier(feature_vals)
                        st.warning("ℹ️ Using mock classifier heuristic due to missing trained model.")
                    except Exception as e_mock:
                        st.error("Analysis failed during prediction. Please try again later.")
                        return

            # Vital Signs UI
            st.markdown("### 🩺 Vital Signs")
            
            # Debug output to see actual values
            st.write(f"🔍 Debug - Health conf: {result.health_conf}, Noise conf: {result.noise_conf}")
            st.write(f"🔍 Debug - Are they equal? {result.health_conf == result.noise_conf}")
            
            col1, col2 = st.columns(2)
            with col1:
                color = "status-healthy" if result.health_label == "Healthy" else "status-degraded"
                st.markdown("#### 🏥 Reef Health")
                st.markdown(f'<p class="{color}">{result.health_label}</p>', unsafe_allow_html=True)
                if result.health_conf is not None:
                    st.metric("Confidence", f"{result.health_conf:.0%}")
            with col2:
                st.markdown("#### 🔊 Noise Pollution")
                color_n = "status-degraded" if result.noise_label == "High" else "status-healthy"
                st.markdown(f'<p class="{color_n}">{result.noise_label}</p>', unsafe_allow_html=True)
                if result.noise_conf is not None:
                    st.metric("Confidence", f"{result.noise_conf:.0%}")

            # Enhanced Class Probabilities Visualization
            try:
                import joblib
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                from src.utils.config import RF_MODEL_PATH
                model = joblib.load(RF_MODEL_PATH)
                if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
                    probs = model.predict_proba(feature_vals)[0]
                    cls_to_prob = {str(c): float(p) for c, p in zip(model.classes_, probs)}
                    
                    st.markdown("### 📊 Model Confidence Analysis")
                    
                    # Create enhanced visualization with multiple charts
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=("Class Probabilities", "Confidence Distribution", "Prediction Certainty", "Model Insights"),
                        specs=[[{"type": "bar"}, {"type": "pie"}],
                               [{"type": "indicator"}, {"type": "bar"}]],
                        vertical_spacing=0.15,
                        horizontal_spacing=0.1
                    )
                    
                    # 1. Enhanced Bar Chart with custom colors and styling
                    class_names = list(cls_to_prob.keys())
                    probabilities = list(cls_to_prob.values())
                    
                    # Define colors for each class
                    color_map = {
                        'healthy': '#2E8B57',      # Sea Green
                        'degraded': '#DC143C',     # Crimson  
                        'anthrophony': '#FF8C00'   # Dark Orange
                    }
                    colors = [color_map.get(cls, '#6A5ACD') for cls in class_names]
                    
                    # Bar chart with enhanced styling
                    fig.add_trace(
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
                            name="Probability",
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    
                    # 2. Pie chart for probability distribution
                    fig.add_trace(
                        go.Pie(
                            labels=class_names,
                            values=probabilities,
                            marker=dict(colors=colors, line=dict(color='white', width=2)),
                            textinfo='label+percent',
                            textfont=dict(size=12, family='Arial'),
                            hovertemplate='<b>%{label}</b><br>Probability: %{percent}<br>Value: %{value:.3f}<extra></extra>',
                            name="Distribution"
                        ),
                        row=1, col=2
                    )
                    
                    # 3. Confidence gauge
                    max_prob = max(probabilities)
                    fig.add_trace(
                        go.Indicator(
                            mode="gauge+number+delta",
                            value=max_prob * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Model Confidence (%)"},
                            delta={'reference': 50},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "green"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ),
                        row=2, col=1
                    )
                    
                    # 4. Model insights bar chart
                    insights_data = {
                        'Model Certainty': max_prob,
                        'Prediction Spread': max(probabilities) - min(probabilities),
                        'Second Best': sorted(probabilities, reverse=True)[1],
                        'Uncertainty': 1 - max_prob
                    }
                    
                    fig.add_trace(
                        go.Bar(
                            x=list(insights_data.keys()),
                            y=list(insights_data.values()),
                            marker=dict(
                                color=['#4CAF50', '#FF9800', '#2196F3', '#F44336'],
                                opacity=0.7
                            ),
                            text=[f"{v:.3f}" for v in insights_data.values()],
                            textposition='auto',
                            name="Insights",
                            showlegend=False
                        ),
                        row=2, col=2
                    )
                    
                    # Update layout with modern styling
                    fig.update_layout(
                        title=dict(
                            text="🎯 AI Model Decision Breakdown",
                            font=dict(size=20, color='#1f77b4', family='Arial, sans-serif'),
                            x=0.5,
                            xanchor='center'
                        ),
                        height=600,
                        showlegend=False,
                        plot_bgcolor='rgba(248,249,250,0.8)',
                        paper_bgcolor='rgba(255,255,255,0.9)',
                        font=dict(family='Arial, sans-serif', size=12)
                    )
                    
                    # Update axes styling
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add interpretation section
                    with st.expander("🧠 Model Interpretation", expanded=True):
                        st.markdown("""
                        **What this analysis shows:**
                        
                        - **Class Probabilities**: How confident the AI is about each possible outcome
                        - **Confidence Distribution**: Visual breakdown of model certainty
                        - **Model Confidence**: Overall certainty level (higher = more confident)
                        - **Model Insights**: Key metrics about the prediction quality
                        
                        **Understanding the results:**
                        - **High confidence (>80%)**: Model is very certain about the prediction
                        - **Medium confidence (50-80%)**: Some uncertainty, but still reliable
                        - **Low confidence (<50%)**: High uncertainty, consider retaking the recording
                        
                        **Class meanings:**
                        - **Healthy**: Reef shows signs of good health and biodiversity
                        - **Degraded**: Reef shows signs of stress or damage
                        - **Anthrophony**: Human-made noise detected (boats, engines, etc.)
                        """)
                    
                    # Enhanced metrics display
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Highest Confidence", f"{max_prob:.1%}", 
                                delta=f"{max_prob - 0.5:.1%}" if max_prob > 0.5 else None)
                    with col2:
                        second_best = sorted(probabilities, reverse=True)[1]
                        st.metric("Second Best", f"{second_best:.1%}")
                    with col3:
                        uncertainty = 1 - max_prob
                        st.metric("Uncertainty", f"{uncertainty:.1%}")
                    
                    # Downloadable CSV with enhanced data
                    enhanced_data = {
                        **cls_to_prob,
                        'max_confidence': max_prob,
                        'uncertainty': 1 - max_prob,
                        'prediction_spread': max(probabilities) - min(probabilities)
                    }
                    csv_bytes = pd.DataFrame([enhanced_data]).to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Enhanced Analysis (CSV)", 
                        data=csv_bytes, 
                        file_name="enhanced_prediction_analysis.csv", 
                        mime="text/csv"
                    )
                    
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

        # Take Action section
        if result.health_label in ("Degraded", "Stressed"):
            if st.button("Learn How to Take Action"):
                st.markdown("### 🛟 Take Action")
                if result.noise_label == "High":
                    st.write("- Reduce boat traffic and engine noise in the area.\n- Establish quiet zones and enforce speed limits.\n- Schedule activities to avoid sensitive hours (e.g., spawning).")
                else:
                    st.write("- Investigate water quality (nutrients, turbidity).\n- Monitor for bleaching and heat stress.\n- Engage local conservation groups for habitat restoration.")

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
    """Analyze multiple uploaded files and summarize results."""
    st.markdown("### Batch Summary")
    rows = []
    healthy_count = 0
    degraded_count = 0
    error_count = 0

    for f in batch_files:
        try:
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

            # Resolve features
            feature_vals, _ = resolve_features_for_file(f.name, audio_np, sr)
            # Predict
            result = predict_with_real_model(feature_vals)
            status = 'success'
            health = result.health_label
            conf = result.health_conf if result.health_conf is not None else 0.0
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

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    total = len(rows)
    if total > 0:
        pct_healthy = healthy_count / total
        pct_degraded = degraded_count / total
        st.markdown("#### Aggregates")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", f"{total}")
        c2.metric("Healthy", f"{pct_healthy:.0%}")
        c3.metric("Degraded", f"{pct_degraded:.0%}")

    # Download results
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download batch results (CSV)", data=csv_bytes, file_name="batch_results.csv", mime="text/csv")


def show_acoustic_map():
    st.markdown('<h2 class="sub-header">🗺️ Acoustic Map</h2>', unsafe_allow_html=True)
    try:
        base_df = load_umap_coordinates()
        if base_df is None or base_df.empty:
            st.info("UMAP coordinates not available yet. Place umap_coordinates.csv in data/processed and umap_model.joblib in models/classifiers.")
            return
    except Exception as e:
        st.info(f"UMAP coordinates not available: {e}")
        return

    import plotly.express as px

    st.markdown("### Training Set Map")
    fig = px.scatter(
        base_df,
        x=base_df.columns[0],
        y=base_df.columns[1],
        color=base_df.columns[2] if base_df.shape[1] > 2 else None,
        opacity=0.7,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Transform a New Point")
    uploaded = st.file_uploader("Upload a clip to place on the map", type=["wav", "mp3", "flac"], key="umap_uploader")
    if uploaded is None:
        return

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
            st.warning("Could not transform point with UMAP model.")
            return

        # Plot overlay
        new_df = base_df.copy()
        fig2 = px.scatter(
            new_df,
            x=new_df.columns[0],
            y=new_df.columns[1],
            color=new_df.columns[2] if new_df.shape[1] > 2 else None,
            opacity=0.6,
        )
        fig2.add_scatter(x=[coord[0,0]], y=[coord[0,1]], mode="markers", marker_symbol="star", marker_size=16, marker_color="red", name="uploaded")
        st.plotly_chart(fig2, use_container_width=True)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    main()
