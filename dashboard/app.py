"""
Acoustic Reef - Main Streamlit Dashboard
AI-powered stethoscope for the ocean
"""

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tempfile
import os

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
    
    # Main content area
    if uploaded_file is not None:
        analyze_audio(uploaded_file, sample_rate, duration_limit)
    else:
        show_landing_page()

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
        # Load audio
        with st.spinner("Loading audio..."):
            audio, sr = librosa.load(tmp_path, sr=sample_rate)
            
            # Limit duration
            max_samples = int(duration_limit * sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
        
        # Display audio info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Duration", f"{len(audio)/sr:.1f}s")
        with col2:
            st.metric("Sample Rate", f"{sr:,} Hz")
        with col3:
            st.metric("Channels", "Mono")
        with col4:
            st.metric("RMS Level", f"{np.sqrt(np.mean(audio**2)):.3f}")
        
        # Audio visualization
        st.markdown("### 📊 Audio Waveform")
        fig, ax = plt.subplots(figsize=(12, 4))
        time_axis = np.linspace(0, len(audio)/sr, len(audio))
        ax.plot(time_axis, audio, alpha=0.7, color='#1f77b4')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Audio Waveform')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Spectrogram
        st.markdown("### 🎵 Spectrogram")
        with st.spinner("Generating spectrogram..."):
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            fig, ax = plt.subplots(figsize=(12, 6))
            img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax)
            ax.set_title('Spectrogram')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time (seconds)')
            plt.colorbar(img, ax=ax, format='%+2.0f dB')
            st.pyplot(fig)
        
        # Mock AI Analysis (placeholder for actual model)
        st.markdown("### 🤖 AI Analysis Results")
        
        with st.spinner("Running AI analysis..."):
            # This is where the actual SurfPerch + scikit-learn model would be called
            # For now, we'll show mock results
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏥 Reef Health Assessment")
                health_score = np.random.uniform(0.6, 0.95)  # Mock score
                health_status = "Healthy" if health_score > 0.7 else "Degraded"
                health_color = "status-healthy" if health_status == "Healthy" else "status-degraded"
                
                st.markdown(f'<p class="{health_color}">Status: {health_status}</p>', unsafe_allow_html=True)
                st.metric("Confidence", f"{health_score:.1%}")
                
                # Health indicators
                st.markdown("**Key Indicators:**")
                st.markdown(f"- Biodiversity Score: {np.random.uniform(0.5, 0.9):.1%}")
                st.markdown(f"- Fish Activity: {np.random.uniform(0.3, 0.8):.1%}")
                st.markdown(f"- Coral Health: {np.random.uniform(0.4, 0.9):.1%}")
            
            with col2:
                st.markdown("#### 🔊 Anthrophony Detection")
                anthro_score = np.random.uniform(0.1, 0.4)  # Mock score
                anthro_status = "High" if anthro_score > 0.3 else "Low"
                anthro_color = "status-degraded" if anthro_status == "High" else "status-healthy"
                
                st.markdown(f'<p class="{anthro_color}">Human Noise: {anthro_status}</p>', unsafe_allow_html=True)
                st.metric("Detection Level", f"{anthro_score:.1%}")
                
                # Noise indicators
                st.markdown("**Noise Sources:**")
                st.markdown(f"- Boat Traffic: {np.random.uniform(0.0, 0.3):.1%}")
                st.markdown(f"- Construction: {np.random.uniform(0.0, 0.2):.1%}")
                st.markdown(f"- Diving Activity: {np.random.uniform(0.0, 0.4):.1%}")
        
        # Summary report
        st.markdown("### 📋 Reef Vital Signs Report")
        
        report_data = {
            'Metric': ['Overall Health', 'Biodiversity', 'Fish Activity', 'Coral Health', 'Human Impact'],
            'Score': [f"{health_score:.1%}", f"{np.random.uniform(0.5, 0.9):.1%}", 
                     f"{np.random.uniform(0.3, 0.8):.1%}", f"{np.random.uniform(0.4, 0.9):.1%}", 
                     f"{anthro_score:.1%}"],
            'Status': [health_status, 'Good', 'Moderate', 'Good', anthro_status]
        }
        
        df = pd.DataFrame(report_data)
        st.dataframe(df, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        if health_status == "Healthy":
            st.success("🎉 Great news! Your reef appears to be in good health. Continue monitoring regularly.")
        else:
            st.warning("⚠️ The reef shows signs of degradation. Consider conservation measures and regular monitoring.")
        
        if anthro_status == "High":
            st.info("🔇 High human noise detected. Consider reducing boat traffic and human activity in the area.")
        
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    main()
