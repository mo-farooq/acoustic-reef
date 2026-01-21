# Acoustic Reef 🌊🎶  
**AI-powered stethoscope for the ocean**

---

## 📌 Mission
Acoustic Reef is an AI-powered platform that analyzes underwater soundscapes to assess coral reef health in real time. Healthy reefs are noisy with life, while degraded reefs fall silent — we use this sound signature as a vital sign.

---

## 🚩 Problem
Coral reefs are collapsing globally, and traditional monitoring methods (manual dive surveys) are too slow, expensive, and unscalable to provide the necessary early warnings for conservation efforts.

---

## 💡 Our Solution
- **Upload Audio**: Users upload .wav files from underwater microphones (hydrophones)
- **AI Analysis**: Multi-output model using Google SurfPerch + custom scikit-learn classifier
- **Reef Vital Signs Report**: Simultaneous classification of:
  - Reef health status (Healthy/Degraded)
  - Anthrophony detection (human-made noise presence)
- **Dashboard**: Simple web interface built with Streamlit

---

## 🔧 Tech Stack
- **Language**: Python
- **Web Dashboard**: Streamlit
- **AI Model Training/Usage**: Kaggle Notebooks, TensorFlow, scikit-learn
- **Audio & Data**: Librosa, Pandas, NumPy
- **Foundation Model**: Google SurfPerch (pre-trained, hosted on Kaggle)

---

## 🎯 Unique Value Proposition
Our innovation is democratizing powerful but inaccessible marine science. We are building the first user-friendly platform designed for non-experts like park rangers, conservation NGOs, and local communities.

---

## 📂 Project Structure
```
acoustic-reef/
├── data/                    # Audio datasets and samples
│   ├── raw/                # Original audio files
│   ├── processed/          # Preprocessed audio data
│   └── embeddings/         # SurfPerch embeddings
├── models/                 # Trained models and weights
│   ├── surfperch/          # SurfPerch model files
│   └── classifiers/        # Custom scikit-learn models
├── src/                    # Source code
│   ├── audio/             # Audio processing utilities
│   ├── models/            # Model training and inference
│   └── utils/              # General utilities
├── notebooks/              # Kaggle notebooks for model development
│   ├── surfperch/         # SurfPerch integration
│   ├── training/          # Model training notebooks
│   └── evaluation/        # Model evaluation notebooks
├── dashboard/              # Streamlit web application
│   ├── app.py             # Main Streamlit app
│   ├── components/        # UI components
│   └── pages/             # Multi-page structure
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

---

## 👩‍💻 Team
- **Onkar Damale** - Project Lead & AI/ML Lead
- **Om Niungare** - Research & API Integration Lead
- **Farooq** - Data Curation & Management Lead
- **Shyamsundar Gitte** - UI/UX & Dashboard Design Lead  

---

## 🚀 Getting Started
```bash
git clone https://github.com/your-username/acoustic-reef.git
cd acoustic-reef

pip install -r requirements.txt

streamlit run dashboard/app.py
\`\`\`

## 🛠 Development
To contribute to this project:
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Run tests: `pytest`

---

## 📦 Data Artifacts
- Precomputed embeddings: `data/embeddings/embeddings.csv`
- UMAP coordinates: `data/processed/umap_coordinates.csv`

---

## 🔬 Technical Approach
1. **Foundation Model**: Use pre-trained Google SurfPerch model from Kaggle
2. **Embeddings**: Generate audio embeddings using TensorFlow saved_model format
3. **Custom Classifier**: Train scikit-learn classifier on SurfPerch embeddings
4. **Multi-Output**: Simultaneous reef health and anthrophony classification
5. **Web Interface**: Streamlit dashboard for user interaction
