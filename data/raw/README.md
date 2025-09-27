# Raw Audio Data

This directory contains the original audio files from hydrophone recordings.

## Directory Structure
```
data/raw/
├── healthy_reefs/          # Audio recordings from healthy coral reefs
├── degraded_reefs/         # Audio recordings from degraded coral reefs
├── high_anthrophony/       # Recordings with high human noise
├── low_anthrophony/        # Recordings with low human noise
└── sample_data/           # Sample recordings for testing
```

## File Naming Convention
- `{location}_{date}_{condition}.wav`
- Example: `great_barrier_reef_2024-01-15_healthy.wav`

## Audio Requirements
- **Format**: WAV files (preferred) or MP3
- **Duration**: 30-300 seconds
- **Sample Rate**: 22.05 kHz or higher
- **Channels**: Mono or Stereo
- **Quality**: Clear recordings without excessive noise

## Data Collection Guidelines
1. Record in calm conditions (minimal boat traffic)
2. Maintain consistent distance from reef (5-10 meters)
3. Record at consistent depth (2-5 meters)
4. Note environmental conditions (weather, visibility, etc.)
5. Include metadata file with each recording

## Metadata Format
Each audio file should have a corresponding `.json` metadata file:
```json
{
    "filename": "great_barrier_reef_2024-01-15_healthy.wav",
    "location": "Great Barrier Reef, Australia",
    "coordinates": {"lat": -18.2871, "lng": 147.6992},
    "date": "2024-01-15",
    "time": "14:30:00",
    "depth": 3.5,
    "distance_from_reef": 8.0,
    "weather": "calm",
    "visibility": "good",
    "reef_health": "healthy",
    "anthrophony_level": "low",
    "recording_equipment": "hydrophone_model_xyz",
    "notes": "Clear recording, no boat traffic"
}
```
