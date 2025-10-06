"""
Split an input WAV into sequential 5-second clips using librosa and soundfile.

Instructions:
  - Adjust INPUT_AUDIO and OUTPUT_DIR as needed.
  - Run:  python -m src.audio.split_into_5s_clips
"""

from __future__ import annotations

import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


# --- Configuration ---
# Path to the input audio file (WAV recommended)
INPUT_AUDIO = Path("fished_maerl_bed.wav")

# Output directory for the generated clips
# Example: Path("processed/degraded/")
OUTPUT_DIR = Path("processed/degraded/")

# Duration of each clip in seconds
CLIP_DURATION_SEC = 5.0


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_clip(audio_segment: np.ndarray, sample_rate: int, index: int, out_dir: Path) -> None:
    """Write a single audio clip to disk with a sequential filename.

    Filenames are of the form: degraded_clip_001.wav, degraded_clip_002.wav, ...
    """
    filename = f"degraded_clip_{index:03d}.wav"
    out_path = out_dir / filename

    # soundfile expects shape (n_samples,) for mono or (n_samples, n_channels) for multi-channel
    if audio_segment.ndim == 1:
        data_to_write = audio_segment
    elif audio_segment.ndim == 2:
        # librosa.load(..., mono=False) returns (n_channels, n_samples)
        # transpose to (n_samples, n_channels)
        data_to_write = audio_segment.T
    else:
        raise ValueError("Unexpected audio array shape: {}".format(audio_segment.shape))

    sf.write(str(out_path), data=data_to_write, samplerate=sample_rate, subtype="PCM_16")


def main() -> None:
    if not INPUT_AUDIO.exists():
        raise FileNotFoundError(f"Input audio not found: {INPUT_AUDIO}")

    ensure_directory(OUTPUT_DIR)

    # Load audio, preserving original sample rate and channels
    # mono=False preserves channels; returns shape (n_channels, n_samples)
    audio, sr = librosa.load(str(INPUT_AUDIO), sr=None, mono=False)

    # Standardize to shape (n_channels, n_samples)
    if audio.ndim == 1:
        # Mono -> add channel axis for consistent slicing
        audio = np.expand_dims(audio, axis=0)

    n_channels, n_samples = audio.shape
    samples_per_clip = int(round(CLIP_DURATION_SEC * sr))
    if samples_per_clip <= 0:
        raise ValueError("CLIP_DURATION_SEC must be positive.")

    total_full_clips = n_samples // samples_per_clip
    if total_full_clips == 0:
        print("Audio is shorter than one clip; no clips written.")
        return

    print(f"Input: {INPUT_AUDIO}")
    print(f"Sample rate: {sr} Hz; Channels: {n_channels}; Duration: {n_samples / sr:.2f} s")
    print(f"Writing {total_full_clips} clips of {CLIP_DURATION_SEC:.1f} s to: {OUTPUT_DIR}")

    for i in range(total_full_clips):
        start = i * samples_per_clip
        end = start + samples_per_clip
        segment = audio[:, start:end]  # shape (n_channels, samples_per_clip)
        write_clip(segment, sr, i + 1, OUTPUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()


