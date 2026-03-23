"""
voiced_unvoiced.py — Voiced/Unvoiced/Silence boundary detection
using cepstral quefrency analysis.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """A contiguous audio region with a single phonation label."""
    start_time: float   # seconds
    end_time: float     # seconds
    label: str          # "voiced" | "unvoiced" | "silence"

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


# ---------------------------------------------------------------------------
# Core cepstral functions
# ---------------------------------------------------------------------------

def compute_cepstrum(frame: np.ndarray, sr: int) -> np.ndarray:
    """Real cepstrum: IFFT(log|FFT(frame)|).

    Parameters
    ----------
    frame : 1-D array of audio samples for one frame.
    sr    : sample rate (unused in computation but kept for API consistency).

    Returns
    -------
    cepstrum : 1-D real array, same length as *frame*.
    """
    eps = 1e-10
    spectrum = np.abs(np.fft.fft(frame))
    log_spectrum = np.log(spectrum + eps)
    cepstrum = np.fft.ifft(log_spectrum).real
    return cepstrum


def low_quefrency_energy(cepstrum: np.ndarray, cutoff_sample: int) -> float:
    """Sum of squared cepstrum values below *cutoff_sample* (formant region).

    Parameters
    ----------
    cepstrum      : 1-D cepstrum array.
    cutoff_sample : index separating low from high quefrency.
    """
    return float(np.sum(cepstrum[:cutoff_sample] ** 2))


def high_quefrency_energy(cepstrum: np.ndarray, cutoff_sample: int) -> float:
    """Sum of squared cepstrum values from *cutoff_sample* onwards (pitch region).

    Parameters
    ----------
    cepstrum      : 1-D cepstrum array.
    cutoff_sample : index separating low from high quefrency.
    """
    return float(np.sum(cepstrum[cutoff_sample:] ** 2))


def classify_frame(
    low_energy: float,
    high_energy: float,
    voiced_threshold: float = 0.3,
    silence_threshold: float = 1e-6,
) -> str:
    """Classify a frame as "voiced", "unvoiced", or "silence".

    Decision logic
    --------------
    - "silence"  : low_energy < silence_threshold  (very quiet frame)
    - "voiced"   : high_energy / (low_energy + eps) > voiced_threshold
    - "unvoiced" : otherwise
    """
    eps = 1e-10
    if low_energy < silence_threshold:
        return "silence"
    ratio = high_energy / (low_energy + eps)
    if ratio > voiced_threshold:
        return "voiced"
    return "unvoiced"


# ---------------------------------------------------------------------------
# Boundary detection
# ---------------------------------------------------------------------------

def _load_audio(audio_path: str, target_sr: int) -> tuple[np.ndarray, int]:
    """Load a WAV file, resampling to *target_sr* if needed."""
    try:
        import soundfile as sf
        signal, sr = sf.read(audio_path, always_2d=False)
    except ImportError:
        from scipy.io import wavfile
        sr, signal = wavfile.read(audio_path)
        if signal.dtype != np.float32 and signal.dtype != np.float64:
            signal = signal.astype(np.float32) / np.iinfo(signal.dtype).max

    # Convert stereo → mono
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    # Resample if needed (simple linear interpolation)
    if sr != target_sr:
        n_samples_new = int(len(signal) * target_sr / sr)
        signal = np.interp(
            np.linspace(0, len(signal) - 1, n_samples_new),
            np.arange(len(signal)),
            signal,
        )
        sr = target_sr

    return signal.astype(np.float64), sr


def _frame_signal(
    signal: np.ndarray,
    sr: int,
    win_len_ms: float,
    hop_len_ms: float,
) -> np.ndarray:
    """Segment *signal* into overlapping frames.

    Returns
    -------
    frames : shape (n_frames, frame_len)
    """
    frame_len = int(round(win_len_ms * 1e-3 * sr))
    hop_len = int(round(hop_len_ms * 1e-3 * sr))
    n_frames = max(1, 1 + (len(signal) - frame_len) // hop_len)
    frames = np.zeros((n_frames, frame_len), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_len
        end = start + frame_len
        chunk = signal[start:min(end, len(signal))]
        frames[i, : len(chunk)] = chunk
    return frames


def detect_boundaries(
    audio_path: str,
    sr: int = 16000,
    win_len_ms: float = 25.0,
    hop_len_ms: float = 10.0,
    voiced_threshold: float = 0.3,
    cutoff_ms: float = 1.5,
) -> List[Segment]:
    """Detect voiced/unvoiced/silence boundaries in an audio file.

    Parameters
    ----------
    audio_path       : path to WAV file.
    sr               : target sample rate.
    win_len_ms       : analysis window length in milliseconds.
    hop_len_ms       : hop size in milliseconds.
    voiced_threshold : ratio threshold for voiced classification.
    cutoff_ms        : quefrency cutoff in milliseconds.

    Returns
    -------
    List of non-overlapping :class:`Segment` objects covering the full audio.
    """
    signal, sr = _load_audio(audio_path, sr)
    frames = _frame_signal(signal, sr, win_len_ms, hop_len_ms)

    cutoff_sample = max(1, round(cutoff_ms * 1e-3 * sr))
    hop_len = int(round(hop_len_ms * 1e-3 * sr))

    labels: List[str] = []
    for frame in frames:
        cepstrum = compute_cepstrum(frame, sr)
        lo = low_quefrency_energy(cepstrum, cutoff_sample)
        hi = high_quefrency_energy(cepstrum, cutoff_sample)
        labels.append(classify_frame(lo, hi, voiced_threshold))

    # Merge consecutive same-label frames into Segment objects
    segments: List[Segment] = []
    if not labels:
        return segments

    current_label = labels[0]
    seg_start = 0.0

    for i, label in enumerate(labels[1:], start=1):
        if label != current_label:
            seg_end = i * hop_len_ms * 1e-3
            segments.append(Segment(seg_start, seg_end, current_label))
            current_label = label
            seg_start = seg_end

    # Final segment — extend to full audio duration
    total_duration = len(signal) / sr
    segments.append(Segment(seg_start, total_duration, current_label))

    return segments


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

_LABEL_COLORS = {
    "voiced": "green",
    "unvoiced": "orange",
    "silence": "gray",
}


def visualize_boundaries(
    audio_path: str,
    segments: List[Segment],
    out_path: str,
) -> None:
    """Save a waveform plot with coloured segment overlays.

    Parameters
    ----------
    audio_path : path to the source WAV file.
    segments   : list of :class:`Segment` objects from :func:`detect_boundaries`.
    out_path   : destination PNG path.
    """
    signal, sr = _load_audio(audio_path, 16000)
    times = np.linspace(0, len(signal) / sr, len(signal))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(times, signal, color="steelblue", linewidth=0.5, alpha=0.8)

    for seg in segments:
        ax.axvspan(
            seg.start_time,
            seg.end_time,
            alpha=0.25,
            color=_LABEL_COLORS.get(seg.label, "purple"),
        )

    # Legend
    patches = [
        mpatches.Patch(color=color, alpha=0.5, label=label)
        for label, color in _LABEL_COLORS.items()
    ]
    ax.legend(handles=patches, loc="upper right")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Voiced/Unvoiced/Silence — {os.path.basename(audio_path)}")
    ax.set_xlim(0, times[-1])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualisation → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    WAV = "examples/0.wav"
    OUT = "data/vuv_plots/0_boundaries.png"

    print(f"Processing {WAV} …")
    segs = detect_boundaries(WAV)

    print(f"\n{'Start':>8}  {'End':>8}  {'Dur':>6}  Label")
    print("-" * 38)
    for s in segs:
        print(f"{s.start_time:8.3f}  {s.end_time:8.3f}  {s.duration:6.3f}  {s.label}")

    visualize_boundaries(WAV, segs, OUT)
