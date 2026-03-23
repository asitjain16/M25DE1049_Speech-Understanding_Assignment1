"""
Manual MFCC pipeline using only numpy and scipy.
No librosa.feature.mfcc or equivalent high-level MFCC functions are used.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.fftpack import dct
from scipy.signal import get_window


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class MFCCConfig:
    sr: int = 16000
    n_mels: int = 40
    n_ceps: int = 13
    win_len_ms: float = 25.0
    hop_len_ms: float = 10.0
    pre_emphasis_coeff: float = 0.97
    window_type: str = "hamming"  # "hamming" | "hanning" | "rectangular"
    fmin: float = 0.0
    fmax: float = 8000.0


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def pre_emphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply pre-emphasis filter: y[n] = x[n] - coeff * x[n-1]."""
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def frame_signal(
    signal: np.ndarray,
    sr: int,
    win_len_ms: float = 25.0,
    hop_len_ms: float = 10.0,
) -> np.ndarray:
    """
    Segment signal into overlapping frames.

    Returns
    -------
    np.ndarray
        Shape (n_frames, frame_len).
    """
    frame_len = int(round(win_len_ms * sr / 1000.0))
    hop_len = int(round(hop_len_ms * sr / 1000.0))

    # Pad signal so every sample is covered
    n_frames = 1 + max(0, (len(signal) - frame_len + hop_len - 1) // hop_len)
    pad_len = (n_frames - 1) * hop_len + frame_len - len(signal)
    if pad_len > 0:
        signal = np.pad(signal, (0, pad_len), mode="constant")

    # Strided view for zero-copy framing
    shape = (n_frames, frame_len)
    strides = (signal.strides[0] * hop_len, signal.strides[0])
    frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
    return frames.copy()  # copy so downstream writes don't corrupt the view


def apply_window(frames: np.ndarray, window_type: str = "hamming") -> np.ndarray:
    """
    Apply a window function to each frame.

    Parameters
    ----------
    window_type : str
        One of "hamming", "hanning", "rectangular".
    """
    frame_len = frames.shape[1]
    wtype = window_type.lower()
    if wtype == "rectangular":
        window = np.ones(frame_len, dtype=np.float64)
    elif wtype == "hamming":
        window = get_window("hamming", frame_len, fftbins=False)
    elif wtype == "hanning":
        # scipy uses "hann" as the canonical name
        window = get_window("hann", frame_len, fftbins=False)
    else:
        raise ValueError(f"Unsupported window type: {window_type!r}. "
                         "Choose from 'hamming', 'hanning', 'rectangular'.")
    return frames * window[np.newaxis, :]


def compute_fft(windowed_frames: np.ndarray) -> np.ndarray:
    """
    Compute magnitude spectrum for each windowed frame.

    Returns
    -------
    np.ndarray
        Shape (n_frames, n_fft//2 + 1) — one-sided magnitude spectrum.
    """
    n_fft = windowed_frames.shape[1]
    # rfft gives the one-sided complex spectrum; take absolute value
    magnitude = np.abs(np.fft.rfft(windowed_frames, n=n_fft, axis=1))
    return magnitude  # shape: (n_frames, n_fft//2 + 1)


def _hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    """Convert Hz to mel scale (HTK formula)."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    """Convert mel scale to Hz (HTK formula)."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int,
    n_fft: int,
    sr: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """
    Build a triangular mel filterbank.

    Returns
    -------
    np.ndarray
        Shape (n_mels, n_fft//2 + 1).
    """
    if fmax is None:
        fmax = sr / 2.0

    n_bins = n_fft // 2 + 1
    # Linearly spaced mel points (n_mels + 2 to include lower and upper edges)
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    # Map Hz to FFT bin indices
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, n_bins - 1)

    filterbank = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(1, n_mels + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]

        # Rising slope
        for k in range(f_left, f_center + 1):
            denom = f_center - f_left
            filterbank[m - 1, k] = (k - f_left) / denom if denom > 0 else 1.0

        # Falling slope
        for k in range(f_center, f_right + 1):
            denom = f_right - f_center
            filterbank[m - 1, k] = (f_right - k) / denom if denom > 0 else 1.0

    return filterbank


def apply_mel_filterbank(
    spectrum: np.ndarray, filterbank: np.ndarray
) -> np.ndarray:
    """
    Apply mel filterbank to magnitude spectrum via matrix multiply.

    Parameters
    ----------
    spectrum : np.ndarray
        Shape (n_frames, n_fft//2 + 1).
    filterbank : np.ndarray
        Shape (n_mels, n_fft//2 + 1).

    Returns
    -------
    np.ndarray
        Shape (n_frames, n_mels).
    """
    return spectrum @ filterbank.T


def log_compress(mel_energies: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Apply log compression: log(mel_energies + eps)."""
    return np.log(mel_energies + eps)


def apply_dct(log_mel: np.ndarray, n_ceps: int = 13) -> np.ndarray:
    """
    Apply DCT Type-II (orthonormal) and return the first n_ceps coefficients.

    Parameters
    ----------
    log_mel : np.ndarray
        Shape (n_frames, n_mels).
    n_ceps : int
        Number of cepstral coefficients to keep.

    Returns
    -------
    np.ndarray
        Shape (n_frames, n_ceps).
    """
    # norm="ortho" gives the orthonormal Type-II DCT
    cepstra = dct(log_mel, type=2, axis=1, norm="ortho")
    return cepstra[:, :n_ceps]


# ---------------------------------------------------------------------------
# Top-level extraction function
# ---------------------------------------------------------------------------

def _load_audio(audio_path: str, target_sr: int) -> tuple[np.ndarray, int]:
    """Load a WAV file and return (signal_float32, sr)."""
    try:
        import soundfile as sf
        signal, sr = sf.read(audio_path, always_2d=False)
    except (ImportError, Exception):
        from scipy.io import wavfile
        sr, signal = wavfile.read(audio_path)

    # Convert to float64 in [-1, 1]
    if signal.dtype.kind == "i":
        signal = signal.astype(np.float64) / np.iinfo(signal.dtype).max
    elif signal.dtype.kind == "u":
        signal = signal.astype(np.float64) / np.iinfo(signal.dtype).max * 2.0 - 1.0
    else:
        signal = signal.astype(np.float64)

    # Downmix to mono if stereo
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    # Resample if needed (simple linear interpolation for basic resampling)
    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(target_sr, sr)
        signal = resample_poly(signal, target_sr // g, sr // g)
        sr = target_sr

    return signal, sr


def extract_mfcc(
    audio_path: str,
    n_mels: int = 40,
    n_ceps: int = 13,
    window_type: str = "hamming",
) -> np.ndarray:
    """
    Run the full MFCC pipeline on a WAV file.

    Parameters
    ----------
    audio_path : str
        Path to a valid WAV file.
    n_mels : int
        Number of mel filterbank channels.
    n_ceps : int
        Number of cepstral coefficients to return.
    window_type : str
        Window function: "hamming", "hanning", or "rectangular".

    Returns
    -------
    np.ndarray
        Shape (n_frames, n_ceps) — all values are finite floats.
    """
    cfg = MFCCConfig(n_mels=n_mels, n_ceps=n_ceps, window_type=window_type)

    signal, sr = _load_audio(audio_path, cfg.sr)

    # 1. Pre-emphasis
    emphasized = pre_emphasis(signal, cfg.pre_emphasis_coeff)

    # 2. Framing
    frames = frame_signal(emphasized, sr, cfg.win_len_ms, cfg.hop_len_ms)

    # 3. Windowing
    windowed = apply_window(frames, cfg.window_type)

    # 4. FFT magnitude spectrum
    spectrum = compute_fft(windowed)

    # 5. Mel filterbank
    n_fft = windowed.shape[1]
    filterbank = mel_filterbank(cfg.n_mels, n_fft, sr, cfg.fmin, cfg.fmax)

    # 6. Apply filterbank
    mel_energies = apply_mel_filterbank(spectrum, filterbank)

    # 7. Log compression
    log_mel = log_compress(mel_energies)

    # 8. DCT → MFCCs
    mfccs = apply_dct(log_mel, cfg.n_ceps)

    assert mfccs.shape[1] == n_ceps, f"Expected {n_ceps} ceps, got {mfccs.shape[1]}"
    assert np.all(np.isfinite(mfccs)), "MFCC output contains non-finite values"

    return mfccs


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    wav_path = os.path.join("examples", "0.wav")
    if not os.path.exists(wav_path):
        print(f"File not found: {wav_path}", file=sys.stderr)
        sys.exit(1)

    mfccs = extract_mfcc(wav_path)
    print(f"MFCC shape: {mfccs.shape}")  # (n_frames, 13)
