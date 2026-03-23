"""
Privacy-Preserving Transformation Demo
=======================================
Loads an audio example, extracts mel spectrogram features (80 mel bins),
runs PrivacyModule.forward with source="male_old" → target="female_young",
and saves the reconstructed audio to q3/examples/transformed_0.wav.
"""

import os
import sys
import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import stft, istft
from scipy.fft import dct

# ---------------------------------------------------------------------------
# Path setup — allow `from q3.privacymodule import ...` or direct import
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from q3.privacymodule import PrivacyModule, ATTRIBUTE_MAP  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_MELS = 80
SR_TARGET = 16000          # resample target (samples/sec)
N_FFT = 512
HOP_LENGTH = 160           # 10 ms at 16 kHz
WIN_LENGTH = 400           # 25 ms at 16 kHz
F_MIN = 80.0
F_MAX = 7600.0
GRIFFIN_LIM_ITERS = 60


# ---------------------------------------------------------------------------
# Mel filterbank (numpy/scipy, no librosa)
# ---------------------------------------------------------------------------

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(n_mels: int, n_fft: int, sr: int,
                         f_min: float, f_max: float) -> np.ndarray:
    """Return (n_mels, n_fft//2+1) filterbank matrix."""
    n_freqs = n_fft // 2 + 1
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, n_freqs))
    for m in range(1, n_mels + 1):
        lo, center, hi = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(lo, center):
            if center != lo:
                fb[m - 1, k] = (k - lo) / (center - lo)
        for k in range(center, hi):
            if hi != center:
                fb[m - 1, k] = (hi - k) / (hi - center)
    return fb


def extract_mel_spectrogram(wav_path: str, n_mels: int = N_MELS) -> tuple[np.ndarray, int]:
    """
    Load wav, compute mel spectrogram.
    Returns (mel_spec, sr) where mel_spec has shape (n_frames, n_mels).
    """
    sr, audio = wavfile.read(wav_path)

    # Convert to float32 in [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    else:
        audio = audio.astype(np.float32)

    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Simple nearest-neighbour resample if needed
    if sr != SR_TARGET:
        ratio = SR_TARGET / sr
        new_len = int(len(audio) * ratio)
        indices = np.round(np.linspace(0, len(audio) - 1, new_len)).astype(int)
        audio = audio[indices]
        sr = SR_TARGET

    # STFT
    hop = HOP_LENGTH
    win = WIN_LENGTH
    n_fft = N_FFT
    window = np.hanning(win)
    _, _, Zxx = stft(audio, fs=sr, window=window, nperseg=win,
                     noverlap=win - hop, nfft=n_fft, padded=True)
    power_spec = np.abs(Zxx) ** 2  # (n_freqs, n_frames)

    # Mel filterbank
    fb = build_mel_filterbank(n_mels, n_fft, sr, F_MIN, F_MAX)
    mel_spec = fb @ power_spec          # (n_mels, n_frames)
    mel_spec = np.log(mel_spec + 1e-9)  # log-mel
    return mel_spec.T, sr               # (n_frames, n_mels)


# ---------------------------------------------------------------------------
# Griffin-Lim reconstruction from mel spectrogram
# ---------------------------------------------------------------------------

def griffin_lim_from_mel(mel_spec: np.ndarray, sr: int,
                         n_mels: int = N_MELS,
                         n_iters: int = GRIFFIN_LIM_ITERS) -> np.ndarray:
    """
    Approximate audio reconstruction from log-mel spectrogram via Griffin-Lim.
    mel_spec: (n_frames, n_mels)
    Returns 1-D float32 audio array.
    """
    n_fft = N_FFT
    hop = HOP_LENGTH
    win = WIN_LENGTH
    window = np.hanning(win)

    # Invert log
    mel_power = np.exp(mel_spec.T)  # (n_mels, n_frames)

    # Pseudo-inverse of mel filterbank to get linear power spectrum
    fb = build_mel_filterbank(n_mels, n_fft, sr, F_MIN, F_MAX)
    fb_pinv = np.linalg.pinv(fb)                    # (n_freqs, n_mels)
    linear_power = np.maximum(fb_pinv @ mel_power, 0.0)  # (n_freqs, n_frames)
    magnitude = np.sqrt(linear_power)

    # Griffin-Lim iterations
    angles = np.exp(1j * np.random.uniform(0, 2 * np.pi, magnitude.shape))
    for _ in range(n_iters):
        Zxx = magnitude * angles
        _, audio_est = istft(Zxx, fs=sr, window=window, nperseg=win,
                             noverlap=win - hop, nfft=n_fft)
        _, _, Zxx_new = stft(audio_est, fs=sr, window=window, nperseg=win,
                             noverlap=win - hop, nfft=n_fft, padded=True)
        angles = np.exp(1j * np.angle(Zxx_new))

    # Final synthesis
    Zxx_final = magnitude * angles
    _, audio_out = istft(Zxx_final, fs=sr, window=window, nperseg=win,
                         noverlap=win - hop, nfft=n_fft)
    return audio_out.astype(np.float32)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    # ---- Paths ----
    wav_path = os.path.join(_ROOT, "examples", "0.wav")
    if not os.path.exists(wav_path):
        # Fall back to any wav in examples/
        examples_dir = os.path.join(_ROOT, "examples")
        wavs = [f for f in os.listdir(examples_dir) if f.endswith(".wav")]
        if not wavs:
            raise FileNotFoundError("No .wav files found in examples/")
        wav_path = os.path.join(examples_dir, sorted(wavs)[0])

    out_dir = os.path.join(_HERE, "examples")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "transformed_0.wav")

    # ---- Feature extraction ----
    mel_spec, sr = extract_mel_spectrogram(wav_path, n_mels=N_MELS)
    # mel_spec: (n_frames, 80)

    # ---- Model ----
    model = PrivacyModule(input_dim=N_MELS, latent_dim=64, n_attributes=4)
    model.eval()

    src_idx = ATTRIBUTE_MAP["male_old"]      # 0
    tgt_idx = ATTRIBUTE_MAP["female_young"]  # 3

    features_t = torch.tensor(mel_spec, dtype=torch.float32)  # (n_frames, 80)
    src_t = torch.full((features_t.shape[0],), src_idx, dtype=torch.long)
    tgt_t = torch.full((features_t.shape[0],), tgt_idx, dtype=torch.long)

    with torch.no_grad():
        transformed_t = model(features_t, src_t, tgt_t)  # (n_frames, 80)

    transformed = transformed_t.numpy()  # (n_frames, 80)

    # ---- Reconstruct audio ----
    audio_out = griffin_lim_from_mel(transformed, sr, n_mels=N_MELS)

    # Normalise to int16 range
    peak = np.max(np.abs(audio_out))
    if peak > 0:
        audio_out = audio_out / peak * 0.9
    audio_int16 = (audio_out * 32767).astype(np.int16)

    wavfile.write(out_path, sr, audio_int16)

    # ---- Summary ----
    print("=" * 50)
    print("Privacy-Preserving Transformation Demo")
    print("=" * 50)
    print(f"  Input wav      : {wav_path}")
    print(f"  Input shape    : {mel_spec.shape}  (n_frames x n_mels)")
    print(f"  Output shape   : {transformed.shape}  (n_frames x n_mels)")
    print(f"  Source attr    : male_old  (index {src_idx})")
    print(f"  Target attr    : female_young  (index {tgt_idx})")
    print(f"  Output wav     : {out_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
