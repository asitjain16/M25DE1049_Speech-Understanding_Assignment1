"""
Audio quality evaluation for privacy-transformed audio files.

Proxy metrics using numpy/scipy only (no external APIs):
  - SNR proxy: signal-to-noise ratio of transformed vs original
  - Spectral distortion: mean absolute difference of log-magnitude spectra
"""

import json
import os
import glob

import numpy as np
from scipy.io import wavfile


def _load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file, return (float32 signal normalised to [-1,1], sample_rate)."""
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
    # If stereo, take first channel
    if data.ndim > 1:
        data = data[:, 0]
    return data, sr


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim both arrays to the shorter length."""
    n = min(len(a), len(b))
    return a[:n], b[:n]


def compute_snr_proxy(original_wav: str, transformed_wav: str) -> float:
    """
    Compute SNR of transformed audio relative to original.

    SNR = 10 * log10(signal_power / noise_power)
    where noise = original - transformed (after aligning lengths).

    Returns
    -------
    float
        SNR in dB.
    """
    orig, _ = _load_wav(original_wav)
    trans, _ = _load_wav(transformed_wav)
    orig, trans = _align(orig, trans)

    signal_power = np.mean(orig ** 2)
    noise = orig - trans
    noise_power = np.mean(noise ** 2)

    if noise_power == 0.0:
        return float("inf")

    return float(10.0 * np.log10(signal_power / noise_power))


def compute_spectral_distortion(original_wav: str, transformed_wav: str) -> float:
    """
    Compute mean absolute difference of log-magnitude spectra (FAD proxy).

    Uses a single-frame FFT over the full (aligned) signal.

    Returns
    -------
    float
        Mean absolute log-spectral difference.
    """
    orig, _ = _load_wav(original_wav)
    trans, _ = _load_wav(transformed_wav)
    orig, trans = _align(orig, trans)

    eps = 1e-10
    orig_spec = np.abs(np.fft.rfft(orig))
    trans_spec = np.abs(np.fft.rfft(trans))

    log_orig = np.log(orig_spec + eps)
    log_trans = np.log(trans_spec + eps)

    return float(np.mean(np.abs(log_orig - log_trans)))


def evaluate_audio_pair(original_path: str, transformed_path: str) -> dict:
    """
    Run both quality metrics on a single original/transformed pair.

    Returns
    -------
    dict with keys: snr_db, spectral_distortion, original, transformed
    """
    snr = compute_snr_proxy(original_path, transformed_path)
    sd = compute_spectral_distortion(original_path, transformed_path)
    return {
        "snr_db": snr,
        "spectral_distortion": sd,
        "original": original_path,
        "transformed": transformed_path,
    }


def evaluate_directory(examples_dir: str, results_dir: str) -> list[dict]:
    """
    Evaluate all original/transformed pairs in examples_dir.

    Pairs are matched as:
      original   : <examples_dir>/<name>.wav  (not prefixed with "transformed_")
      transformed: <examples_dir>/transformed_<name>.wav

    Prints a results table to stdout and saves
    <results_dir>/quality_metrics.json.

    Parameters
    ----------
    examples_dir : str
        Directory containing WAV files.
    results_dir : str
        Directory where quality_metrics.json will be written.

    Returns
    -------
    list[dict]
        List of per-pair result dicts.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Find all transformed files and derive original counterparts
    transformed_pattern = os.path.join(examples_dir, "transformed_*.wav")
    transformed_files = sorted(glob.glob(transformed_pattern))

    results = []
    for tf in transformed_files:
        basename = os.path.basename(tf)  # e.g. "transformed_0.wav"
        original_name = basename[len("transformed_"):]  # e.g. "0.wav"
        original_path = os.path.join(examples_dir, original_name)

        if not os.path.isfile(original_path):
            print(f"[WARN] Original not found for {tf}, skipping.")
            continue

        result = evaluate_audio_pair(original_path, tf)
        results.append(result)

    # Print table
    if results:
        col_w = 40
        header = f"{'Original':<{col_w}} {'Transformed':<{col_w}} {'SNR (dB)':>10} {'Spec Dist':>12}"
        print(header)
        print("-" * len(header))
        for r in results:
            orig_short = os.path.basename(r["original"])
            trans_short = os.path.basename(r["transformed"])
            snr_str = f"{r['snr_db']:>10.2f}" if np.isfinite(r["snr_db"]) else f"{'inf':>10}"
            print(f"{orig_short:<{col_w}} {trans_short:<{col_w}} {snr_str} {r['spectral_distortion']:>12.4f}")
    else:
        print("No original/transformed pairs found in", examples_dir)

    # Save JSON
    out_path = os.path.join(results_dir, "quality_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    evaluate_directory("q3/examples", "q3/results")
