"""
leakage_snr.py — Spectral leakage and SNR analysis for three window functions.

Measures the effect of rectangular, Hamming, and Hanning windows on spectral
quality when applied to a speech segment.
"""

import os
import numpy as np
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Core DSP functions
# ---------------------------------------------------------------------------

def compute_power_spectrum(signal: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Apply window to signal, compute power spectrum via FFT.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Input signal (float).
    window : np.ndarray, shape (N,)
        Window coefficients (same length as signal).

    Returns
    -------
    np.ndarray, shape (n_fft//2 + 1,)
        One-sided power spectrum in dB (reference: 1.0).
    """
    windowed = signal * window
    n_fft = len(windowed)
    spectrum = np.fft.rfft(windowed, n=n_fft)
    power = (np.abs(spectrum) ** 2) / n_fft  # normalise by FFT length
    # Convert to dB; clip to avoid log(0)
    power_db = 10.0 * np.log10(np.maximum(power, 1e-12))
    return power_db


def compute_snr(
    signal: np.ndarray,
    window: np.ndarray,
    noise_floor_db: float = -60.0,
) -> float:
    """Compute SNR in dB for a windowed signal.

    The signal bin is the peak bin in the power spectrum.  Noise is defined as
    all other bins whose power is above *noise_floor_db*.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    window : np.ndarray
        Window coefficients (same length as signal).
    noise_floor_db : float
        Bins below this threshold are excluded from the noise estimate.

    Returns
    -------
    float
        SNR in dB.  Returns 0.0 when no noise bins are found above the floor.
    """
    power_db = compute_power_spectrum(signal, window)
    signal_bin = int(np.argmax(power_db))
    signal_power_db = power_db[signal_bin]

    # Noise: all bins except the signal bin that are above the noise floor
    noise_mask = np.ones(len(power_db), dtype=bool)
    noise_mask[signal_bin] = False
    noise_mask &= power_db > noise_floor_db

    if not np.any(noise_mask):
        return 0.0

    # Average noise power in linear scale, then convert to dB
    noise_power_linear = np.mean(10.0 ** (power_db[noise_mask] / 10.0))
    noise_power_db = 10.0 * np.log10(noise_power_linear)

    return float(signal_power_db - noise_power_db)


def compute_leakage_ratio(power_spectrum: np.ndarray, signal_bin: int) -> float:
    """Ratio of power outside the main lobe to total power.

    The main lobe is defined as ±2 bins around *signal_bin*.

    Parameters
    ----------
    power_spectrum : np.ndarray
        Power spectrum in dB, shape (n_fft//2 + 1,).
    signal_bin : int
        Index of the dominant frequency bin.

    Returns
    -------
    float
        Leakage ratio in [0, 1].
    """
    # Convert dB back to linear power for ratio computation
    power_linear = 10.0 ** (power_spectrum / 10.0)
    total_power = np.sum(power_linear)

    if total_power == 0.0:
        return 0.0

    # Main lobe: signal_bin ± 2 bins (clamped to valid range)
    lobe_start = max(0, signal_bin - 2)
    lobe_end = min(len(power_linear) - 1, signal_bin + 2)
    main_lobe_power = np.sum(power_linear[lobe_start : lobe_end + 1])

    leakage_power = total_power - main_lobe_power
    return float(np.clip(leakage_power / total_power, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Window factory
# ---------------------------------------------------------------------------

def _make_window(window_type: str, length: int) -> np.ndarray:
    """Return a window array of the requested type."""
    wt = window_type.lower()
    if wt == "rectangular":
        return np.ones(length, dtype=np.float64)
    elif wt == "hamming":
        return np.hamming(length)
    elif wt == "hanning":
        return np.hanning(length)
    else:
        raise ValueError(f"Unknown window type: {window_type!r}. "
                         "Choose from 'rectangular', 'hamming', 'hanning'.")


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_windows(
    signal: np.ndarray,
    sr: int,
    window_types: list = None,
) -> dict:
    """Analyse spectral leakage and SNR for multiple window functions.

    For each window type the function computes the power spectrum, SNR, and
    leakage ratio, prints a comparison table to stdout, and saves plots to
    ``data/leakage_plots/``.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1-D float array).
    sr : int
        Sample rate (Hz) — used for frequency axis labelling.
    window_types : list[str], optional
        Window names to analyse.  Defaults to
        ``["rectangular", "hamming", "hanning"]``.

    Returns
    -------
    dict
        ``{window_name: {"snr_db": float, "leakage_ratio": float}, ...}``
    """
    if window_types is None:
        window_types = ["rectangular", "hamming", "hanning"]

    n = len(signal)
    results: dict = {}
    spectra: dict = {}

    for wt in window_types:
        window = _make_window(wt, n)
        ps = compute_power_spectrum(signal, window)
        signal_bin = int(np.argmax(ps))
        snr = compute_snr(signal, window)
        leakage = compute_leakage_ratio(ps, signal_bin)
        results[wt] = {"snr_db": snr, "leakage_ratio": leakage}
        spectra[wt] = ps

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    header = f"{'Window':<14} {'SNR (dB)':>10} {'Leakage Ratio':>15}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for wt in window_types:
        snr = results[wt]["snr_db"]
        lr = results[wt]["leakage_ratio"]
        print(f"{wt:<14} {snr:>10.2f} {lr:>15.4f}")
    print(sep)

    # ------------------------------------------------------------------
    # Save plots
    # ------------------------------------------------------------------
    out_dir = os.path.join("data", "leakage_plots")
    os.makedirs(out_dir, exist_ok=True)

    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    # Figure 1: power spectra for each window (3 subplots)
    fig, axes = plt.subplots(len(window_types), 1, figsize=(10, 3 * len(window_types)),
                             sharex=True)
    if len(window_types) == 1:
        axes = [axes]

    for ax, wt in zip(axes, window_types):
        ax.plot(freqs, spectra[wt], linewidth=0.8)
        ax.set_title(f"{wt.capitalize()} window  |  "
                     f"SNR={results[wt]['snr_db']:.1f} dB  |  "
                     f"Leakage={results[wt]['leakage_ratio']:.4f}")
        ax.set_ylabel("Power (dB)")
        ax.set_ylim(bottom=-100)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Power Spectra — Window Comparison", fontsize=13, y=1.01)
    fig.tight_layout()
    spectra_path = os.path.join(out_dir, "power_spectra.png")
    fig.savefig(spectra_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved power spectra plot → {spectra_path}")

    # Figure 2: bar chart comparing SNR and leakage ratio
    fig2, (ax_snr, ax_lr) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(window_types))
    snr_vals = [results[wt]["snr_db"] for wt in window_types]
    lr_vals = [results[wt]["leakage_ratio"] for wt in window_types]
    labels = [wt.capitalize() for wt in window_types]

    ax_snr.bar(x, snr_vals, color=["steelblue", "darkorange", "seagreen"][:len(window_types)])
    ax_snr.set_xticks(x)
    ax_snr.set_xticklabels(labels)
    ax_snr.set_ylabel("SNR (dB)")
    ax_snr.set_title("SNR by Window Type")
    ax_snr.grid(True, axis="y", alpha=0.3)

    ax_lr.bar(x, lr_vals, color=["steelblue", "darkorange", "seagreen"][:len(window_types)])
    ax_lr.set_xticks(x)
    ax_lr.set_xticklabels(labels)
    ax_lr.set_ylabel("Leakage Ratio")
    ax_lr.set_title("Spectral Leakage Ratio by Window Type")
    ax_lr.set_ylim(0, 1)
    ax_lr.grid(True, axis="y", alpha=0.3)

    fig2.suptitle("Window Function Comparison", fontsize=13)
    fig2.tight_layout()
    bar_path = os.path.join(out_dir, "window_comparison_bar.png")
    fig2.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved bar chart → {bar_path}")

    return results


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    wav_path = os.path.join("examples", "0.wav")
    if not os.path.exists(wav_path):
        print(f"Error: {wav_path} not found.", file=sys.stderr)
        sys.exit(1)

    sr, data = wavfile.read(wav_path)

    # Convert to float in [-1, 1]
    if data.dtype == np.int16:
        audio = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float64) / 2147483648.0
    else:
        audio = data.astype(np.float64)

    # Handle stereo — take first channel
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Extract 1-second segment from the middle
    total_samples = len(audio)
    seg_len = sr  # 1 second
    mid = total_samples // 2
    start = max(0, mid - seg_len // 2)
    end = start + seg_len
    if end > total_samples:
        end = total_samples
        start = max(0, end - seg_len)
    segment = audio[start:end]

    print(f"Loaded {wav_path}  |  sr={sr} Hz  |  segment: {start/sr:.2f}s – {end/sr:.2f}s")
    print()

    results = analyze_windows(segment, sr)

    print()
    print("Results dict:")
    for wt, metrics in results.items():
        print(f"  {wt}: {metrics}")
