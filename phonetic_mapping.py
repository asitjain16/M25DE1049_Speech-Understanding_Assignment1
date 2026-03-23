"""
phonetic_mapping.py — Phonetic boundary mapping and forced-alignment comparison.

Wraps the Wav2TextGrid forced aligner, parses TextGrid output, and computes
RMSE between manual cepstral boundaries and forced-alignment reference boundaries.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Add src/ to sys.path so Wav2TextGrid can be imported
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ---------------------------------------------------------------------------
# Optional Wav2TextGrid import — fail gracefully
# ---------------------------------------------------------------------------
try:
    from Wav2TextGrid import align_file as _wav2tg_align_file
    from Wav2TextGrid.aligner_core.aligner import xVecSAT_forced_aligner
    from Wav2TextGrid.aligner_core.xvec_extractor import xVecExtractor
    _WAV2TG_AVAILABLE = True
except ImportError as _e:
    _WAV2TG_AVAILABLE = False
    _WAV2TG_IMPORT_ERROR = str(_e)

# ---------------------------------------------------------------------------
# Optional praatio import
# ---------------------------------------------------------------------------
try:
    from praatio import textgrid as praatio_tg
    _PRAATIO_AVAILABLE = True
except ImportError:
    _PRAATIO_AVAILABLE = False

# ---------------------------------------------------------------------------
# Import Segment from voiced_unvoiced
# ---------------------------------------------------------------------------
from voiced_unvoiced import Segment, detect_boundaries

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. run_forced_alignment
# ---------------------------------------------------------------------------

def run_forced_alignment(
    wav_path: str,
    lab_path: str,
    out_textgrid: str,
) -> Optional[str]:
    """Wrap Wav2TextGrid align_file() to produce a TextGrid from wav + lab.

    Parameters
    ----------
    wav_path      : path to the WAV audio file.
    lab_path      : path to the .lab transcript file.
    out_textgrid  : destination path for the output TextGrid file.

    Returns
    -------
    out_textgrid path on success, or None if alignment could not be run.
    """
    if not _WAV2TG_AVAILABLE:
        logger.error(
            "Wav2TextGrid package is not available: %s. "
            "Ensure src/Wav2TextGrid is installed and its dependencies are met.",
            _WAV2TG_IMPORT_ERROR,
        )
        return None

    if not os.path.exists(lab_path):
        logger.warning(
            "Lab file not found: %s — skipping forced alignment for %s.",
            lab_path,
            wav_path,
        )
        return None

    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    # Create output directory if needed
    out_dir = os.path.dirname(out_textgrid)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    logger.info("Initialising xVecExtractor …")
    xvec_ext = xVecExtractor(method="xvector")

    logger.info("Initialising xVecSAT_forced_aligner …")
    aligner = xVecSAT_forced_aligner("pkadambi/Wav2TextGrid", satvector_size=512)

    logger.info("Running forced alignment: %s → %s", wav_path, out_textgrid)
    _wav2tg_align_file(
        wav_path,
        lab_path,
        out_textgrid,
        xvec_extractor=xvec_ext,
        forced_aligner=aligner,
        use_speaker_adaptation=True,
    )

    return out_textgrid


# ---------------------------------------------------------------------------
# 2. parse_textgrid
# ---------------------------------------------------------------------------

def parse_textgrid(textgrid_path: str) -> List[Segment]:
    """Parse a TextGrid file and return a list of Segment objects.

    Uses the "phones" tier if present, otherwise falls back to the first
    available interval tier.

    Parameters
    ----------
    textgrid_path : path to the .TextGrid file.

    Returns
    -------
    List of :class:`~voiced_unvoiced.Segment` objects, one per interval.
    """
    if not _PRAATIO_AVAILABLE:
        raise ImportError(
            "praatio is required for parse_textgrid. Install it with: pip install praatio"
        )

    if not os.path.exists(textgrid_path):
        raise FileNotFoundError(f"TextGrid file not found: {textgrid_path}")

    tg = praatio_tg.openTextgrid(textgrid_path, includeEmptyIntervals=True)

    # Prefer the "phones" tier; fall back to first available tier
    tier_names = tg.tierNames
    if "phones" in tier_names:
        tier = tg.getTier("phones")
    elif tier_names:
        tier = tg.getTier(tier_names[0])
    else:
        logger.warning("TextGrid %s has no tiers.", textgrid_path)
        return []

    segments: List[Segment] = []
    for entry in tier.entries:
        start_time = float(entry.start)
        end_time = float(entry.end)
        label = str(entry.label).strip()
        if end_time > start_time:  # skip zero-duration intervals
            segments.append(Segment(start_time=start_time, end_time=end_time, label=label))

    return segments


# ---------------------------------------------------------------------------
# 3. align_segments
# ---------------------------------------------------------------------------

def align_segments(
    manual: List[Segment],
    reference: List[Segment],
) -> List[Tuple[Segment, Segment]]:
    """Align manual segments to reference segments by nearest boundary time.

    For each manual segment, find the reference segment whose start_time is
    closest to the manual segment's start_time.

    Parameters
    ----------
    manual    : list of manually detected segments.
    reference : list of forced-alignment reference segments.

    Returns
    -------
    List of (manual_seg, ref_seg) pairs.
    """
    if not manual or not reference:
        return []

    ref_starts = np.array([s.start_time for s in reference])
    pairs: List[Tuple[Segment, Segment]] = []

    for man_seg in manual:
        diffs = np.abs(ref_starts - man_seg.start_time)
        nearest_idx = int(np.argmin(diffs))
        pairs.append((man_seg, reference[nearest_idx]))

    return pairs


# ---------------------------------------------------------------------------
# 4. compute_rmse
# ---------------------------------------------------------------------------

def compute_rmse(
    manual: List[Segment],
    reference: List[Segment],
) -> float:
    """Compute RMSE of boundary times between manual and reference segments.

    Extracts all boundary times (start and end) from both lists, aligns each
    manual boundary to the nearest reference boundary, then computes
    sqrt(mean(squared_differences)).

    Parameters
    ----------
    manual    : list of manually detected segments.
    reference : list of forced-alignment reference segments.

    Returns
    -------
    RMSE in seconds (non-negative float). Returns 0.0 if either list is empty.
    """
    if not manual or not reference:
        return 0.0

    # Collect all boundary times from each list
    manual_boundaries = sorted(
        {s.start_time for s in manual} | {s.end_time for s in manual}
    )
    ref_boundaries = sorted(
        {s.start_time for s in reference} | {s.end_time for s in reference}
    )

    ref_arr = np.array(ref_boundaries)
    squared_diffs: List[float] = []

    for t in manual_boundaries:
        nearest = ref_arr[int(np.argmin(np.abs(ref_arr - t)))]
        squared_diffs.append((t - nearest) ** 2)

    rmse = math.sqrt(float(np.mean(squared_diffs)))
    return rmse


# ---------------------------------------------------------------------------
# 5. run_q1_pipeline
# ---------------------------------------------------------------------------

def run_q1_pipeline(
    wav_path: str,
    lab_path: str,
    out_dir: str,
) -> dict:
    """Run the full Q1 pipeline for one audio file.

    Steps
    -----
    1. Detect manual boundaries via cepstral analysis (voiced_unvoiced.py).
    2. Run forced alignment to produce a TextGrid.
    3. Parse the TextGrid to get reference segments.
    4. Compute RMSE between manual and reference boundaries.

    Parameters
    ----------
    wav_path : path to the WAV file.
    lab_path : path to the .lab transcript file.
    out_dir  : output directory; TextGrids are saved to ``out_dir/textgrids/``.

    Returns
    -------
    dict with keys:
        "manual_segments"    : List[Segment]
        "reference_segments" : List[Segment]  (empty if alignment failed)
        "rmse"               : float
    """
    # 1. Manual boundary detection
    logger.info("Detecting manual boundaries for %s …", wav_path)
    manual_segments = detect_boundaries(wav_path)

    # 2. Forced alignment
    basename = os.path.splitext(os.path.basename(wav_path))[0]
    textgrid_dir = os.path.join(out_dir, "textgrids")
    out_textgrid = os.path.join(textgrid_dir, f"{basename}.TextGrid")

    tg_path = run_forced_alignment(wav_path, lab_path, out_textgrid)

    # 3. Parse TextGrid
    reference_segments: List[Segment] = []
    if tg_path and os.path.exists(tg_path):
        try:
            reference_segments = parse_textgrid(tg_path)
        except Exception as exc:
            logger.warning("Could not parse TextGrid %s: %s", tg_path, exc)
    else:
        logger.warning(
            "No TextGrid produced for %s — RMSE will be computed against empty reference.",
            wav_path,
        )

    # 4. RMSE
    rmse = compute_rmse(manual_segments, reference_segments)

    return {
        "manual_segments": manual_segments,
        "reference_segments": reference_segments,
        "rmse": rmse,
    }


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    EXAMPLES = [
        ("examples/0.wav", "examples/0.lab"),
        ("examples/1.wav", "examples/1.lab"),
    ]
    OUT_DIR = "data"

    for wav, lab in EXAMPLES:
        print(f"\n{'='*60}")
        print(f"Processing: {wav}")
        print(f"{'='*60}")
        result = run_q1_pipeline(wav, lab, OUT_DIR)

        manual = result["manual_segments"]
        reference = result["reference_segments"]
        rmse = result["rmse"]

        print(f"  Manual segments   : {len(manual)}")
        print(f"  Reference segments: {len(reference)}")
        print(f"  RMSE              : {rmse:.4f} s")
