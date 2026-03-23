"""
q3/audit.py — Bias audit and documentation debt analysis for Common Voice dataset.

Audits the Mozilla Common Voice dataset for representation bias across gender,
age, and dialect groups, and detects documentation debt (missing/unlabeled metadata).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BiasReport:
    """Summary of representation statistics and documentation debt."""

    gender_distribution: dict  # {label: fraction}
    age_distribution: dict
    dialect_distribution: dict
    documentation_debt_items: list  # list of debt description strings
    underrepresented_groups: list   # groups with < 10% representation


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

_GENDER_CATEGORIES = ["male", "female", "other", "unknown"]
_AGE_CATEGORIES = ["teens", "twenties", "thirties", "forties", "fifties",
                   "sixties", "seventies", "eighties", "nineties", "unknown"]
_ACCENT_CATEGORIES = [
    "us", "england", "australia", "canada", "india", "scotland",
    "ireland", "newzealand", "wales", "singapore", "unknown",
]


def _synthetic_dataframe(n: int = 5000) -> pd.DataFrame:
    """Return a synthetic DataFrame that mimics Common Voice metadata columns."""
    rng = np.random.default_rng(42)

    gender_weights = [0.68, 0.25, 0.04, 0.03]
    age_weights    = [0.05, 0.30, 0.25, 0.15, 0.10, 0.07, 0.04, 0.02, 0.01, 0.01]
    accent_weights = [0.30, 0.20, 0.10, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02]

    # Normalise weights (in case they don't sum to 1 exactly)
    gw = np.array(gender_weights) / sum(gender_weights)
    aw = np.array(age_weights)    / sum(age_weights)
    dw = np.array(accent_weights) / sum(accent_weights)

    genders = rng.choice(_GENDER_CATEGORIES, size=n, p=gw)
    ages    = rng.choice(_AGE_CATEGORIES,    size=n, p=aw)
    accents = rng.choice(_ACCENT_CATEGORIES, size=n, p=dw)

    # Inject some missing values to simulate real-world sparsity
    for arr in (genders, ages, accents):
        missing_idx = rng.choice(n, size=int(n * 0.05), replace=False)
        arr[missing_idx] = ""

    return pd.DataFrame({"gender": genders, "age": ages, "accent": accents})


def load_dataset_metadata(
    dataset_name: str = "mozilla-foundation/common_voice_11_0",
    split: str = "train",
    max_samples: int = 5000,
) -> pd.DataFrame:
    """Load metadata from Common Voice via HuggingFace datasets.

    Falls back to a synthetic DataFrame if the dataset requires authentication
    or any other error occurs.  Always returns a DataFrame with columns:
    ``gender``, ``age``, ``accent``.
    """
    try:
        from datasets import load_dataset  # type: ignore

        print(f"[audit] Loading '{dataset_name}' ({split}) from HuggingFace …")
        ds = load_dataset(
            dataset_name,
            "en",
            split=split,
            trust_remote_code=True,
            streaming=True,
        )

        records = []
        for i, sample in enumerate(ds):
            if i >= max_samples:
                break
            records.append({
                "gender": sample.get("gender", ""),
                "age":    sample.get("age",    ""),
                "accent": sample.get("accent", ""),
            })

        if not records:
            raise ValueError("Dataset returned no records.")

        df = pd.DataFrame(records)
        print(f"[audit] Loaded {len(df)} samples from HuggingFace.")
        return df

    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"[audit] Could not load '{dataset_name}' ({exc}). "
            "Falling back to synthetic data.",
            stacklevel=2,
        )
        df = _synthetic_dataframe(max_samples)
        print(f"[audit] Generated {len(df)} synthetic samples.")
        return df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _normalised_distribution(series: pd.Series) -> dict:
    """Return a {label: fraction} dict for *series*, treating blanks as 'unknown'."""
    cleaned = series.copy()
    cleaned = cleaned.replace("", "unknown").fillna("unknown")
    cleaned = cleaned.str.strip().replace("", "unknown")
    counts = cleaned.value_counts(normalize=True)
    return {str(k): float(v) for k, v in counts.items()}


def compute_representation_stats(df: pd.DataFrame) -> BiasReport:
    """Compute normalised distributions for gender, age, and accent columns.

    Missing / empty values are marked as ``"unknown"``.
    Groups with < 10 % representation are listed in ``underrepresented_groups``.
    """
    gender_dist  = _normalised_distribution(df["gender"])
    age_dist     = _normalised_distribution(df["age"])
    dialect_dist = _normalised_distribution(df["accent"])

    debt_items = detect_documentation_debt(df)

    # Collect underrepresented groups (< 10 % in any distribution)
    underrepresented: list[str] = []
    for dist_name, dist in [
        ("gender",  gender_dist),
        ("age",     age_dist),
        ("dialect", dialect_dist),
    ]:
        for label, fraction in dist.items():
            if fraction < 0.10:
                underrepresented.append(f"{dist_name}:{label} ({fraction:.1%})")

    return BiasReport(
        gender_distribution=gender_dist,
        age_distribution=age_dist,
        dialect_distribution=dialect_dist,
        documentation_debt_items=debt_items,
        underrepresented_groups=underrepresented,
    )


# ---------------------------------------------------------------------------
# Documentation debt detection
# ---------------------------------------------------------------------------

def detect_documentation_debt(df: pd.DataFrame) -> list:
    """Detect documentation debt in the metadata DataFrame.

    Checks for:
    - Missing fields (NaN / empty strings)
    - Unlabeled / unknown categories
    - Imbalanced groups (< 5 % representation)

    Returns a list of human-readable debt description strings.
    """
    debt: list[str] = []

    for col in ("gender", "age", "accent"):
        if col not in df.columns:
            debt.append(f"Column '{col}' is entirely absent from the dataset.")
            continue

        # Missing / NaN values
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            pct = n_nan / len(df) * 100
            debt.append(
                f"'{col}': {n_nan} NaN values ({pct:.1f}% of samples) — missing metadata."
            )

        # Empty-string values
        n_empty = (df[col].fillna("").str.strip() == "").sum()
        if n_empty > 0:
            pct = n_empty / len(df) * 100
            debt.append(
                f"'{col}': {n_empty} empty-string values ({pct:.1f}% of samples) — unlabeled entries."
            )

        # Imbalanced groups (< 5 %)
        dist = _normalised_distribution(df[col])
        for label, fraction in dist.items():
            if fraction < 0.05:
                debt.append(
                    f"'{col}' group '{label}' has only {fraction:.1%} representation "
                    f"(< 5% threshold) — imbalanced group."
                )

    return debt


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _bar_chart(
    distribution: dict,
    title: str,
    xlabel: str,
    out_path: str,
) -> None:
    """Save a single bar chart for *distribution* to *out_path*."""
    labels  = list(distribution.keys())
    values  = [distribution[k] * 100 for k in labels]  # convert to %

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 4))
    bars = ax.bar(labels, values, color="steelblue", edgecolor="white")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    ax.tick_params(axis="x", rotation=30)

    # Annotate bars with percentage labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[audit] Saved plot → {out_path}")


def generate_audit_plots(report: BiasReport, out_dir: str = "q3/results") -> None:
    """Save three bar charts (gender, age, dialect distributions) to *out_dir*."""
    os.makedirs(out_dir, exist_ok=True)

    _bar_chart(
        report.gender_distribution,
        title="Gender Distribution",
        xlabel="Gender",
        out_path=os.path.join(out_dir, "gender_distribution.png"),
    )
    _bar_chart(
        report.age_distribution,
        title="Age Distribution",
        xlabel="Age Group",
        out_path=os.path.join(out_dir, "age_distribution.png"),
    )
    _bar_chart(
        report.dialect_distribution,
        title="Dialect / Accent Distribution",
        xlabel="Dialect",
        out_path=os.path.join(out_dir, "dialect_distribution.png"),
    )


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load metadata
    df = load_dataset_metadata()

    # 2. Compute representation statistics
    report = compute_representation_stats(df)

    # 3. Detect documentation debt (already embedded in report, but print separately)
    debt_items = detect_documentation_debt(df)

    # 4. Generate plots
    generate_audit_plots(report, out_dir="q3/results")

    # 5. Print summary
    print("\n" + "=" * 60)
    print("BIAS AUDIT SUMMARY")
    print("=" * 60)

    print("\n--- Gender Distribution ---")
    for label, frac in sorted(report.gender_distribution.items(), key=lambda x: -x[1]):
        print(f"  {label:<20} {frac:>6.1%}")

    print("\n--- Age Distribution ---")
    for label, frac in sorted(report.age_distribution.items(), key=lambda x: -x[1]):
        print(f"  {label:<20} {frac:>6.1%}")

    print("\n--- Dialect Distribution ---")
    for label, frac in sorted(report.dialect_distribution.items(), key=lambda x: -x[1]):
        print(f"  {label:<20} {frac:>6.1%}")

    print(f"\n--- Documentation Debt ({len(debt_items)} items) ---")
    if debt_items:
        for item in debt_items:
            print(f"  • {item}")
    else:
        print("  (none)")

    print(f"\n--- Underrepresented Groups (< 10%) — {len(report.underrepresented_groups)} total ---")
    for grp in report.underrepresented_groups:
        print(f"  • {grp}")

    print("\n[audit] Done.")
