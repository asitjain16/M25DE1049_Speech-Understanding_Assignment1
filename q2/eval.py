"""
Evaluation script for the disentangled speaker representation model.
Computes EER and TAR@FAR metrics from speaker embeddings.
"""

import os
import json

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from q2.train import build_model, build_dataloaders

MAX_EVAL_SAMPLES = 1000


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(model, loader):
    """
    Run model in eval mode and extract speaker embeddings.

    Returns
    -------
    embeddings : np.ndarray, shape (N, speaker_dim)
    labels     : np.ndarray, shape (N,)
    """
    model.eval()
    device = next(model.parameters()).device

    all_embs = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            _, speaker_emb, _ = model(features)
            all_embs.append(speaker_emb.cpu().numpy())
            all_labels.append(labels.numpy())

            if sum(len(e) for e in all_embs) >= MAX_EVAL_SAMPLES:
                break

    embeddings = np.concatenate(all_embs, axis=0)[:MAX_EVAL_SAMPLES]
    labels = np.concatenate(all_labels, axis=0)[:MAX_EVAL_SAMPLES]
    return embeddings, labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _pairwise_scores_and_targets(embeddings, labels):
    """
    Compute cosine similarity scores and binary targets for all pairs.
    Returns (scores, targets) as 1-D arrays.
    """
    # cosine similarity = 1 - cosine distance
    cos_dist = cdist(embeddings, embeddings, metric="cosine")  # (N, N)
    scores = 1.0 - cos_dist  # similarity in [-1, 1]

    n = len(labels)
    # Upper-triangle indices (exclude diagonal)
    idx_i, idx_j = np.triu_indices(n, k=1)
    scores = scores[idx_i, idx_j]
    targets = (labels[idx_i] == labels[idx_j]).astype(int)
    return scores, targets


def compute_eer(embeddings, labels):
    """
    Compute Equal Error Rate (EER) using cosine similarity scores.

    Returns
    -------
    eer : float in [0, 1]
    """
    scores, targets = _pairwise_scores_and_targets(embeddings, labels)

    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    far_list = []
    frr_list = []

    pos_mask = targets == 1
    neg_mask = targets == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    if n_pos == 0 or n_neg == 0:
        return 0.5  # degenerate case

    for thr in thresholds:
        accepted = scores >= thr
        far = accepted[neg_mask].sum() / n_neg   # false accepts / total negatives
        frr = (~accepted[pos_mask]).sum() / n_pos  # false rejects / total positives
        far_list.append(far)
        frr_list.append(frr)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)

    # EER is where FAR ≈ FRR — find crossing point
    diff = np.abs(far_arr - frr_arr)
    idx = diff.argmin()
    eer = float((far_arr[idx] + frr_arr[idx]) / 2.0)
    return eer


def compute_tar_at_far(embeddings, labels, far=0.01):
    """
    Compute True Acceptance Rate at a fixed False Acceptance Rate.

    Returns
    -------
    tar : float in [0, 1]
    """
    scores, targets = _pairwise_scores_and_targets(embeddings, labels)

    pos_mask = targets == 1
    neg_mask = targets == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    if n_pos == 0 or n_neg == 0:
        return 0.0

    thresholds = np.linspace(scores.min(), scores.max(), 1000)

    # Walk from high threshold (strict) to low (lenient); pick highest threshold
    # where FAR <= target_far, then read TAR at that threshold.
    best_tar = 0.0
    for thr in sorted(thresholds, reverse=True):
        accepted = scores >= thr
        current_far = accepted[neg_mask].sum() / n_neg
        if current_far <= far:
            tar = accepted[pos_mask].sum() / n_pos
            best_tar = float(tar)
            break

    return best_tar


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

def run_evaluation(checkpoint_path, config_path):
    """
    Load checkpoint, extract embeddings, compute metrics, save results.

    Saves
    -----
    q2/results/metrics.json
    q2/results/eer_curve.png

    Returns
    -------
    dict with keys "eer" and "tar_at_far_0.01"
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    eval_cfg = config.get("evaluation", {})
    results_dir = eval_cfg.get("results_dir", "q2/results")
    target_far = eval_cfg.get("far", 0.01)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = build_model(config).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()

    # Build val dataloader
    _, val_loader = build_dataloaders(config)

    # Extract embeddings
    embeddings, labels = extract_embeddings(model, val_loader)

    # Compute metrics
    eer = compute_eer(embeddings, labels)
    tar = compute_tar_at_far(embeddings, labels, far=target_far)

    metrics = {"eer": eer, "tar_at_far_0.01": tar}

    # Save metrics.json
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {metrics_path}")
    print(f"  EER          : {eer:.4f}")
    print(f"  TAR@FAR=0.01 : {tar:.4f}")

    # Save EER curve
    _save_eer_curve(embeddings, labels, results_dir)

    return metrics


def _save_eer_curve(embeddings, labels, results_dir):
    """Plot FAR vs FRR curve with EER point and save to results_dir."""
    scores, targets = _pairwise_scores_and_targets(embeddings, labels)

    pos_mask = targets == 1
    neg_mask = targets == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    if n_pos == 0 or n_neg == 0:
        return

    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    far_list, frr_list = [], []

    for thr in thresholds:
        accepted = scores >= thr
        far_list.append(accepted[neg_mask].sum() / n_neg)
        frr_list.append((~accepted[pos_mask]).sum() / n_pos)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)

    idx = np.abs(far_arr - frr_arr).argmin()
    eer_point = (far_arr[idx] + frr_arr[idx]) / 2.0

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(far_arr, frr_arr, label="FAR vs FRR", color="steelblue")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="EER line")
    ax.scatter([far_arr[idx]], [frr_arr[idx]], color="red", zorder=5,
               label=f"EER = {eer_point:.4f}")
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("False Rejection Rate (FRR)")
    ax.set_title("FAR vs FRR Curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    curve_path = os.path.join(results_dir, "eer_curve.png")
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"EER curve saved → {curve_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation("q2/checkpoints/best_model.pt", "q2/configs/model.yaml")
