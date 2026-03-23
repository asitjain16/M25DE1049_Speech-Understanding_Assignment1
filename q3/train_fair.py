"""
train_fair.py — Fairness-aware ASR training with a custom FairnessLoss.

Implements:
  - FairnessLoss: penalises variance in per-group cross-entropy losses
  - train_with_fairness: standard training loop using FairnessLoss
  - SimpleASRModel: minimal linear ASR proxy for demonstration
"""

import logging
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FairnessLoss
# ---------------------------------------------------------------------------

class FairnessLoss(nn.Module):
    """Cross-entropy loss with a fairness penalty based on per-group loss variance.

    Args:
        protected_groups: List of group name strings (used for logging).
        lambda_fair: Weight applied to the fairness penalty term.
    """

    def __init__(self, protected_groups: list[str], lambda_fair: float = 0.1):
        super().__init__()
        self.protected_groups = protected_groups
        self.lambda_fair = lambda_fair

    def forward(self, logits: Tensor, targets: Tensor, group_ids: Tensor) -> Tensor:
        """Compute base cross-entropy + lambda_fair * sqrt(variance of per-group losses).

        Args:
            logits:    (N, n_classes) — raw model outputs.
            targets:   (N,)           — integer class labels.
            group_ids: (N,)           — integer group membership for each sample.

        Returns:
            Scalar loss tensor with gradients.
        """
        unique_groups = group_ids.unique()
        group_losses: list[Tensor] = []

        for g in unique_groups:
            mask = group_ids == g
            n_samples = mask.sum().item()

            if n_samples == 0:
                # Determine a human-readable group name if available
                g_idx = g.item()
                name = (
                    self.protected_groups[g_idx]
                    if isinstance(g_idx, int) and g_idx < len(self.protected_groups)
                    else str(g_idx)
                )
                warnings.warn(f"Group '{name}' has 0 samples in this batch — skipping.")
                logger.warning("Group '%s' has 0 samples in this batch — skipping.", name)
                continue

            g_loss = F.cross_entropy(logits[mask], targets[mask])
            group_losses.append(g_loss)

        base_loss = F.cross_entropy(logits, targets)

        if len(group_losses) < 2:
            # Not enough groups to compute a meaningful variance; return base loss only.
            return base_loss

        stacked = torch.stack(group_losses)          # (n_groups,)
        variance = stacked.var(unbiased=False)        # population variance
        fairness_penalty = torch.sqrt(variance + 1e-8)

        return base_loss + self.lambda_fair * fairness_penalty


# ---------------------------------------------------------------------------
# SimpleASRModel
# ---------------------------------------------------------------------------

class SimpleASRModel(nn.Module):
    """Minimal linear ASR proxy: a single fully-connected layer.

    Args:
        input_dim: Number of input features (e.g. 40 MFCCs).
        n_classes:  Number of output classes (e.g. 10 phonemes).
    """

    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_with_fairness(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    fairness_loss: FairnessLoss,
    n_epochs: int = 5,
) -> None:
    """Train *model* using FairnessLoss for *n_epochs* epochs.

    Each batch from *loader* must be a 3-tuple: (features, targets, group_ids).

    Args:
        model:          The ASR model to train.
        loader:         DataLoader yielding (features, targets, group_ids) batches.
        optimizer:      Optimiser (e.g. Adam).
        fairness_loss:  Instantiated FairnessLoss module.
        n_epochs:       Number of full passes over the dataset.
    """
    model.train()

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0

        for features, targets, group_ids in loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = fairness_loss(logits, targets, group_ids)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info("Epoch %d/%d — avg loss: %.4f", epoch, n_epochs, avg_loss)


# ---------------------------------------------------------------------------
# __main__ demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    # Reproducibility
    torch.manual_seed(42)

    # Synthetic dataset: 200 samples, 40-dim features, 10 classes, 3 groups
    N, INPUT_DIM, N_CLASSES, N_GROUPS = 200, 40, 10, 3

    features = torch.randn(N, INPUT_DIM)
    targets = torch.randint(0, N_CLASSES, (N,))
    group_ids = torch.randint(0, N_GROUPS, (N,))

    dataset = TensorDataset(features, targets, group_ids)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model, loss, optimiser
    model = SimpleASRModel(input_dim=INPUT_DIM, n_classes=N_CLASSES)
    protected_groups = ["group_0", "group_1", "group_2"]
    fairness_loss = FairnessLoss(protected_groups=protected_groups, lambda_fair=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train for 3 epochs
    train_with_fairness(model, loader, optimizer, fairness_loss, n_epochs=3)

    # Report final loss on the full dataset (single forward pass)
    model.eval()
    with torch.no_grad():
        logits = model(features)
        final_loss = fairness_loss(logits, targets, group_ids)
    print(f"Final fairness loss: {final_loss.item():.4f}")
