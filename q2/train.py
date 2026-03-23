"""
Disentangled Speaker Representation Model
Based on: "Disentangled Representation Learning for Environment-agnostic Speaker Recognition"
arxiv 2406.14559
"""

import os
import logging
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class ContentEncoder(nn.Module):
    """3-layer MLP: input_dim → 512 → 512 → content_dim (ReLU activations)."""

    def __init__(self, input_dim: int, content_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, content_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpeakerEncoder(nn.Module):
    """3-layer MLP: input_dim → 512 → 256 → speaker_dim (ReLU activations)."""

    def __init__(self, input_dim: int, speaker_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, speaker_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """3-layer MLP: (content_dim + speaker_dim) → decoder_hidden → decoder_hidden → output_dim."""

    def __init__(self, content_dim: int, speaker_dim: int, decoder_hidden: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(content_dim + speaker_dim, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, output_dim),
        )

    def forward(self, content_emb: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([content_emb, speaker_emb], dim=-1)
        return self.net(combined)


class SpeakerClassifier(nn.Module):
    """Linear classifier: speaker_dim → n_speakers."""

    def __init__(self, speaker_dim: int, n_speakers: int):
        super().__init__()
        self.fc = nn.Linear(speaker_dim, n_speakers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DisentangledModel(nn.Module):
    """
    Full disentangled speaker representation model.
    forward(x) → (content_emb, speaker_emb, recon)
    recon has the same shape as x.
    """

    def __init__(
        self,
        input_dim: int,
        content_dim: int,
        speaker_dim: int,
        decoder_hidden: int,
        n_speakers: int,
    ):
        super().__init__()
        self.content_encoder = ContentEncoder(input_dim, content_dim)
        self.speaker_encoder = SpeakerEncoder(input_dim, speaker_dim)
        self.decoder = Decoder(content_dim, speaker_dim, decoder_hidden, input_dim)
        self.speaker_classifier = SpeakerClassifier(speaker_dim, n_speakers)
        self.content_classifier = SpeakerClassifier(content_dim, n_speakers)

    def forward(self, x: torch.Tensor):
        content_emb = self.content_encoder(x)
        speaker_emb = self.speaker_encoder(x)
        recon = self.decoder(content_emb, speaker_emb)
        return content_emb, speaker_emb, recon


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    content_emb: torch.Tensor,
    speaker_emb: torch.Tensor,
    speaker_labels: torch.Tensor,
    speaker_classifier: SpeakerClassifier,
    lambda_dis: float = 0.1,
    lambda_cls: float = 1.0,
    content_classifier: SpeakerClassifier = None,
):
    """
    Combined loss:
      recon_loss  = L1(recon, target)
      cls_loss    = CE(speaker_classifier(speaker_emb), labels)
      dis_loss    = -CE(content_classifier(content_emb.detach()), labels)
      total       = recon_loss + lambda_cls * cls_loss + lambda_dis * dis_loss
    """
    recon_loss = F.l1_loss(recon, target)
    cls_loss = F.cross_entropy(speaker_classifier(speaker_emb), speaker_labels)
    # Use content_classifier if provided, otherwise fall back to speaker_classifier
    clf = content_classifier if content_classifier is not None else speaker_classifier
    dis_loss = -F.cross_entropy(clf(content_emb.detach()), speaker_labels)
    total = recon_loss + lambda_cls * cls_loss + lambda_dis * dis_loss
    return total, {
        "recon": recon_loss.item(),
        "cls": cls_loss.item(),
        "dis": dis_loss.item(),
    }


# ---------------------------------------------------------------------------
# Model / dataloader builders
# ---------------------------------------------------------------------------

def build_model(config: dict) -> DisentangledModel:
    """Instantiate DisentangledModel from a config dict."""
    m = config["model"]
    return DisentangledModel(
        input_dim=m["input_dim"],
        content_dim=m["content_dim"],
        speaker_dim=m["speaker_dim"],
        decoder_hidden=m["decoder_hidden"],
        n_speakers=m["n_speakers"],
    )


def build_dataloaders(config: dict):
    """
    Build train/val DataLoaders.

    Tries to load LibriSpeech via torchaudio. Falls back to a synthetic
    dataset (random tensors) when the real data is unavailable.

    Returns: (train_loader, val_loader)
    """
    m = config["model"]
    t = config["training"]
    input_dim: int = m["input_dim"]
    n_speakers: int = m["n_speakers"]
    batch_size: int = t["batch_size"]
    dataset_name: str = t.get("dataset", "librispeech")
    data_dir: str = t.get("data_dir", "data/librispeech")

    dataset = None

    # --- attempt real dataset ---
    if dataset_name == "librispeech":
        try:
            import torchaudio
            logger.info("Attempting to load LibriSpeech from %s …", data_dir)
            raw = torchaudio.datasets.LIBRISPEECH(data_dir, url="train-clean-100", download=False)

            # Build a flat tensor dataset: extract mean mel features per utterance
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_mels=input_dim
            )
            features, labels = [], []
            speaker_id_map: dict = {}
            for waveform, sr, _, speaker_id, *_ in raw:
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(0, keepdim=True)
                mel = mel_transform(waveform).squeeze(0).mean(-1)  # (n_mels,)
                sid = speaker_id_map.setdefault(speaker_id, len(speaker_id_map))
                features.append(mel)
                labels.append(sid)
                if len(features) >= batch_size * 20:
                    break  # cap for speed

            if features:
                X = torch.stack(features)
                y = torch.tensor(labels, dtype=torch.long)
                dataset = TensorDataset(X, y)
                logger.info("Loaded %d LibriSpeech utterances.", len(features))
        except Exception as exc:
            logger.warning("LibriSpeech unavailable (%s). Using synthetic data.", exc)

    # --- synthetic fallback ---
    if dataset is None:
        logger.info("Generating synthetic dataset (input_dim=%d, n_speakers=%d).", input_dim, n_speakers)
        n_samples = batch_size * 10
        X = torch.randn(n_samples, input_dim)
        y = torch.randint(0, n_speakers, (n_samples,))
        dataset = TensorDataset(X, y)

    # 80/20 train/val split
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: DisentangledModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    speaker_classifier: SpeakerClassifier,
    config: dict,
) -> dict:
    """Run one training epoch. Returns dict of average losses."""
    model.train()
    speaker_classifier.train()

    loss_cfg = config.get("loss", {})
    lambda_dis: float = loss_cfg.get("lambda_dis", 0.1)
    lambda_cls: float = loss_cfg.get("lambda_cls", 1.0)

    totals: dict = {"recon": 0.0, "cls": 0.0, "dis": 0.0, "total": 0.0}
    n_batches = 0

    for features, labels in loader:
        optimizer.zero_grad()
        content_emb, speaker_emb, recon = model(features)
        total, components = compute_loss(
            recon, features, content_emb, speaker_emb, labels,
            speaker_classifier, lambda_dis=lambda_dis, lambda_cls=lambda_cls,
            content_classifier=model.content_classifier,
        )
        total.backward()
        optimizer.step()

        totals["total"] += total.item()
        for k, v in components.items():
            totals[k] += v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    """Load YAML config, train model, save best checkpoint."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    t = config["training"]
    n_epochs: int = t["n_epochs"]
    lr: float = t["lr"]
    checkpoint_dir: str = t["checkpoint_dir"]
    val_interval: int = t.get("val_interval", 5)
    log_interval: int = t.get("log_interval", 10)

    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = build_model(config).to(device)
    speaker_classifier = model.speaker_classifier  # shared reference

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(speaker_classifier.parameters()), lr=lr
    )

    train_loader, val_loader = build_dataloaders(config)
    logger.info("Train batches: %d  Val batches: %d", len(train_loader), len(val_loader))

    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        train_losses = train_epoch(model, train_loader, optimizer, speaker_classifier, config)

        if epoch % log_interval == 0 or epoch == 1:
            logger.info(
                "Epoch %d/%d — train total=%.4f recon=%.4f cls=%.4f dis=%.4f",
                epoch, n_epochs,
                train_losses["total"], train_losses["recon"],
                train_losses["cls"], train_losses["dis"],
            )

        if epoch % val_interval == 0 or epoch == n_epochs:
            model.eval()
            val_totals: dict = {"recon": 0.0, "cls": 0.0, "dis": 0.0, "total": 0.0}
            n_val = 0
            loss_cfg = config.get("loss", {})
            lambda_dis = loss_cfg.get("lambda_dis", 0.1)
            lambda_cls = loss_cfg.get("lambda_cls", 1.0)

            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    content_emb, speaker_emb, recon = model(features)
                    total, components = compute_loss(
                        recon, features, content_emb, speaker_emb, labels,
                        speaker_classifier, lambda_dis=lambda_dis, lambda_cls=lambda_cls,
                        content_classifier=model.content_classifier,
                    )
                    val_totals["total"] += total.item()
                    for k, v in components.items():
                        val_totals[k] += v
                    n_val += 1

            avg_val = {k: v / max(n_val, 1) for k, v in val_totals.items()}
            logger.info(
                "Epoch %d/%d — val   total=%.4f recon=%.4f cls=%.4f dis=%.4f",
                epoch, n_epochs,
                avg_val["total"], avg_val["recon"],
                avg_val["cls"], avg_val["dis"],
            )

            if avg_val["total"] < best_val_loss:
                best_val_loss = avg_val["total"]
                ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "config": config,
                    },
                    ckpt_path,
                )
                logger.info("Saved best checkpoint → %s (val_loss=%.4f)", ckpt_path, best_val_loss)

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)


if __name__ == "__main__":
    main("q2/configs/model.yaml")
