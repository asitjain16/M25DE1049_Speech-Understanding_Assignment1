import torch
import torch.nn as nn
from torch import Tensor

ATTRIBUTE_MAP = {
    "male_old": 0,
    "male_young": 1,
    "female_old": 2,
    "female_young": 3,
}


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    mid = (in_dim + out_dim) // 2
    return nn.Sequential(
        nn.Linear(in_dim, mid),
        nn.ReLU(),
        nn.Linear(mid, mid),
        nn.ReLU(),
        nn.Linear(mid, out_dim),
        nn.ReLU(),
    )


class PrivacyModule(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, n_attributes: int):
        super().__init__()
        self.content_encoder = _mlp(input_dim, latent_dim)
        self.attribute_encoder = _mlp(input_dim, latent_dim)
        self.attribute_embedding = nn.Embedding(n_attributes, n_attributes)
        # decoder: (latent_dim + n_attributes) → input_dim, 3-layer MLP + final Linear
        dec_in = latent_dim + n_attributes
        mid = (dec_in + input_dim) // 2
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, mid),
            nn.ReLU(),
            nn.Linear(mid, mid),
            nn.ReLU(),
            nn.Linear(mid, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
        )

    def encode(self, audio_features: Tensor) -> tuple[Tensor, Tensor]:
        content_latent = self.content_encoder(audio_features)
        attribute_latent = self.attribute_encoder(audio_features)
        return content_latent, attribute_latent

    def decode(self, content_latent: Tensor, target_attr: Tensor) -> Tensor:
        attr_emb = self.attribute_embedding(target_attr)          # (B, n_attributes)
        x = torch.cat([content_latent, attr_emb], dim=-1)        # (B, latent_dim + n_attributes)
        return self.decoder(x)                                    # (B, input_dim)

    def forward(self, audio_features: Tensor, source_attr: Tensor, target_attr: Tensor) -> Tensor:
        content_latent, _ = self.encode(audio_features)
        return self.decode(content_latent, target_attr)


if __name__ == "__main__":
    model = PrivacyModule(input_dim=80, latent_dim=64, n_attributes=4)
    x = torch.randn(2, 80)
    src = torch.tensor([0, 0])
    tgt = torch.tensor([3, 3])
    out = model(x, src, tgt)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print("PrivacyModule OK")
