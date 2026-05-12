import torch
from torchvision import models


EMBEDDING_DIM = 128

FASHIONCLIP_DIM = 512
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_stylist_encoder() -> torch.nn.Module:
    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, EMBEDDING_DIM)
    return model


class FashionCLIPEncoder(torch.nn.Module):
    """Vision encoder backed by Marqo/marqo-fashionCLIP (ViT-B/16, 512-dim)."""

    HF_REPO = "hf-hub:Marqo/marqo-fashionCLIP"

    def __init__(self):
        super().__init__()
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open-clip-torch is required for FashionCLIP.\n"
                "  venv/bin/pip install open-clip-torch"
            )
        model, _, _ = open_clip.create_model_from_pretrained(self.HF_REPO)
        self._clip = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self._clip.encode_image(images)


def build_fashionclip_encoder() -> FashionCLIPEncoder:
    return FashionCLIPEncoder()


class CompatibilityHead(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 4, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        left_embedding: torch.Tensor,
        right_embedding: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                left_embedding,
                right_embedding,
                torch.abs(left_embedding - right_embedding),
                left_embedding * right_embedding,
            ],
            dim=1,
        )
        return self.net(features).squeeze(1)
