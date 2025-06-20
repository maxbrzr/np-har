from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor

from np_har.encoder.tiny_har import TinyHAR


class ContextualizerBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_classes: int, num_heads: int, mlp_dim: int
    ) -> None:
        super(ContextualizerBlock, self).__init__()

        self.class_embedding = nn.Embedding(num_classes, embed_dim)

        self.mhsa = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # (batch_size, context_size, embed_dim)

        x += self.class_embedding(x)
        # (batch_size, context_size, embed_dim)

        x, _ = self.mhsa(x, x, x)
        # (batch_size, context_size, embed_dim)

        x = self.mlp(x)
        # (batch_size, context_size, embed_dim)

        return x


class ContextualAggregator(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_heads: int,
        mlp_dim: int,
        num_blocks: int,
    ) -> None:
        super(ContextualAggregator, self).__init__()

        self.contextualizer = nn.Sequential(
            *[
                ContextualizerBlock(
                    embed_dim=embed_dim,
                    num_classes=num_classes,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        # (batch_size, context_size, embed_dim)

        x = self.contextualizer(x)
        # (batch_size, context_size, embed_dim)

        x = torch.mean(x, dim=1)
        # (batch_size, embed_dim)

        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, num_layers: int) -> None:
        super(Decoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: Tensor, phi: Tensor) -> Tensor:
        # (batch_size, embed_dim)
        # (batch_size, embed_dim)

        logits = self.mlp(torch.cat([x, phi], dim=1))
        # (batch_size, num_classes)

        return logits


class NPHAR(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        num_blocks: int,
        num_layers_dec: int,
    ) -> None:
        super(NPHAR, self).__init__()

        self.encoder = TinyHAR(embed_dim=embed_dim)

        self.contextual_aggregator = ContextualAggregator(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_blocks=num_blocks,
        )

        self.decoder = Decoder(
            embed_dim=embed_dim, num_classes=num_classes, num_layers=num_layers_dec
        )

    def forward(self, x: Tensor, context: Tensor) -> Tuple[Tensor, Tensor]:
        # (batch_size, 1, window_size, num_sensors)
        # (batch_size, context_size, 1, window_size, num_sensors)

        batch_size = context.shape[0]
        context_size = context.shape[1]

        x = self.encoder(x)
        # (batch_size, embed_dim)

        context = context.view(
            batch_size * context_size,
            context.shape[2],
            context.shape[3],
            context.shape[4],
        )  # (batch_size * context_size, 1, window_size, num_sensors)

        context = self.encoder(context)
        # (batch_size * context_size, embed_dim)

        context = context.view(batch_size, context_size, -1)
        # (batch_size, context_size, embed_dim)

        phi = self.contextual_aggregator(context)
        # (batch_size, embed_dim)

        logits = self.decoder(x, phi)
        # (batch_size, num_classes)

        return logits


if __name__ == "__main__":
    model = NPHAR(
        num_classes=10,
        embed_dim=32,
        num_heads=8,
        mlp_dim=64,
        num_blocks=2,
        num_layers_dec=2,
    )
    print(model)
