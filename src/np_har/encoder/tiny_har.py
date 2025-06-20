from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor


class ConvSubnet(nn.Module):
    def __init__(
        self,
        num_conv_layers: int,
        num_filters: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
    ) -> None:
        super(ConvSubnet, self).__init__()

        layers = []

        for i in range(num_conv_layers):
            # start with 1 and end with num_filters
            in_channels = int(i * num_filters / num_conv_layers) or 1
            out_channels = int((i + 1) * num_filters / num_conv_layers)

            # use 2d because handles batching better
            block = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ]

            layers.extend(block)

        self.conv_subnet = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # (batch_size, 1, window_size, num_sensors)

        x = self.conv_subnet(x)
        # (batch_size, num_filters, length, num_sensors)

        return x


class SensorInteractionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int) -> None:
        super(SensorInteractionBlock, self).__init__()

        # embed_dim corresponds to num_filters
        self.mhsa = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # (batch_size, num_filters, length, num_sensors)

        batch_size = x.shape[0]
        num_filters = x.shape[1]
        length = x.shape[2]
        num_sensors = x.shape[3]

        x = x.view(batch_size * length, num_sensors, num_filters)
        # (batch_size * length, num_sensors, num_filters)

        x, _ = self.mhsa(x, x, x)
        x = self.mlp(x)
        # (batch_size * length, num_sensors, num_filters)

        x = x.view(batch_size, num_filters, length, num_sensors)
        # (batch_size, num_filters, length, num_sensors)

        return x


class SensorFusion(nn.Module):
    def __init__(
        self, input_embed_dim: int, num_sensors: int, output_embed_dim: int
    ) -> None:
        super(SensorFusion, self).__init__()
        self.output_embed_dim = output_embed_dim
        self.proj = nn.Linear(input_embed_dim * num_sensors, output_embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        # (batch_size, num_filters, length, num_sensors)

        batch_size = x.shape[0]
        num_filters = x.shape[1]
        length = x.shape[2]
        num_sensors = x.shape[3]

        x = x.view(batch_size * length, num_filters * num_sensors)
        # (batch_size * length, num_filters * num_sensors)

        x = self.proj(x)
        # (batch_size * length, output_embed_dim)

        x = x.view(batch_size, length, self.output_embed_dim)
        # (batch_size, length, output_embed_dim)

        return x


class TemporalFusion(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int) -> None:
        super(TemporalFusion, self).__init__()

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        # (batch_size, length, input_dim)

        out, _ = self.lstm(x)
        # (batch_size, length, input_dim)

        return out


class TinyHAR(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        num_sensors: int = 6,
        num_conv_layers: int = 4,
        num_filters: int = 32,
        kernel_size: Tuple[int, int] = (5, 1),
        stride: Tuple[int, int] = (2, 1),
        num_heads: int = 4,
        mlp_dim: int = 64,
        num_blocks: int = 2,
        lstm_layers: int = 2,
    ) -> None:
        super(TinyHAR, self).__init__()

        self.conv_subnet = ConvSubnet(
            num_conv_layers=num_conv_layers,
            num_filters=num_filters,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.sensor_interaction = nn.Sequential(
            *[
                SensorInteractionBlock(
                    embed_dim=num_filters, num_heads=num_heads, mlp_dim=mlp_dim
                )
                for _ in range(num_blocks)
            ]
        )

        self.sensor_fusion = SensorFusion(
            input_embed_dim=num_filters,
            num_sensors=num_sensors,
            output_embed_dim=2 * num_filters,
        )

        self.temporal_fusion = TemporalFusion(
            embed_dim=2 * num_filters,
            num_layers=lstm_layers,
        )

        self.predictor = nn.Linear(2 * num_filters, num_classes)

    def encode(self, x: Tensor) -> Tensor:
        # (batch_size, num_filters, window_size, num_sensors)

        # print(x.shape)

        x = self.conv_subnet(x)
        # (batch_size, num_filters, length, num_sensors)

        # print(x.shape)

        x = self.sensor_interaction(x)
        # (batch_size, num_filters, length, num_sensors)

        # print(x.shape)

        x = self.sensor_fusion(x)
        # (batch_size, length, num_filters)

        # print(x.shape)

        x = self.temporal_fusion(x)
        # (batch_size, length, num_filters)

        # print(x.shape)

        return x

    def forward(self, x: Tensor) -> Tensor:
        # (batch_size, length, num_filters)

        x = self.encode(x)
        # (batch_size, length, num_filters)

        logits = self.predictor(x[:, -1, :])
        # (batch_size, num_classes)

        # print(logits.shape)

        return logits


if __name__ == "__main__":
    model = TinyHAR()
    input = torch.rand(16, 1, 128, 6)
    output = model(input)
    print(model)
