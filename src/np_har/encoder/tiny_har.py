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


class CrossSensorInteractionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int) -> None:
        super(CrossSensorInteractionBlock, self).__init__()

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


class CrossSensorFusion(nn.Module):
    def __init__(self, embed_dim: int, num_sensors: int) -> None:
        super(CrossSensorFusion, self).__init__()

        self.proj = nn.Linear(embed_dim * num_sensors, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        # (batch_size, num_filters, length, num_sensors)

        batch_size = x.shape[0]
        num_filters = x.shape[1]
        length = x.shape[2]
        num_sensors = x.shape[3]

        x = x.view(batch_size * length, num_filters * num_sensors)
        # (batch_size * length, num_filters * num_sensors)

        x = self.proj(x)
        # (batch_size * length, embed_dim)

        x = x.view(batch_size, length, num_filters)
        # (batch_size, length, embed_dim)

        return x


class TinyHAR(nn.Module):
    def __init__(
        self,
        num_sensors: int = 6,
        num_conv_layers: int = 4,
        num_filters: int = 32,
        kernel_size: Tuple[int, int] = (5, 1),
        stride: Tuple[int, int] = (2, 1),
        num_heads: int = 4,
        mlp_dim: int = 64,
        num_blocks: int = 2,
    ) -> None:
        super(TinyHAR, self).__init__()

        self.conv_subnet = ConvSubnet(
            num_conv_layers=num_conv_layers,
            num_filters=num_filters,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.cross_channel_interaction = nn.Sequential(
            *[
                CrossSensorInteractionBlock(
                    embed_dim=num_filters, num_heads=num_heads, mlp_dim=mlp_dim
                )
                for _ in range(num_blocks)
            ]
        )

        self.cross_sensor_fusion = CrossSensorFusion(
            embed_dim=num_filters, num_sensors=num_sensors
        )

    def forward(self, x: Tensor) -> Tensor:
        # (batch_size, num_filters, window_size, num_sensors)

        x = self.conv_subnet(x)
        # (batch_size, num_filters, length, num_sensors)

        print(x.shape)

        x = self.cross_channel_interaction(x)
        # (batch_size, num_filters, length, num_sensors)

        print(x.shape)

        x = self.cross_sensor_fusion(x)
        # (batch_size, length, num_filters)

        print(x.shape)

        return x


if __name__ == "__main__":
    model = TinyHAR()
    input = torch.rand(16, 1, 128, 6)
    output = model(input)
    print(model)
