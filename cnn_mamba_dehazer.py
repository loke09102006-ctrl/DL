import torch
import torch.nn as nn


class MambaBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.norm = nn.LayerNorm(channels)

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels
        )

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1
        )

        self.activation = nn.GELU()

    def forward(self, x):

        # x = B, C, H, W
        B, C, H, W = x.shape

        # Convert to sequence
        x = x.flatten(2)
        x = x.transpose(1, 2)

        residual = x

        x = self.norm(x)

        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)

        x = x.transpose(1, 2)

        x = x + residual

        x = x.transpose(1, 2)

        return x.reshape(B, C, H, W)


class CNNMambaDehazer(nn.Module):

    def __init__(self):
        super().__init__()

        # CNN feature extraction
        self.encoder = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        # Mamba-style sequence processing
        self.mamba = MambaBlock(64)

        self.mamba2 = MambaBlock(64)

        # CNN reconstruction
        self.decoder = nn.Sequential(

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 3, 3, padding=1),

            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.encoder(x)

        x = self.mamba(x)

        x = self.mamba2(x)

        x = self.decoder(x)

        return x