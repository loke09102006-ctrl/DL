import torch
import torch.nn as nn


# ==========================================
# DWT
# ==========================================

class DWT(nn.Module):

    def forward(self, x):

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        LL = (x00 + x01 + x10 + x11) / 2
        LH = (x00 - x01 + x10 - x11) / 2
        HL = (x00 + x01 - x10 - x11) / 2
        HH = (x00 - x01 - x10 + x11) / 2

        return LL, LH, HL, HH


# ==========================================
# MAMBA-STYLE BLOCK
# ==========================================

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

        # Convert image features into sequence

        x = x.flatten(2)

        x = x.transpose(1, 2)

        residual = x

        x = self.norm(x)

        x = x.transpose(1, 2)

        x = self.conv1(x)

        x = self.activation(x)

        x = self.conv2(x)

        x = x.transpose(1, 2)

        # Residual connection

        x = x + residual

        x = x.transpose(1, 2)

        x = x.reshape(B, C, H, W)

        return x


# ==========================================
# DWT + CNN + MAMBA DEHAZER
# ==========================================

class DWTCNNMambaDehazer(nn.Module):

    def __init__(self):

        super().__init__()

        self.dwt = DWT()

        # ==================================
        # CNN FEATURE EXTRACTION
        # ==================================

        self.encoder = nn.Sequential(

            nn.Conv2d(
                12,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )

        # ==================================
        # MAMBA BLOCK
        # ==================================

        self.mamba = MambaBlock(64)

        self.mamba2 = MambaBlock(64)

        # ==================================
        # CNN DECODER
        # ==================================

        self.decoder = nn.Sequential(

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                12,
                kernel_size=3,
                padding=1
            )
        )

        # ==================================
        # OUTPUT RECONSTRUCTION
        # ==================================

        self.output = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                3,
                kernel_size=3,
                padding=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        # ==================================
        # DWT
        # ==================================

        LL, LH, HL, HH = self.dwt(x)

        # ==================================
        # FREQUENCY FUSION
        # ==================================

        features = torch.cat(
            [LL, LH, HL, HH],
            dim=1
        )

        # ==================================
        # CNN
        # ==================================

        features = self.encoder(features)

        # ==================================
        # MAMBA
        # ==================================

        features = self.mamba(features)

        features = self.mamba2(features)

        # ==================================
        # CNN DECODER
        # ==================================

        features = self.decoder(features)

        # ==================================
        # SPLIT WAVELET COMPONENTS
        # ==================================

        LL = features[:, 0:3]

        LH = features[:, 3:6]

        HL = features[:, 6:9]

        HH = features[:, 9:12]

        # ==================================
        # INVERSE DWT
        # ==================================

        h = LL.shape[2]
        w = LL.shape[3]

        output = torch.zeros(
            x.shape[0],
            3,
            h * 2,
            w * 2,
            device=x.device
        )

        output[:, :, 0::2, 0::2] = (
            LL + LH + HL + HH
        ) / 2

        output[:, :, 0::2, 1::2] = (
            LL - LH + HL - HH
        ) / 2

        output[:, :, 1::2, 0::2] = (
            LL + LH - HL - HH
        ) / 2

        output[:, :, 1::2, 1::2] = (
            LL - LH - HL + HH
        ) / 2

        # ==================================
        # FINAL CNN
        # ==================================

        output = self.output(output)

        return output