import torch
import torch.nn as nn


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


class DWTCNNDehazer(nn.Module):

    def __init__(self):

        super().__init__()

        self.dwt = DWT()

        self.encoder = nn.Sequential(

            nn.Conv2d(12, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 12, 3, padding=1)
        )

        self.output = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        LL, LH, HL, HH = self.dwt(x)

        features = torch.cat(
            [LL, LH, HL, HH],
            dim=1
        )

        features = self.encoder(features)

        features = self.decoder(features)

        LL = features[:, 0:3]
        LH = features[:, 3:6]
        HL = features[:, 6:9]
        HH = features[:, 9:12]

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

        output = self.output(output)

        return output