import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """
    Safe Spatial Transformer Module (PyTorch 2.x compatible)
    """

    def __init__(self, in_channels, spatial_dims, hidden_dim=128):
        super().__init__()
        self.in_channels = in_channels

        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 6)
        )

        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x, return_grid=False):
        B = x.size(0)

        xs = self.localization(x)
        xs = xs.view(B, -1)

        theta = self.fc(xs).view(-1, 2, 3)

        grid = F.affine_grid(
            theta,
            size=x.size(),
            align_corners=False
        )

        x_trans = F.grid_sample(
            x,
            grid,
            align_corners=False
        )

        if return_grid:
            return x_trans, grid
        return x_trans
