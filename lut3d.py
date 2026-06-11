# lut3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def _logit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Numerically stable logit for x in [0,1].
    logit(x) = log(x/(1-x))
    """
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


class LUT3D(nn.Module):
    """
    3D LUT implemented via torch.grid_sample (trilinear for 3D).
    LUT volume is shaped as [1, 3, N, N, N] interpreted as:
        D = r-axis, H = g-axis, W = b-axis
    Input image:
        img: [B, 3, H, W] in RGB, range [0,1]
    """

    def __init__(self, N: int = 33, init_identity: bool = True, eps: float = 1e-6):
        super().__init__()
        self.N = int(N)
        self.eps = float(eps)

        if init_identity:
            # Build identity LUT in [0,1]: out = (r,g,b)
            grid = torch.linspace(0.0, 1.0, self.N)
            r, g, b = torch.meshgrid(grid, grid, grid, indexing="ij")  # [N,N,N] axes (r,g,b)

            identity = torch.stack([r, g, b], dim=0)  # [3, N, N, N], values in [0,1]

            # IMPORTANT:
            # We parameterize LUT as sigmoid(lut_param) so it stays in [0,1].
            # To make sigmoid(lut_param) == identity, we must initialize lut_param = logit(identity).
            lut_param_init = _logit(identity, eps=self.eps)
        else:
            # Random init in logit space (so sigmoid will be in (0,1))
            # Start near 0.5 to avoid saturation.
            lut_param_init = torch.zeros(3, self.N, self.N, self.N)

        self.lut_param = nn.Parameter(lut_param_init)  # unconstrained

    def lut(self) -> torch.Tensor:
        """
        Return LUT volume constrained to [0,1].
        Shape: [3, N, N, N]
        """
        return torch.sigmoid(self.lut_param)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: [B,3,H,W] RGB float in [0,1]
        returns: [B,3,H,W] RGB float in [0,1]
        """
        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError("img must be [B,3,H,W]")

        x = img.clamp(0.0, 1.0)
        B, _, H, W = x.shape

        # grid_sample for 3D expects:
        #   input: [B, C, D, H, W]
        #   grid : [B, outD, outH, outW, 3]  (coords in [-1,1])
        # We set outD=1, outH=H, outW=W.
        #
        # Axis meaning of LUT volume: D=r, H=g, W=b.
        # grid[...,0]=x corresponds to W axis => b
        # grid[...,1]=y corresponds to H axis => g
        # grid[...,2]=z corresponds to D axis => r

        r = x[:, 0:1, :, :]  # [B,1,H,W]
        g = x[:, 1:2, :, :]
        b = x[:, 2:3, :, :]

        # Map [0,1] -> [-1,1]
        gx = b * 2.0 - 1.0
        gy = g * 2.0 - 1.0
        gz = r * 2.0 - 1.0

        # Build grid: [B,1,H,W,3]
        grid = torch.cat([gx, gy, gz], dim=1)          # [B,3,H,W]
        grid = grid.permute(0, 2, 3, 1).unsqueeze(1)   # [B,1,H,W,3]

        # LUT volume: [B,3,N,N,N]
        lut_vol = self.lut().unsqueeze(0).expand(B, -1, -1, -1, -1).contiguous()

        # Trilinear sampling via grid_sample(mode="bilinear" for 3D)
        out = F.grid_sample(
            lut_vol, grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )  # [B,3,1,H,W]
        out = out.squeeze(2)  # [B,3,H,W]
        return out

    def smoothness_loss(self) -> torch.Tensor:
        """Total variation on LUT grid (encourage smooth mapping)."""
        lut = self.lut()  # [3,N,N,N]
        loss = 0.0
        # along r-axis (D dimension)
        loss = loss + (lut[:, 1:, :, :] - lut[:, :-1, :, :]).abs().mean()
        # along g-axis (H dimension)
        loss = loss + (lut[:, :, 1:, :] - lut[:, :, :-1, :]).abs().mean()
        # along b-axis (W dimension)
        loss = loss + (lut[:, :, :, 1:] - lut[:, :, :, :-1]).abs().mean()
        return loss

    def identity_loss(self) -> torch.Tensor:
        """Penalty to keep LUT near identity."""
        N = self.N
        device = self.lut_param.device
        grid = torch.linspace(0.0, 1.0, N, device=device)
        r, g, b = torch.meshgrid(grid, grid, grid, indexing="ij")
        identity = torch.stack([r, g, b], dim=0)  # [3,N,N,N]
        return (self.lut() - identity).abs().mean()
