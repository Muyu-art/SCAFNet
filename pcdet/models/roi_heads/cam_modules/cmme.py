import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


def _norm2d(num_channels: int, use_gn: bool = False, gn_groups: int = 32) -> nn.Module:
    """
    Small-batch friendly option:
      - BN (default)
      - or GN if use_gn=True
    """
    if use_gn:
        g = min(gn_groups, num_channels)
        while num_channels % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, num_channels)
    return nn.BatchNorm2d(num_channels)


class LiDARGuidedImageFusion(nn.Module):
    """
    LI-Fusion: LiDAR -> guide Image
      attn = sigmoid(Wa * tanh(Wl(F_L) + Wi(F_I)))
      F_I' = Conv3x3([F_I, attn * F_L])
    """

    def __init__(self, channels: int, use_gn: bool = False, gn_groups: int = 32):
        super().__init__()
        self.channels = channels

        self.w_l = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.w_i = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.w_attn = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

        self.out_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            _norm2d(channels, use_gn=use_gn, gn_groups=gn_groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_lidar: torch.Tensor, feat_img: torch.Tensor) -> torch.Tensor:
        assert feat_lidar.shape == feat_img.shape, \
            f"LI-Fusion shape mismatch: lidar={feat_lidar.shape}, img={feat_img.shape}"

        h = torch.tanh(self.w_l(feat_lidar) + self.w_i(feat_img))  # [B,C,H,W]
        attn = torch.sigmoid(self.w_attn(h))                       # [B,1,H,W]

        fused = torch.cat([feat_img, attn * feat_lidar], dim=1)    # [B,2C,H,W]
        feat_img_enh = self.out_conv(fused)                        # [B,C,H,W]
        return feat_img_enh


class ImageGuidedLiDARFusion(nn.Module):
    """
    IL-Fusion: Image -> guide LiDAR
      attn = sigmoid(Wa * tanh(Wi(F_I) + Wl(F_L)))
      F_L' = Conv3x3([F_L, attn * Proj(F_I)])   ✅关键：注入图像语义，而不是 lidar 自己 gate 自己
    """

    def __init__(self, channels: int, use_gn: bool = False, gn_groups: int = 32):
        super().__init__()
        self.channels = channels

        self.w_i = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.w_l = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.w_attn = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

        # 若你未来让 image/lidar 通道不同，可把这里改成 in_ch_img -> channels
        self.img_proj = nn.Identity()

        self.out_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            _norm2d(channels, use_gn=use_gn, gn_groups=gn_groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_lidar: torch.Tensor, feat_img: torch.Tensor) -> torch.Tensor:
        assert feat_lidar.shape == feat_img.shape, \
            f"IL-Fusion shape mismatch: lidar={feat_lidar.shape}, img={feat_img.shape}"

        h = torch.tanh(self.w_i(feat_img) + self.w_l(feat_lidar))  # [B,C,H,W]
        attn = torch.sigmoid(self.w_attn(h))                       # [B,1,H,W]

        img_inj = self.img_proj(feat_img)
        fused = torch.cat([feat_lidar, attn * img_inj], dim=1)     # [B,2C,H,W]
        feat_lidar_enh = self.out_conv(fused)                      # [B,C,H,W]
        return feat_lidar_enh


class CascadeMultiModalEnhancer(nn.Module):
    """
    CMME: cascade stages
      stage k:
        I_k' = LI(F_L^k, F_I^k)
        L_k' = IL(F_L^k, I_k')
      optional residual:
        I_k_out = I_k + alpha * I_k'
        L_k_out = L_k + alpha * L_k'
    """

    def __init__(
        self,
        in_channels: int,
        num_stages: int = 2,
        use_residual: bool = True,
        residual_alpha: float = 1.0,
        use_gn: bool = False,
        gn_groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_stages = num_stages
        self.use_residual = use_residual
        self.residual_alpha = float(residual_alpha)

        self.li_fusion_blocks = nn.ModuleList([
            LiDARGuidedImageFusion(channels=in_channels, use_gn=use_gn, gn_groups=gn_groups)
            for _ in range(num_stages)
        ])
        self.il_fusion_blocks = nn.ModuleList([
            ImageGuidedLiDARFusion(channels=in_channels, use_gn=use_gn, gn_groups=gn_groups)
            for _ in range(num_stages)
        ])

    def forward(
        self,
        lidar_feats: List[torch.Tensor],
        image_feats: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        assert len(lidar_feats) == len(image_feats), \
            f"LiDAR/Image list length mismatch: {len(lidar_feats)} vs {len(image_feats)}"
        assert len(lidar_feats) == self.num_stages, \
            f"Given {len(lidar_feats)} stages, but num_stages={self.num_stages}"

        enhanced_lidar_feats: List[torch.Tensor] = []
        enhanced_image_feats: List[torch.Tensor] = []

        for k in range(self.num_stages):
            F_L_k = lidar_feats[k]
            F_I_k = image_feats[k]
            assert F_L_k.shape == F_I_k.shape, \
                f"Stage {k} shape mismatch: lidar={F_L_k.shape}, img={F_I_k.shape}"

            # 1) LI-Fusion: LiDAR -> Image
            F_I_k_enh = self.li_fusion_blocks[k](F_L_k, F_I_k)

            # 2) IL-Fusion: Image -> LiDAR (use enhanced image)
            F_L_k_enh = self.il_fusion_blocks[k](F_L_k, F_I_k_enh)

            if self.use_residual:
                a = self.residual_alpha
                F_I_k_out = F_I_k + a * F_I_k_enh
                F_L_k_out = F_L_k + a * F_L_k_enh
            else:
                F_I_k_out = F_I_k_enh
                F_L_k_out = F_L_k_enh

            enhanced_image_feats.append(F_I_k_out)
            enhanced_lidar_feats.append(F_L_k_out)

        return enhanced_lidar_feats, enhanced_image_feats


if __name__ == "__main__":
    B, C = 2, 128
    Hs = [64, 32]
    Ws = [64, 32]

    lidar_feats = [torch.randn(B, C, Hs[i], Ws[i]) for i in range(2)]
    image_feats = [torch.randn(B, C, Hs[i], Ws[i]) for i in range(2)]

    cmme = CascadeMultiModalEnhancer(in_channels=C, num_stages=2, use_residual=True, residual_alpha=1.0, use_gn=False)
    out_lidar, out_image = cmme(lidar_feats, image_feats)
    for i in range(2):
        print(f"Stage {i}: LiDAR {out_lidar[i].shape}, Image {out_image[i].shape}")
