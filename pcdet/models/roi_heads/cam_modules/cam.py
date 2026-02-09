import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import CascadeAdaptiveFusion


class ChannelSpatialAttention(nn.Module):
    """
    通道 + 空间注意力模块（CAM）：
        M_c = σ(MLP(GAP(F_M)))
        F_c = M_c ⊙ F_M

        M_s = σ(Conv3×3([Avg(F_c), Max(F_c)]))
        F_s = M_s ⊙ F_c
    """

    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        self.channels = int(channels)

        hidden = max(1, self.channels // int(ratio))

        # 通道注意力 M_c
        self.mlp = nn.Sequential(
            nn.Linear(self.channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.channels, bias=True)  # ✅必要修复：允许学习偏移
        )

        # 空间注意力 M_s
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C, H, W] 输入融合特征 F_M

        Returns:
            x_cs: [B, C, H, W] 注意力后的特征 F_s
            M_c:  [B, C, 1, 1] 通道注意力权重
            M_s:  [B, 1, H, W] 空间注意力权重
        """
        B, C, H, W = x.shape
        assert C == self.channels, f"Channel mismatch: x has C={C}, but module channels={self.channels}"

        # ---- 通道注意力 ----
        gap = F.adaptive_avg_pool2d(x, 1).view(B, C)     # [B,C]
        mc_vec = torch.sigmoid(self.mlp(gap))            # [B,C]
        M_c = mc_vec.view(B, C, 1, 1)                    # [B,C,1,1]
        x_c = x * M_c                                    # [B,C,H,W]

        # ---- 空间注意力 ----
        avg_map = torch.mean(x_c, dim=1, keepdim=True)   # [B,1,H,W]
        max_map, _ = torch.max(x_c, dim=1, keepdim=True) # [B,1,H,W]
        ms_input = torch.cat([avg_map, max_map], dim=1)  # [B,2,H,W]
        M_s = torch.sigmoid(self.spatial_conv(ms_input)) # [B,1,H,W]

        x_cs = x_c * M_s
        return x_cs, M_c, M_s


class CAFCAMModule(nn.Module):
    """
    CAF-CAM 级联模块：
        - CAF: CascadeAdaptiveFusion 得到 F_M^(k)
        - CAM: ChannelSpatialAttention 得到 F_s^(k)
        - 残差：F_F^(k) = F_s^(k) + F_M^(k)
    """

    def __init__(self,
                 in_channels: int,
                 num_stages: int = 3,
                 attn_ratio: int = 16):
        super().__init__()

        self.in_channels = int(in_channels)
        self.num_stages = int(num_stages)

        self.caf_blocks = nn.ModuleList([
            CascadeAdaptiveFusion(in_channels=self.in_channels)
            for _ in range(self.num_stages)
        ])

        self.cam_blocks = nn.ModuleList([
            ChannelSpatialAttention(channels=self.in_channels, ratio=attn_ratio)
            for _ in range(self.num_stages)
        ])

    def forward(self, lidar_feats, image_feats):
        """
        Args:
            lidar_feats: List[Tensor], each [B, C, H_k, W_k]
            image_feats: List[Tensor], each [B, C, H_k, W_k]

        Returns:
            fused_feats: List[Tensor], each [B, C, H_k, W_k]
            mc_list: List[Tensor], each [B, C, 1, 1]
            ms_list: List[Tensor], each [B, 1, H_k, W_k]
        """
        assert len(lidar_feats) == len(image_feats), \
            f"CAFCAM 输入特征数量不一致: lidar={len(lidar_feats)}, img={len(image_feats)}"
        assert len(lidar_feats) == self.num_stages, \
            f"给定特征层数 {len(lidar_feats)} 与 num_stages={self.num_stages} 不匹配"

        fused_feats, mc_list, ms_list = [], [], []

        for k in range(self.num_stages):
            F_L_k = lidar_feats[k]
            F_I_k = image_feats[k]

            # 1) CAF：得到 F_M^(k)
            F_M_k, alpha_l, alpha_i = self.caf_blocks[k](F_L_k, F_I_k)

            # 2) CAM：得到 F_s^(k)
            F_s_k, M_c_k, M_s_k = self.cam_blocks[k](F_M_k)

            # 3) 残差
            F_F_k = F_s_k + F_M_k

            fused_feats.append(F_F_k)
            mc_list.append(M_c_k)
            ms_list.append(M_s_k)

        return fused_feats, mc_list, ms_list

    @staticmethod
    def fuse_multiscale_to_bev(fused_feats, mode: str = "concat"):
        """
        多尺度融合为统一 BEV 特征图。

        Args:
            fused_feats: List[Tensor], each [B,C,H_k,W_k]
            mode: "concat" or "sum"

        Returns:
            bev_feat: concat -> [B, C*K, H0, W0], sum -> [B, C, H0, W0]
        """
        assert len(fused_feats) > 0
        B, C, H0, W0 = fused_feats[0].shape

        upsampled = []
        for f in fused_feats:
            if f.shape[2:] != (H0, W0):
                f = F.interpolate(f, size=(H0, W0), mode="bilinear", align_corners=False)
            upsampled.append(f)

        if mode == "concat":
            return torch.cat(upsampled, dim=1)
        elif mode == "sum":
            return torch.stack(upsampled, dim=0).sum(dim=0)
        else:
            raise ValueError(f"Unsupported fuse mode: {mode}")
