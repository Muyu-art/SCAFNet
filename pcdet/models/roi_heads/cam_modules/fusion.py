import torch
import torch.nn as nn
import torch.nn.functional as F


class CascadeAdaptiveFusion(nn.Module):
    """
    CAF: 级联自适应模态融合模块

    对应论文中的公式 (3.17) ~ (3.20)：
        F̂_L = W_L * F_L'
        F̂_I = W_I * F_I'
        v_L = GAP(F̂_L), v_I = GAP(F̂_I)
        [α_L, α_I] = softmax([W_α v_L, W_α v_I])
        F_M = α_L * F̂_L + α_I * F̂_I
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = int(in_channels)

        # W_L, W_I: 模态内线性变换（加 BN 稳定分布）
        self.proj_lidar = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )
        self.proj_img = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        # 对 GAP 向量做归一化，避免尺度差导致权重塌缩
        self.ln = nn.LayerNorm(self.in_channels)

        # W_α: 共享的模态权重产生网络
        self.modal_fc = nn.Linear(self.in_channels, 1, bias=True)

    def forward(self, feat_lidar: torch.Tensor, feat_img: torch.Tensor):
        """
        Args:
            feat_lidar: [B, C, H, W]
            feat_img:   [B, C, H, W]

        Returns:
            fused_feat: [B, C, H, W]
            alpha_l:    [B, 1, 1, 1]
            alpha_i:    [B, 1, 1, 1]
        """
        assert feat_lidar.shape == feat_img.shape, \
            f"CAF 输入尺寸不匹配: lidar={feat_lidar.shape}, img={feat_img.shape}"

        B, C, H, W = feat_lidar.shape
        assert C == self.in_channels, f"Channel mismatch: got C={C}, expect {self.in_channels}"

        # 1) 线性映射到共享空间
        F_l = self.proj_lidar(feat_lidar)  # [B,C,H,W]
        F_i = self.proj_img(feat_img)      # [B,C,H,W]

        # 2) GAP 得到全局描述向量
        v_l = F.adaptive_avg_pool2d(F_l, 1).view(B, C)  # [B,C]
        v_i = F.adaptive_avg_pool2d(F_i, 1).view(B, C)  # [B,C]

        # 归一化，避免模态尺度差异导致 softmax 一边倒
        v_l = self.ln(v_l)
        v_i = self.ln(v_i)

        # 3) 计算模态权重分布 α_L, α_I
        s_l = self.modal_fc(v_l)  # [B,1]
        s_i = self.modal_fc(v_i)  # [B,1]
        logits = torch.cat([s_l, s_i], dim=-1)  # [B,2]
        alpha = torch.softmax(logits, dim=-1)   # [B,2]

        alpha_l = alpha[:, 0].view(B, 1, 1, 1)
        alpha_i = alpha[:, 1].view(B, 1, 1, 1)

        # 4) 根据模态权重融合
        fused_feat = alpha_l * F_l + alpha_i * F_i
        return fused_feat, alpha_l, alpha_i
