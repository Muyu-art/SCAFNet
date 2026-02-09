import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageFPN(nn.Module):
    """
    图像特征金字塔网络（FPN）
    输入：来自图像主干网络的多尺度特征 [C2, C3, C4, C5]
    输出：
        - pyramid_feats: [P2, P3, P4, P5]，每个都是 [B, C_out, H_l, W_l]
        - fused_feat: 将各层上采样到 P2 尺度后在通道维拼接的融合特征 [B, 4*C_out, H_2, W_2]
    """

    def __init__(self, in_channels_list, out_channels):
        """
        Args:
            in_channels_list (list[int]): 主干各层通道数，如 [256, 512, 1024, 2048]
            out_channels (int): FPN 每层输出的统一通道数，例如 128
        """
        super().__init__()

        assert len(in_channels_list) == 4, "现在默认支持 C2~C5 四个尺度"

        self.out_channels = out_channels

        # 1x1 侧向卷积：将主干输出映射到统一通道数
        self.lateral_convs = nn.ModuleList()
        # 3x3 输出卷积：融合后特征细化 & 去混叠
        self.output_convs = nn.ModuleList()

        for in_ch in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_ch, out_channels, kernel_size=1)
            )
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, feats):
        """
        Args:
            feats (list[Tensor]):
                来自主干网络的特征列表 [C2, C3, C4, C5]
                其中 C2 空间分辨率最高，C5 最低

        Returns:
            pyramid_feats (list[Tensor]): [P2, P3, P4, P5]
            fused_feat (Tensor): [B, 4*out_channels, H_2, W_2]
        """
        assert len(feats) == len(self.lateral_convs), \
            f"期望 {len(self.lateral_convs)} 个特征层，实际给了 {len(feats)} 个"

        # step 1: 侧向映射到统一通道
        lateral_feats = [
            conv(x) for conv, x in zip(self.lateral_convs, feats)
        ]  # [B, C_out, H_l, W_l] * 4

        # step 2: 自顶向下逐级融合（P5 -> P4 -> P3 -> P2）
        # 这里直接在 lateral_feats 上原地更新
        for i in range(len(lateral_feats) - 1, 0, -1):
            up = F.interpolate(
                lateral_feats[i],
                size=lateral_feats[i - 1].shape[2:],
                mode="nearest"
            )
            lateral_feats[i - 1] = lateral_feats[i - 1] + up

        # step 3: 3x3 conv 细化，得到 P2~P5
        pyramid_feats = [
            out_conv(f) for out_conv, f in zip(self.output_convs, lateral_feats)
        ]  # [P2, P3, P4, P5]

        # step 4: 统一上采样到 P2 尺度并在通道维拼接作为融合语义图
        target_h, target_w = pyramid_feats[0].shape[2:]
        upsampled = [
            F.interpolate(f, size=(target_h, target_w), mode="nearest")
            for f in pyramid_feats
        ]
        fused_feat = torch.cat(upsampled, dim=1)  # [B, 4*C_out, H_2, W_2]

        return pyramid_feats, fused_feat