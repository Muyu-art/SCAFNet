import torch
import torch.nn as nn
import torchvision.models as models


class ImageBackbone(nn.Module):
    """
    Image Backbone: ResNet18
    输出 C2/C3/C4/C5 特征：
        C2: 64  @ stride 4
        C3: 128 @ stride 8
        C4: 256 @ stride 16
        C5: 512 @ stride 32
    对应 YAML:
        IN_CHANNELS_LIST: [64, 128, 256, 512]
    """

    def __init__(self, backbone='resnet18', pretrained=True, freeze_bn=False):
        super().__init__()

        if backbone == 'resnet18':
            # torchvision 新版本推荐 weights=...；你这份写法也能跑
            resnet = models.resnet18(pretrained=pretrained)
            self.out_channels = [64, 128, 256, 512]
        else:
            raise ValueError("只允许使用 ResNet18，因为 YAML 指定的是 [64,128,256,512]")

        # stem: conv1(s2) + maxpool(s2) => stride=4
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer1 = resnet.layer1   # stride=4   -> C2
        self.layer2 = resnet.layer2   # stride=8   -> C3
        self.layer3 = resnet.layer3   # stride=16  -> C4
        self.layer4 = resnet.layer4   # stride=32  -> C5

        if freeze_bn:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, images):
        """
        images: [B, 3, H, W]
        return: [C2, C3, C4, C5]
        """
        assert images.ndim == 4 and images.shape[1] == 3, f"expect [B,3,H,W], got {tuple(images.shape)}"

        x = self.stem(images)        # [B,64,H/4,W/4]
        C2 = self.layer1(x)          # [B,64,H/4,W/4]
        C3 = self.layer2(C2)         # [B,128,H/8,W/8]
        C4 = self.layer3(C3)         # [B,256,H/16,W/16]
        C5 = self.layer4(C4)         # [B,512,H/32,W/32]

        return [C2, C3, C4, C5]
