import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .detector3d_template import Detector3DTemplate
from pcdet.models.backbones_2d import base_bev_backbone

from pcdet.models.backbones_2d.image_fpn import ImageFPN
from pcdet.models.backbones_2d.image_resnet import ImageBackbone
from pcdet.models.roi_heads.cam_modules.cmme import CascadeMultiModalEnhancer
from pcdet.models.roi_heads.cam_modules.cam import CAFCAMModule


def tensorify(x, device, force_int=False):
    """只把 numpy 转成 Tensor，保持其他内容不变"""
    if isinstance(x, np.ndarray):
        if force_int:
            return torch.from_numpy(x).int().to(device)
        else:
            return torch.from_numpy(x).float().to(device)
    return x


class SCAFNet(Detector3DTemplate):
    """
    SCAFNet Framework (方案1：原链路，稳定提点优先):
        - Image Backbone (ResNet) → multi-scale C2,C3,C4,C5
        - ImageFPN → multi-scale semantic features
        - (NO Point-level Aligner)
        - CMME (bi-directional LiDAR-Image enhancement)
        - CAF-CAM (adaptive fusion + attention)
        - Fusion → BEV → BACKBONE_2D → DENSE_HEAD → ROI_HEAD
        - SDL (in AnchorHeadSingle)

    NOTE（关键改动）：
        当 use_caf_cam=False 时，不再走原先 “stage 内 cat + stage 间大 cat”
        （会导致通道翻倍，破坏 BACKBONE_2D 输入接口），而是：
            - 每个 stage: cat([lidar, image]) -> 2*out_ch
            - 1x1 conv 压回 out_ch（等维朴素融合，不含自适应权重/注意力）
            - 再按 fuse_mode 将多尺度融合到 BEV（concat: out_ch*num_stages；否则: out_ch）
        从而保证与 Full 的网络宽度/接口完全一致，只移除“自适应融合+注意力”。
    """

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        self.use_alignment = getattr(model_cfg, "USE_ALIGNMENT", True)
        self.use_cmme = getattr(model_cfg, "USE_CMME", True)
        self.use_caf_cam = getattr(model_cfg, "USE_CAF_CAM", True)
        self.use_sdl = getattr(model_cfg, "USE_SDL", True)

        # ============================================================
        # 是否使用 image（与 DATA_CONFIG.GET_ITEM_LIST 对齐）
        # ============================================================
        get_item_list = list(getattr(model_cfg, "GET_ITEM_LIST", []))
        if len(get_item_list) == 0 and hasattr(dataset, "dataset_cfg"):
            get_item_list = list(getattr(dataset.dataset_cfg, "GET_ITEM_LIST", ["points"]))
        self.get_item_list = get_item_list

        self.use_image = ("image" in get_item_list) or ("images" in get_item_list) or bool(
            getattr(model_cfg, "USE_IMAGE", False)
        )

        # =====================================
        # 基础网络 (VFE / BACKBONE_3D / MAP_TO_BEV / BACKBONE_2D / DENSE_HEAD / ROI_HEAD)
        # =====================================
        self.module_list = self.build_networks()

        # =====================================
        # 1. 图像主干 Image Backbone + FPN（仅在 use_image=True 时构建）
        # =====================================
        if self.use_image:
            self.image_backbone = ImageBackbone(pretrained=True)
            self.image_fpn = ImageFPN(
                in_channels_list=model_cfg.IMAGE_FPN.IN_CHANNELS_LIST,
                out_channels=model_cfg.IMAGE_FPN.OUT_CHANNELS,
            )
        else:
            self.image_backbone = None
            self.image_fpn = None

        # =====================================
        # 1.5 方案1：不使用点级对齐（稳定提点优先）
        # =====================================
        self.use_point_aligner = False
        self.point_aligner = None

        # =====================================
        # 2. LiDAR 对齐卷积（改为 lazy init，避免通道不匹配）
        # =====================================
        out_ch = 128
        if hasattr(model_cfg, "IMAGE_FPN"):
            out_ch = int(getattr(model_cfg.IMAGE_FPN, "OUT_CHANNELS", out_ch))
        self.lidar_out_ch = out_ch
        self.lidar_align_conv = None  # lazy init in forward_multimodal

        # =====================================
        # 3. CMME 多模态增强（只有 use_image=True 才会用到）
        # =====================================
        cmme_in_channels = out_ch
        cmme_num_stages = 2
        cmme_use_residual = True
        if hasattr(model_cfg, "CMME"):
            cmme_in_channels = int(getattr(model_cfg.CMME, "IN_CHANNELS", cmme_in_channels))
            cmme_num_stages = int(getattr(model_cfg.CMME, "NUM_STAGES", cmme_num_stages))
            cmme_use_residual = bool(getattr(model_cfg.CMME, "USE_RESIDUAL", cmme_use_residual))

        self.cmme = CascadeMultiModalEnhancer(
            in_channels=cmme_in_channels,
            num_stages=cmme_num_stages,
            use_residual=cmme_use_residual,
        )

        # =====================================
        # 4. CAF-CAM 级联融合 + 注意力（只有 use_image=True 才会用到）
        # =====================================
        cam_in_channels = out_ch
        cam_num_stages = cmme_num_stages
        cam_attn_ratio = 16
        self.fuse_mode = "concat"
        if hasattr(model_cfg, "CAM"):
            cam_in_channels = int(getattr(model_cfg.CAM, "IN_CHANNELS", cam_in_channels))
            cam_num_stages = int(getattr(model_cfg.CAM, "NUM_STAGES", cam_num_stages))
            cam_attn_ratio = int(getattr(model_cfg.CAM, "ATTN_RATIO", cam_attn_ratio))
            self.fuse_mode = str(getattr(model_cfg.CAM, "FUSE_MODE", self.fuse_mode))

        self.caf_cam = CAFCAMModule(
            in_channels=cam_in_channels,
            num_stages=cam_num_stages,
            attn_ratio=cam_attn_ratio,
        )

        # ============================================================
        # ✅ NaiveFusion（仅在 use_caf_cam=False 时启用）：
        # 每个 stage: cat -> 1x1 conv 压回 out_ch，保证接口不变
        # ============================================================
        self.naive_fuse_convs = nn.ModuleList()  # lazy append to match num_stages on first forward

    # ===================================================================
    # 修改 BACKBONE_2D 输入通道：来自 CAF-CAM 的输出
    # ===================================================================
    def build_backbone_2d(self, model_info_dict):
        cam_in_channels = self.model_cfg.CAM.IN_CHANNELS
        cam_num_stages = getattr(self.model_cfg.CAM, "NUM_STAGES", 2)
        fuse_mode = getattr(self.model_cfg.CAM, "FUSE_MODE", "concat")

        if fuse_mode == "concat":
            cam_output_dim = cam_in_channels * cam_num_stages
        else:
            cam_output_dim = cam_in_channels

        backbone_2d_module = base_bev_backbone.BaseBEVBackbone(
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=cam_output_dim,
            num_frames=getattr(self, "num_frames", 1),
        )

        model_info_dict["module_list"].append(backbone_2d_module)
        model_info_dict["num_bev_features"] = backbone_2d_module.num_bev_features
        return backbone_2d_module, model_info_dict

    @staticmethod
    def _check_gt_boxes(gt_boxes, num_class: int):
        assert gt_boxes is not None, "gt_boxes is None"
        assert gt_boxes.shape[-1] >= 8, f"gt_boxes dim must be >=8, got {gt_boxes.shape}"

        cls = gt_boxes[..., -1]
        unique = torch.unique(cls.detach().to("cpu"))
        assert (cls > 0).any(), f"class id seems invalid (all <=0), unique={unique.tolist()[:20]}"
        assert cls.max() <= num_class, (
            f"class id > num_class, max={cls.max().item()}, num_class={num_class}, "
            f"unique={unique.tolist()[:20]}"
        )

    def forward(self, batch_dict):
        device = batch_dict["points"].device if ("points" in batch_dict and isinstance(batch_dict["points"], torch.Tensor)) \
            else torch.device("cuda")

        keys_fp32 = ["points", "voxels"]
        keys_int = ["voxel_coords", "voxel_num_points"]

        if self.use_image:
            # 兼容 dataset 输出 images
            if "image" not in batch_dict and "images" in batch_dict:
                batch_dict["image"] = batch_dict["images"]
            keys_fp32.append("image")

        if "gt_boxes" in batch_dict and batch_dict["gt_boxes"] is not None:
            if isinstance(batch_dict["gt_boxes"], np.ndarray):
                batch_dict["gt_boxes"] = torch.from_numpy(batch_dict["gt_boxes"]).float().to(device)
            elif isinstance(batch_dict["gt_boxes"], torch.Tensor):
                batch_dict["gt_boxes"] = batch_dict["gt_boxes"].float().to(device)
            else:
                raise TypeError(f"Unsupported gt_boxes type: {type(batch_dict['gt_boxes'])}")

            if self.training:
                self._check_gt_boxes(batch_dict["gt_boxes"], num_class=self.num_class)
                if not hasattr(self, "_dbg_once"):
                    self._dbg_once = True
                    cls = batch_dict["gt_boxes"][..., -1]
                    u = torch.unique(cls.detach().cpu())
                    print("[DBG] gt_boxes dim:", batch_dict["gt_boxes"].shape[-1], "unique cls:", u.tolist()[:20])

        for k in keys_fp32:
            if k in batch_dict and isinstance(batch_dict[k], np.ndarray):
                batch_dict[k] = torch.from_numpy(batch_dict[k]).float().to(device)

        for k in keys_int:
            if k in batch_dict and isinstance(batch_dict[k], np.ndarray):
                batch_dict[k] = torch.from_numpy(batch_dict[k]).int().to(device)

        # 1) 图像主流程
        if self.use_image:
            if "image" not in batch_dict:
                raise KeyError("模型配置 use_image=True，但 batch_dict['image'] 缺失；请检查 Dataset/GET_ITEM_LIST")

            images = batch_dict["image"]

            if self.training and (not hasattr(self, "_dbg_img_once")):
                self._dbg_img_once = True
                print("[DBG] image dtype:", images.dtype, "min/max:", float(images.min()), float(images.max()), "mean:",
                      float(images.mean()))
            if self.training and (not hasattr(self, "_dbg_img_once2")):
                self._dbg_img_once2 = True
                x = batch_dict["image"]
                if float(x.max()) > 50:
                    print("[WARN] image seems NOT normalized! max=", float(x.max()))

            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images)

            if images.ndim == 4 and images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2).contiguous()

            images = images.to(device).float()
            batch_dict["image"] = images

            if self.image_backbone is None:
                raise RuntimeError("use_image=True 但 image_backbone 未构建，请检查 __init__ 逻辑")

            image_backbone_feats = self.image_backbone(images)
            batch_dict["image_backbone_feats"] = image_backbone_feats
        else:
            batch_dict.pop("image", None)
            batch_dict.pop("images", None)
            batch_dict.pop("image_backbone_feats", None)
            batch_dict.pop("image_fpn_feats", None)
            batch_dict.pop("image_fpn_fused", None)

        # 2) 点云主流程
        map_to_bev = getattr(self, "map_to_bev_module", None)

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
            if map_to_bev is not None and cur_module is map_to_bev:
                batch_dict = self.forward_multimodal(batch_dict)

        if map_to_bev is None:
            batch_dict = self.forward_multimodal(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            return {"loss": loss}, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def _ensure_naive_fuse_convs(self, num_stages: int, device: torch.device):
        """
        仅用于 use_caf_cam=False 的朴素融合：
        cat([lidar,image]) -> 2*out_ch，经 1x1 conv 压回 out_ch，保证与 Full 接口一致
        """
        if len(self.naive_fuse_convs) == num_stages:
            return
        if len(self.naive_fuse_convs) != 0:
            # 若之前以不同 num_stages 初始化过，直接清空重建（一般不会发生）
            self.naive_fuse_convs = nn.ModuleList()

        out_ch = int(self.lidar_out_ch)
        for _ in range(num_stages):
            self.naive_fuse_convs.append(
                nn.Conv2d(in_channels=2 * out_ch, out_channels=out_ch, kernel_size=1, bias=False).to(device)
            )

    def forward_multimodal(self, branch_dict):
        if not self.use_image:
            return branch_dict

        bev_feat = branch_dict.get("encoded_spconv_tensor", None)
        if bev_feat is None:
            bev_feat = branch_dict.get("spatial_features_2d", None)
        if bev_feat is None:
            return branch_dict

        if hasattr(bev_feat, "dense"):
            bev_feat = bev_feat.dense()

        if bev_feat.ndim == 5:
            bev_feat = bev_feat.max(dim=2)[0]

        # ✅lazy init lidar_align_conv：用 map_to_bev 后真实通道数创建
        if self.lidar_align_conv is None:
            in_ch_lidar = int(bev_feat.shape[1])
            self.lidar_align_conv = nn.Conv2d(
                in_channels=in_ch_lidar,
                out_channels=self.lidar_out_ch,
                kernel_size=1,
                bias=False,
            ).to(bev_feat.device)

        num_stages = int(self.cmme.num_stages)

        lidar_feats = []
        cur_feat = bev_feat
        for s in range(num_stages):
            cur_feat_aligned = self.lidar_align_conv(cur_feat)
            lidar_feats.append(cur_feat_aligned)
            if s != num_stages - 1:
                cur_feat = F.avg_pool2d(cur_feat, kernel_size=2, stride=2)

        if "image_backbone_feats" not in branch_dict:
            raise KeyError("use_image=True，但缺少 image_backbone_feats（请检查 forward 中 image_backbone 逻辑）")
        if self.image_fpn is None:
            raise RuntimeError("use_image=True 但 image_fpn 未构建，请检查 __init__")

        img_backbone_feats = branch_dict["image_backbone_feats"]

        fpn_out = self.image_fpn(img_backbone_feats)
        if isinstance(fpn_out, (tuple, list)) and len(fpn_out) == 2:
            pyramid_feats, fused_feat = fpn_out
        else:
            pyramid_feats, fused_feat = fpn_out, None

        if not isinstance(pyramid_feats, (list, tuple)):
            raise TypeError(f"image_fpn 输出异常：{type(pyramid_feats)}，期望 list/tuple[P2..]")

        branch_dict["image_fpn_feats"] = pyramid_feats
        if fused_feat is not None:
            branch_dict["image_fpn_fused"] = fused_feat

        img_multi_feats = list(pyramid_feats)
        if len(img_multi_feats) > num_stages:
            img_multi_feats = img_multi_feats[:num_stages]
        else:
            while len(img_multi_feats) < num_stages:
                img_multi_feats.append(F.avg_pool2d(img_multi_feats[-1], 2))

        aligned_img_feats = []
        for lf, imf in zip(lidar_feats, img_multi_feats):
            _, _, H_l, W_l = lf.shape
            if self.use_alignment:
                aligned = F.interpolate(imf, size=(H_l, W_l), mode="bilinear", align_corners=False)
            else:
                aligned = F.adaptive_avg_pool2d(imf, (H_l, W_l))
            aligned_img_feats.append(aligned)

        if self.use_cmme:
            enh_lidar, enh_image = self.cmme(lidar_feats, aligned_img_feats)
        else:
            enh_lidar, enh_image = lidar_feats, aligned_img_feats

        # ============================================================
        # 4) CAF-CAM 或 NaiveFusion（仅移除自适应融合+注意力，接口不变）
        # ============================================================
        if self.use_caf_cam:
            fused_feats, mc_list, ms_list = self.caf_cam(enh_lidar, enh_image)
            branch_dict["cam_mc_list"] = mc_list
            branch_dict["cam_ms_list"] = ms_list

            bev_fused = self.caf_cam.fuse_multiscale_to_bev(fused_feats, mode=self.fuse_mode)
        else:
            # ✅ NaiveFusion：stage 内 cat -> 1x1 conv 压回 out_ch（保证与 Full 宽度一致）
            self._ensure_naive_fuse_convs(num_stages=num_stages, device=bev_feat.device)

            fused_feats = []
            for i, (lf, imf) in enumerate(zip(enh_lidar, enh_image)):
                f = torch.cat([lf, imf], dim=1)              # [B, 2*out_ch, H, W]
                f = self.naive_fuse_convs[i](f)              # [B, out_ch, H, W]
                fused_feats.append(f)

            # 多尺度融合到 BEV：保持与 build_backbone_2d 的 fuse_mode 逻辑一致
            target_H, target_W = fused_feats[0].shape[2], fused_feats[0].shape[3]
            upsampled = [F.interpolate(f, size=(target_H, target_W), mode="nearest") for f in fused_feats]

            if str(self.fuse_mode).lower() == "concat":
                bev_fused = torch.cat(upsampled, dim=1)      # [B, out_ch*num_stages, H, W]
            else:
                # 非 concat：聚合到 out_ch（sum/mean 都可以，这里用 mean 更稳）
                bev_fused = torch.stack(upsampled, dim=0).mean(dim=0)  # [B, out_ch, H, W]

            # w/o CAF-CAM 时不产出注意力中间量，避免误用
            branch_dict.pop("cam_mc_list", None)
            branch_dict.pop("cam_ms_list", None)

        branch_dict["encoded_spconv_tensor"] = bev_fused
        branch_dict["spatial_features_2d"] = bev_fused
        branch_dict["lidar_features"] = enh_lidar[-1]
        branch_dict["rgb_features"] = enh_image[-1]
        return branch_dict

    def get_training_loss(self):
        disp_dict = {}
        tb_dict = {}
        loss = 0

        if hasattr(self, "dense_head") and self.dense_head is not None:
            loss_dense, tb_dict_dense = self.dense_head.get_loss()
            loss += loss_dense
            tb_dict.update(tb_dict_dense)

        if hasattr(self, "roi_head") and self.roi_head is not None:
            loss_rcnn, tb_dict_rcnn = self.roi_head.get_loss()
            loss += loss_rcnn
            tb_dict.update(tb_dict_rcnn)

        return loss, tb_dict, disp_dict
