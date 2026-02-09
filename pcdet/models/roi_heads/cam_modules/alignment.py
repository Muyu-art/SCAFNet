import torch
import torch.nn as nn
import torch.nn.functional as F


class DualModalFeatureAligner(nn.Module):
    """
    点级对齐：LiDAR 点 (velo) -> rect -> P2 -> 像素(u,v) -> 缩放到 resize 后图像 -> 映射到 P2(feature) -> grid_sample
    输出：
      - batch_dict['point_img_feats'] : [M_valid, C]
      - batch_dict['points_valid']    : [M_valid, D] (原 points 中有效点)
      - batch_dict['point_valid_mask']: [M] bool (在原 points 上的有效 mask)
    其中 M 是 batch 内所有点拼接后的总点数（points[:,0] 是 batch_idx）
    """

    def __init__(self, img_size_hw=(256, 704), p2_stride=4):
        """
        Args:
            img_size_hw: DataProcessor process_image_features 后的最终输入图像尺寸 (H, W)
            p2_stride:   P2(或C2) 相对输入图像的 stride，ResNet18 的 C2 = 4
        """
        super().__init__()
        self.img_h, self.img_w = int(img_size_hw[0]), int(img_size_hw[1])
        self.p2_stride = float(p2_stride)

    @staticmethod
    def _get_calib_mats(calib, device, dtype=torch.float32):
        """
        兼容：
        - PCDet calibration_kitti.Calibration 对象：有 P2 / V2C / R0
        - dict: {'P2':..., 'Tr_velo_to_cam':..., 'R0_rect':...}（你 infos 里那种）
        返回 torch.Tensor:
          P2: [3,4]
          V2C: [3,4]
          R0: [3,3]
        """
        if isinstance(calib, dict):
            # infos 里常见：P2 是 [4,4] 或 [3,4]；Tr_velo_to_cam 可能是 [4,4]；R0_rect 可能 [4,4]
            P2 = calib.get("P2", None)
            V2C = calib.get("Tr_velo_to_cam", None)
            R0 = calib.get("R0_rect", None)

            assert P2 is not None and V2C is not None and R0 is not None, \
                f"calib dict missing keys, got: {list(calib.keys())}"

            P2 = torch.as_tensor(P2, device=device, dtype=dtype)
            V2C = torch.as_tensor(V2C, device=device, dtype=dtype)
            R0 = torch.as_tensor(R0, device=device, dtype=dtype)

            # 裁成需要的形状
            if P2.shape == (4, 4):
                P2 = P2[:3, :4]
            else:
                P2 = P2[:3, :4]

            # Tr_velo_to_cam：可能 [4,4]，取前3行4列
            if V2C.shape == (4, 4):
                V2C = V2C[:3, :4]
            else:
                V2C = V2C[:3, :4]

            # R0_rect：可能 [4,4]，取左上3x3
            if R0.shape == (4, 4):
                R0 = R0[:3, :3]
            else:
                R0 = R0[:3, :3]

            return P2, V2C, R0

        # Calibration object
        # 常见属性名：P2, V2C, R0
        P2 = torch.as_tensor(getattr(calib, "P2"), device=device, dtype=dtype)   # [3,4]
        V2C = torch.as_tensor(getattr(calib, "V2C"), device=device, dtype=dtype) # [3,4]
        R0  = torch.as_tensor(getattr(calib, "R0"), device=device, dtype=dtype)  # [3,3]
        return P2, V2C, R0

    @staticmethod
    def _velo_to_rect(xyz_velo, V2C, R0):
        """
        xyz_velo: [N,3]
        V2C: [3,4]  Tr_velo_to_cam
        R0:  [3,3]  R0_rect
        return xyz_rect: [N,3]
        """
        # cam = V2C * [x,y,z,1]
        ones = torch.ones((xyz_velo.shape[0], 1), device=xyz_velo.device, dtype=xyz_velo.dtype)
        xyz1 = torch.cat([xyz_velo, ones], dim=1)                    # [N,4]
        xyz_cam = xyz1 @ V2C.t()                                     # [N,3]
        xyz_rect = xyz_cam @ R0.t()                                  # [N,3]
        return xyz_rect

    @staticmethod
    def _rect_to_img(xyz_rect, P2):
        """
        xyz_rect: [N,3]
        P2: [3,4]
        return:
          uv: [N,2] (pixel coords on ORIGINAL image coordinate system)
          depth: [N]
        """
        ones = torch.ones((xyz_rect.shape[0], 1), device=xyz_rect.device, dtype=xyz_rect.dtype)
        xyz1 = torch.cat([xyz_rect, ones], dim=1)                    # [N,4]
        uvw = xyz1 @ P2.t()                                          # [N,3]
        depth = uvw[:, 2].clamp(min=1e-6)
        u = uvw[:, 0] / depth
        v = uvw[:, 1] / depth
        uv = torch.stack([u, v], dim=1)
        return uv, depth

    @staticmethod
    def _normalize_to_grid(u, v, W, H):
        """
        将 feature-map 像素坐标 (u,v) -> [-1,1] grid
        align_corners=False 的常用写法：x = (u/(W-1))*2-1
        """
        x = (u / (W - 1)) * 2 - 1
        y = (v / (H - 1)) * 2 - 1
        return torch.stack([x, y], dim=-1)  # [N,2]

    def forward(self, batch_dict, pyramid_feats):
        """
        Args:
            batch_dict:
              - 'points': [M, 4(+C)], points[:,0] = batch_idx, points[:,1:4] = xyz_velo
              - 'calib' : 通常是 list[Calibration]，也可能是 list[dict] 或单个对象（B=1）
              - 'image_shape': [B,2] or list/tuple (H,W)  (ORIGINAL image size, e.g., 375x1242)
            pyramid_feats:
              - list[P2,P3,...]，其中 P2 为最高分辨率特征 [B,C,Hf,Wf]
        """
        assert "points" in batch_dict, "batch_dict missing 'points'"
        assert isinstance(pyramid_feats, (list, tuple)) and len(pyramid_feats) > 0, "pyramid_feats invalid"

        points = batch_dict["points"]
        device = points.device
        dtype = torch.float32

        batch_idx = points[:, 0].long()
        xyz_velo = points[:, 1:4].to(dtype)

        P2_feat = pyramid_feats[0]  # [B,C,Hf,Wf]
        B, C, Hf, Wf = P2_feat.shape

        # calib 兼容：B=1 可能是单个对象；B>1 常是 list
        calib = batch_dict.get("calib", None)
        assert calib is not None, "batch_dict missing 'calib'"

        if not isinstance(calib, (list, tuple)):
            calib_list = [calib]
        else:
            calib_list = list(calib)
        assert len(calib_list) == B, f"calib batch mismatch: len(calib)={len(calib_list)} vs B={B}"

        # 原图尺寸 image_shape：常是 [B,2] numpy/torch，也可能 list
        img_shape = batch_dict.get("image_shape", None)
        assert img_shape is not None, "batch_dict missing 'image_shape' (original H,W)"
        if isinstance(img_shape, torch.Tensor):
            img_shape_t = img_shape.to(device=device)
        else:
            img_shape_t = torch.as_tensor(img_shape, device=device)

        # 统一成 [B,2] = (H,W)
        if img_shape_t.ndim == 1:
            img_shape_t = img_shape_t.view(1, 2).repeat(B, 1)
        assert img_shape_t.shape[0] == B and img_shape_t.shape[1] == 2, f"image_shape bad: {tuple(img_shape_t.shape)}"

        # 输出容器（按原 points 顺序）
        valid_mask = torch.zeros((points.shape[0],), device=device, dtype=torch.bool)
        feats_out = torch.zeros((points.shape[0], C), device=device, dtype=P2_feat.dtype)

        # 逐 batch 处理
        for b in range(B):
            sel = (batch_idx == b)
            if sel.sum() == 0:
                continue

            xyz_b = xyz_velo[sel]  # [Nb,3]

            # calib mats
            P2, V2C, R0 = self._get_calib_mats(calib_list[b], device=device, dtype=dtype)

            # velo -> rect
            xyz_rect = self._velo_to_rect(xyz_b, V2C, R0)

            # rect -> original pixel uv
            uv, depth = self._rect_to_img(xyz_rect, P2)

            H_org = float(img_shape_t[b, 0].item())
            W_org = float(img_shape_t[b, 1].item())

            # 有效点：在相机前方 + 落在原图内
            u_org = uv[:, 0]
            v_org = uv[:, 1]
            m = (depth > 0) & (u_org >= 0) & (u_org < W_org) & (v_org >= 0) & (v_org < H_org)
            if m.sum() == 0:
                continue

            # 原图 uv -> resize 后图像 uv
            sx = self.img_w / W_org
            sy = self.img_h / H_org
            u_rs = u_org[m] * sx
            v_rs = v_org[m] * sy

            # resize 后图像 uv -> P2(feature) 像素坐标（除 stride）
            u_f = u_rs / self.p2_stride
            v_f = v_rs / self.p2_stride

            # 再做一次 feature-map 范围过滤
            m2 = (u_f >= 0) & (u_f < (Wf - 1)) & (v_f >= 0) & (v_f < (Hf - 1))
            if m2.sum() == 0:
                continue

            u_f = u_f[m2]
            v_f = v_f[m2]

            # grid_sample 需要 [1, N, 1, 2]
            grid = self._normalize_to_grid(u_f, v_f, Wf, Hf).view(1, -1, 1, 2)

            # 采样该 batch 的 P2 feature
            feat_b = P2_feat[b:b+1]  # [1,C,Hf,Wf]
            sampled = F.grid_sample(
                feat_b, grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False
            )  # [1,C,N,1]
            sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)  # [N,C]

            # 把有效点写回全局
            # idx_sel: 原 points 的下标
            idx_sel = torch.nonzero(sel, as_tuple=False).view(-1)          # [Nb]
            idx_m = idx_sel[m]                                             # [N_m]
            idx_m2 = idx_m[m2]                                             # [N_m2]

            feats_out[idx_m2] = sampled.to(P2_feat.dtype)
            valid_mask[idx_m2] = True

        # 打包输出（只保留有效点）
        batch_dict["point_valid_mask"] = valid_mask
        batch_dict["points_valid"] = points[valid_mask]
        batch_dict["point_img_feats"] = feats_out[valid_mask]  # [M_valid,C]
        return batch_dict
