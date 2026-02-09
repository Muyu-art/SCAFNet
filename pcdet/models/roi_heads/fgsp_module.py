# fgsp_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ForegroundScoreModule(nn.Module):
    """
    4.3.2 RoI 单元前景评分计算方法

    输入:
        roi_feats: [B, M, G, C]    RoI 内网格 / 单元特征
        roi_grid_xyz: [B, M, G, 3] 对应网格中心坐标 (RoI 内坐标或相对中心坐标)

    输出:
        fg_scores: [B, M, G, 1]    取值范围 [0, 1]
    """
    def __init__(self, feat_channels, hidden_dim=64, geo_scale=1.0, score_scale=1.0):
        super().__init__()
        # 语义分支
        self.sem_mlp = nn.Sequential(
            nn.Linear(feat_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        # 几何分支
        self.geo_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.act = nn.Sigmoid()
        self.geo_scale = geo_scale      # 几何权重 γ
        self.score_scale = score_scale  # 分数缩放(温度)

    def forward(self, roi_feats, roi_grid_xyz):
        B, M, G, C = roi_feats.shape
        sem_in = roi_feats.contiguous().view(B * M * G, C)
        geo_in = roi_grid_xyz.contiguous().view(B * M * G, 3)

        sem_score = self.sem_mlp(sem_in)  # [B*M*G, 1]
        geo_score = self.geo_mlp(geo_in)  # [B*M*G, 1]

        # (sem + γ*geo) * score_scale -> sigmoid
        score = (sem_score + self.geo_scale * geo_score) * self.score_scale
        score = self.act(score)

        fg_scores = score.view(B, M, G, 1)
        return fg_scores


class SupportPointConstructor(nn.Module):
    """
    基于前景评分的支撑点构建方法 (4.3.3 的前半部分)

    - select_with_score = True:
        使用 fg_scores 在每个 RoI 内选择 Top-K 单元作为支撑点。
        若实际单元数 G < K，则在 Top-G 的基础上重复最高分单元，填满 K 个支撑点。
    - select_with_score = False:
        不使用前景分数，按照网格索引均匀选取 K 个单元，可作为 Random-SP 的确定性实现。
    """
    def __init__(self, num_support_points=8, select_with_score=True):
        super().__init__()
        self.num_support = num_support_points
        self.select_with_score = select_with_score

    def forward(self, roi_feats, roi_grid_xyz, fg_scores):
        B, M, G, C = roi_feats.shape
        K = self.num_support

        if self.select_with_score:
            scores_flat = fg_scores.squeeze(-1)  # [B, M, G]
            K_eff = min(K, G)

            # torch.topk(sorted=True) -> 从大到小（用于“选点”）
            topk_scores, topk_idx = torch.topk(
                scores_flat, k=K_eff, dim=-1, sorted=True
            )  # [B, M, K_eff]

            # 若 G < K：重复最高分单元填充（注意：[:1] 才是最高分）
            if K_eff < K:
                best_idx = topk_idx[..., :1]  # [B, M, 1]
                pad_idx = best_idx.expand(B, M, K - K_eff)
                topk_idx = torch.cat([topk_idx, pad_idx], dim=-1)

                best_score = topk_scores[..., :1]  # [B, M, 1]
                pad_score = best_score.expand(B, M, K - K_eff)
                topk_scores = torch.cat([topk_scores, pad_score], dim=-1)

            # 必改：稳定排序（用于对顺序敏感的后续重组）
            # 选点仍由分数决定，但输出顺序固定为网格索引升序，减少抖动。
            topk_idx, _ = torch.sort(topk_idx, dim=-1)

        else:
            device = roi_feats.device
            base_idx = torch.linspace(0, max(G - 1, 0), steps=K, device=device).round().long()  # [K]
            topk_idx = base_idx.view(1, 1, K).expand(B, M, K)

            scores_flat = fg_scores.squeeze(-1)
            topk_scores = torch.gather(scores_flat, dim=-1, index=topk_idx.clamp(0, G - 1))

        idx_feat = topk_idx.unsqueeze(-1).expand(-1, -1, -1, C)
        idx_xyz = topk_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        idx_score = topk_idx.unsqueeze(-1)

        sp_seed_feats = torch.gather(roi_feats, dim=2, index=idx_feat)
        sp_xyz = torch.gather(roi_grid_xyz, dim=2, index=idx_xyz)
        sp_scores = torch.gather(fg_scores, dim=2, index=idx_score)

        return sp_xyz, sp_seed_feats, sp_scores


class MultiRadiusAggregator(nn.Module):
    """
    4.3.4 多感受域前景支撑点的特征聚合方法（增强版）
    """
    def __init__(
        self,
        in_channels,
        radii=(0.4, 0.8, 1.2),
        out_channels=None,
        use_softmax_weight=False,
        topk_neighbors=16,
        residual_scale=0.3,
        eps=1e-6,
        use_scale_gating=False,
        gating_mode="roi",
        gate_init_bias=2.0,
    ):
        super().__init__()
        self.radii = list(radii)
        self.num_scales = len(self.radii)
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        self.use_softmax_weight = use_softmax_weight
        self.topk_neighbors = topk_neighbors
        self.residual_scale = residual_scale
        self.eps = eps

        self.use_scale_gating = use_scale_gating
        assert gating_mode in ("roi", "sp")
        self.gating_mode = gating_mode
        self.gate_init_bias = gate_init_bias

        self.scale_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(self.num_scales)
        ])

        self.fuse_mlp = nn.Sequential(
            nn.Linear(in_channels * self.num_scales, out_channels),
            nn.ReLU(inplace=True)
        )

        if self.use_scale_gating:
            self.gate_net = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels, self.num_scales)
            )
            nn.init.zeros_(self.gate_net[-1].weight)
            nn.init.constant_(self.gate_net[-1].bias, self.gate_init_bias)

    def forward(
        self,
        sp_xyz,
        roi_grid_xyz,
        roi_feats,
        fg_scores=None,
        sp_seed_feats=None,
        return_vis=False,
        vis_use_scale="largest",
    ):
        B, M, K, _ = sp_xyz.shape
        _, _, G, C = roi_feats.shape

        sp = sp_xyz.unsqueeze(-2)         # [B, M, K, 1, 3]
        pts = roi_grid_xyz.unsqueeze(-3)  # [B, M, 1, G, 3]
        dist = torch.norm(sp - pts, dim=-1)  # [B, M, K, G]

        if fg_scores is not None:
            alpha = fg_scores.squeeze(-1)               # [B, M, G]
            alpha = alpha.unsqueeze(2).expand(B, M, K, G)
        else:
            alpha = None

        feats_exp = roi_feats.unsqueeze(2)  # [B, M, 1, G, C]

        gates = None
        if self.use_scale_gating and (sp_seed_feats is not None):
            if self.gating_mode == "roi":
                roi_ctx = sp_seed_feats.mean(dim=2)          # [B, M, C]
                gate_logits = self.gate_net(roi_ctx)         # [B, M, S]
                gates = torch.sigmoid(gate_logits)           # [B, M, S]
            else:
                gate_logits = self.gate_net(sp_seed_feats)   # [B, M, K, S]
                gates = torch.sigmoid(gate_logits)           # [B, M, K, S]

        multi_scale_feats = []
        weights_per_scale = []
        masks_per_scale = []

        for s, r in enumerate(self.radii):
            mask = (dist <= r)          # [B, M, K, G] bool
            mask_f = mask.float()

            has_neighbor = mask.any(dim=-1)      # [B, M, K]
            no_neighbor = ~has_neighbor          # [B, M, K]

            if alpha is not None:
                base_w = alpha * mask_f
            else:
                base_w = mask_f

            topk_applied = (
                self.topk_neighbors is not None
                and self.topk_neighbors > 0
                and self.topk_neighbors < G
            )
            if topk_applied:
                B2 = B * M * K
                base_w_flat = base_w.view(B2, G)
                k_eff = min(self.topk_neighbors, G)
                topk_vals, topk_idx = torch.topk(base_w_flat, k=k_eff, dim=-1)

                new_w_flat = torch.zeros_like(base_w_flat)
                new_w_flat.scatter_(1, topk_idx, topk_vals)
                base_w = new_w_flat.view(B, M, K, G)

            if self.use_softmax_weight:
                if alpha is not None:
                    log_alpha = torch.log(alpha + self.eps)
                else:
                    log_alpha = torch.zeros_like(dist)

                logits = log_alpha - dist / (r + self.eps)

                valid_mask = (base_w > 0) if topk_applied else mask

                valid_cnt = valid_mask.sum(dim=-1, keepdim=True)
                need_fb = (valid_cnt == 0) & has_neighbor.unsqueeze(-1)
                if need_fb.any():
                    valid_mask = torch.where(need_fb.expand_as(valid_mask), mask, valid_mask)

                logits = logits.masked_fill(~valid_mask, float("-inf"))
                weights = torch.softmax(logits, dim=-1)
                weights[no_neighbor.unsqueeze(-1).expand_as(weights)] = 0.0
            else:
                w = base_w
                w_sum = w.sum(dim=-1, keepdim=True)

                fallback = (w_sum <= self.eps) & has_neighbor.unsqueeze(-1)
                if fallback.any():
                    w = torch.where(fallback.expand_as(w), mask_f, w)
                    w_sum = w.sum(dim=-1, keepdim=True)

                weights = w / (w_sum + self.eps)
                weights[no_neighbor.unsqueeze(-1).expand_as(weights)] = 0.0

            feats_scale = (weights.unsqueeze(-1) * feats_exp).sum(dim=3)  # [B,M,K,C]

            if sp_seed_feats is not None:
                feats_scale[no_neighbor] = sp_seed_feats[no_neighbor]

            x = feats_scale.contiguous().view(B * M * K, C)
            x = self.scale_mlps[s](x)
            x = x.view(B, M, K, C)

            if gates is not None:
                if self.gating_mode == "roi":
                    g = gates[..., s].unsqueeze(-1).unsqueeze(-1)
                    x = x * g
                else:
                    g = gates[..., s].unsqueeze(-1)
                    x = x * g

            multi_scale_feats.append(x)

            if return_vis:
                weights_per_scale.append(weights.detach())
                masks_per_scale.append(mask.detach())

        ms_cat = torch.cat(multi_scale_feats, dim=-1)  # [B,M,K,C*num_scales]
        ms_feat = self.fuse_mlp(ms_cat.contiguous().view(B * M * K, -1))
        ms_feat = ms_feat.view(B, M, K, self.out_channels)

        if sp_seed_feats is not None and self.out_channels == sp_seed_feats.shape[-1]:
            sp_out = sp_seed_feats + self.residual_scale * ms_feat
        else:
            sp_out = ms_feat

        if not return_vis:
            return sp_out

        if len(weights_per_scale) == 0:
            weights_vis = None
        else:
            if vis_use_scale == "mean":
                w = torch.stack(weights_per_scale, dim=0).mean(dim=0)
                w_sum = w.sum(dim=-1, keepdim=True) + self.eps
                weights_vis = w / w_sum
            else:
                weights_vis = weights_per_scale[-1]

        vis = {
            "radii": self.radii,
            "weights_per_scale": weights_per_scale,
            "radius_masks": masks_per_scale,
            "weights_vis": weights_vis,
        }
        return sp_out, vis


class RoIRebuildFromSupportPoints(nn.Module):
    """
     最终版（稳定）：用“集合池化”替代“拼接展平”，降低对支撑点顺序/抖动敏感性。

    输入:
        sp_feats:  [B, M, K, C]
        sp_scores: [B, M, K, 1] 或 [B, M, K]（可选）
    输出:
        roi_repr:  [B, M, out_channels]
    """
    def __init__(self, in_channels, num_support_points, out_channels=256, use_score_pool=True, eps=1e-6):
        super().__init__()
        self.num_support = num_support_points
        self.out_channels = out_channels
        self.use_score_pool = use_score_pool
        self.eps = eps

        # pooled(C) -> out_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, sp_feats, sp_scores=None):
        B, M, K, C = sp_feats.shape
        assert K == self.num_support

        # 默认均值池化（最稳）
        if (not self.use_score_pool) or (sp_scores is None):
            pooled = sp_feats.mean(dim=2)  # [B, M, C]
        else:
            if sp_scores.dim() == 4:
                w = sp_scores.squeeze(-1)  # [B, M, K]
            else:
                w = sp_scores              # [B, M, K]

            # softmax 归一化更稳（对整体尺度不敏感）
            w = torch.softmax(w, dim=-1)   # [B, M, K]
            pooled = (sp_feats * w.unsqueeze(-1)).sum(dim=2)  # [B, M, C]

        roi_repr = self.mlp(pooled)
        return roi_repr


class FGSPRoIEncoder(nn.Module):
    """
    FGSP RoI 编码模块:
    - 计算 RoI 内单元前景评分;
    - 构建前景支撑点;
    - 可选多半径特征聚合;
    - 用集合池化重组 RoI 表示（替代拼接展平）
    """
    def __init__(
        self,
        feat_channels,
        num_support_points=8,
        radii=(0.4, 0.8, 1.2),
        out_channels=256,
        use_multi_radius=True,
        use_softmax_weight=False,
        select_with_score=True,
        geo_scale=1.0,
        score_scale=1.0,
        use_score_weight=True,
        topk_neighbors=16,
        residual_scale=0.3,
        # 新增：是否在重组时使用支撑点评分做加权池化
        use_score_pool=True,
    ):
        super().__init__()
        self.use_multi_radius = use_multi_radius
        self.select_with_score = select_with_score
        self.use_score_weight = use_score_weight

        self.fg_score_module = ForegroundScoreModule(
            feat_channels=feat_channels,
            hidden_dim=64,
            geo_scale=geo_scale,
            score_scale=score_scale,
        )
        self.support_point_module = SupportPointConstructor(
            num_support_points=num_support_points,
            select_with_score=select_with_score
        )

        if self.use_multi_radius:
            self.multi_radius_agg = MultiRadiusAggregator(
                in_channels=feat_channels,
                radii=radii,
                out_channels=feat_channels,
                use_softmax_weight=use_softmax_weight,
                topk_neighbors=topk_neighbors,
                residual_scale=residual_scale,
            )

        self.roi_rebuild_module = RoIRebuildFromSupportPoints(
            in_channels=feat_channels,
            num_support_points=num_support_points,
            out_channels=out_channels,
            use_score_pool=use_score_pool,
        )

    def forward(self, roi_feats, roi_grid_xyz, return_vis=False, vis_use_scale="largest"):
        fg_scores = self.fg_score_module(roi_feats, roi_grid_xyz)

        sp_xyz, sp_seed_feats, sp_scores = self.support_point_module(
            roi_feats, roi_grid_xyz, fg_scores
        )

        vis = None
        if self.use_multi_radius:
            fg_for_agg = fg_scores if self.use_score_weight else None

            if return_vis:
                sp_feats, vis = self.multi_radius_agg(
                    sp_xyz, roi_grid_xyz, roi_feats,
                    fg_scores=fg_for_agg,
                    sp_seed_feats=sp_seed_feats,
                    return_vis=True,
                    vis_use_scale=vis_use_scale,
                )
            else:
                sp_feats = self.multi_radius_agg(
                    sp_xyz, roi_grid_xyz, roi_feats,
                    fg_scores=fg_for_agg,
                    sp_seed_feats=sp_seed_feats,
                    return_vis=False,
                )
        else:
            sp_feats = sp_seed_feats

        # 必改：集合池化重组（传入 sp_scores 做加权池化）
        roi_repr = self.roi_rebuild_module(sp_feats, sp_scores=sp_scores)

        aux = {
            "fg_scores": fg_scores,
            "sp_xyz": sp_xyz,
            "sp_scores": sp_scores,
            "sp_feats": sp_feats,
        }
        if vis is not None:
            aux["vis"] = vis

        return roi_repr, aux
