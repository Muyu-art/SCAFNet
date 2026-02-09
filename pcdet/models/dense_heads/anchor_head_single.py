import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate
import torch
import cv2
import numpy as np
import torch.nn.functional as F

def get_layer(dim,out_dim,init = None):
    init_func = nn.init.kaiming_normal_
    layers = []
    conv = nn.Conv2d(dim, dim,
                      kernel_size=3, padding=1, bias=True)
    nn.init.normal_(conv.weight, mean=0, std=0.001)
    layers.append(conv)
    layers.append(nn.BatchNorm2d(dim))
    layers.append(nn.ReLU())
    conv2 = nn.Conv2d(dim, out_dim,
                     kernel_size=1, bias=True)

    if init is None:
        nn.init.normal_(conv2.weight, mean=0, std=0.001)
        layers.append(conv2)

    else:
        conv2.bias.data.fill_(init)
        layers.append(conv2)

    return nn.Sequential(*layers)

class AnchorHeadSingleV2(AnchorHeadTemplate):
    def __init__(self, model_cfg, num_frames, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg,num_frames=num_frames, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        self.grid_size = grid_size  # [1408 1600   40]
        self.range = point_cloud_range

        self.voxel_size = (point_cloud_range[3] - point_cloud_range[0]) / grid_size[0]


        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        shard_c = 64

        self.shared_conv = nn.Sequential(
            nn.Conv2d(input_channels, shard_c,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(shard_c),
            nn.ReLU(inplace=True)
        )

        self.conv_cls = get_layer(shard_c,self.num_anchors_per_location * self.num_class,-4.59)

        self.conv_reg = get_layer(shard_c,self.num_anchors_per_location * 2)
        self.conv_height = get_layer(shard_c,self.num_anchors_per_location * 1)

        self.conv_dim = get_layer(shard_c,self.num_anchors_per_location * 3)

        self.conv_ang = get_layer(shard_c,self.num_anchors_per_location * 1)

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:

            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        #self.init_weights()

        #for child in self.children():
        #    for param in child.parameters():
        #        param.requires_grad = False

    def init_weights(self):

        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def get_anchor_mask(self,data_dict,shape):

        stride = np.round(self.voxel_size*8.*10.)

        minx=self.range[0]
        miny=self.range[1]

        points = data_dict["points"]

        mask = torch.zeros(shape[-2],shape[-1])

        mask_large = torch.zeros(shape[-2]//10,shape[-1]//10)

        in_x = (points[:, 1] - minx) / stride
        in_y = (points[:, 2] - miny) / stride

        in_x = in_x.long().clamp(max=shape[-1]//10-1)
        in_y = in_y.long().clamp(max=shape[-2]//10-1)


        mask_large[in_y,in_x] = 1

        mask_large = mask_large.clone().int().detach().cpu().numpy()

        mask_large_index = np.argwhere( mask_large>0 )

        mask_large_index = mask_large_index*10

        index_list=[]

        for i in np.arange(-10, 10, 1):
            for j in np.arange(-10, 10, 1):
                index_list.append(mask_large_index+[i,j])

        index_list = np.concatenate(index_list,0)

        inds = torch.from_numpy(index_list).cuda().long()

        mask[inds[:,0],inds[:,1]]=1

        return mask.bool()




    def forward(self, data_dict):

        anchor_mask = self.get_anchor_mask(data_dict, data_dict['st_features_2d'].shape)

        new_anchors = []
        for anchors in self.anchors_root:
            new_anchors.append(anchors[:, anchor_mask, ...])

        self.anchors = new_anchors


        for i in range(self.num_frames):
            if i==0:
                frame_id = ''
            else:
                frame_id = str(-i)
            if 'st_features_2d'+frame_id not in data_dict:
                continue
            st_features_2d = data_dict['st_features_2d'+frame_id]

            shard = self.shared_conv(st_features_2d)

            cls_feat = shard

            reg_feat = shard

            cls_preds = self.conv_cls(cls_feat)

            box_reg = self.conv_reg(reg_feat)
            box_height = self.conv_height(reg_feat)
            box_dim = self.conv_dim(reg_feat)
            box_ang = self.conv_ang(reg_feat)

            box_preds = torch.cat([box_reg,box_height,box_dim,box_ang],dim=1)

            cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]  # [N, H, W, C]
            box_preds = box_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]  # [N, H, W, C]

            self.forward_ret_dict['cls_preds'+frame_id] = cls_preds
            self.forward_ret_dict['box_preds'+frame_id] = box_preds

            if self.conv_dir_cls is not None:
                dir_cls_preds = self.conv_dir_cls(st_features_2d)
                dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]
                self.forward_ret_dict['dir_cls_preds'+frame_id] = dir_cls_preds
            else:
                dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)
            data_dict['gt_ious'] = targets_dict['gt_ious']

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, num_frames, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg,num_frames=num_frames, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        self.grid_size = grid_size  # [1408 1600   40]
        self.range = point_cloud_range

        self.voxel_size = (point_cloud_range[3] - point_cloud_range[0]) / grid_size[0]


        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )


        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

        #for child in self.children():
        #    for param in child.parameters():
        #        param.requires_grad = False

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def get_anchor_mask(self,data_dict,shape):

        stride = np.round(self.voxel_size*8.*10.)

        minx=self.range[0]
        miny=self.range[1]

        points = data_dict["points"]

        mask = torch.zeros(shape[-2],shape[-1])

        mask_large = torch.zeros(shape[-2]//10,shape[-1]//10)

        in_x = (points[:, 1] - minx) / stride
        in_y = (points[:, 2] - miny) / stride

        in_x = in_x.long().clamp(max=shape[-1]//10-1)
        in_y = in_y.long().clamp(max=shape[-2]//10-1)


        mask_large[in_y,in_x] = 1

        mask_large = mask_large.clone().int().detach().cpu().numpy()

        mask_large_index = np.argwhere( mask_large>0 )

        mask_large_index = mask_large_index*10

        index_list=[]

        for i in np.arange(-10, 10, 1):
            for j in np.arange(-10, 10, 1):
                index_list.append(mask_large_index+[i,j])

        index_list = np.concatenate(index_list,0)

        inds = torch.from_numpy(index_list).cuda().long()

        mask[inds[:,0],inds[:,1]]=1

        return mask.bool()

    def compute_semantic_distillation_loss(self,
                                           rgb_logits,
                                           lidar_logits,
                                           attn_c=None,
                                           attn_s=None):
        assert rgb_logits.shape == lidar_logits.shape, \
            f"SDL logits 形状不一致: rgb={rgb_logits.shape}, lidar={lidar_logits.shape}"

        x_rgb = rgb_logits
        x_lidar = lidar_logits

        if x_rgb.dim() != 4:
            raise ValueError(f"期望 4D tensor，得到 {x_rgb.dim()}D")

        # -------- 修复：正确判断 BCHW / BHWC --------
        # 经验规则：H/W 通常比类别数大很多；类别数通常很小（KITTI=3 等）
        # 如果最后一维更像“类别维”，认为是 BHWC；否则认为是 BCHW。
        if x_rgb.shape[-1] <= 32 and x_rgb.shape[1] > 32:
            # 更像 BHWC: [B,H,W,C]，无需 permute
            pass
        else:
            # 更像 BCHW: [B,C,H,W] -> [B,H,W,C]
            x_rgb = x_rgb.permute(0, 2, 3, 1).contiguous()
            x_lidar = x_lidar.permute(0, 2, 3, 1).contiguous()

        P_I = F.softmax(x_rgb, dim=-1)
        P_L = F.softmax(x_lidar, dim=-1)
        P_A = 0.5 * (P_I + P_L)

        eps = 1e-6
        log_P_I = torch.log(P_I.clamp(min=eps))
        log_P_L = torch.log(P_L.clamp(min=eps))
        log_P_A = torch.log(P_A.clamp(min=eps))

        kl_I = (P_I * (log_P_I - log_P_A)).sum(dim=-1)  # [B,H,W]
        kl_L = (P_L * (log_P_L - log_P_A)).sum(dim=-1)  # [B,H,W]
        L = kl_I + kl_L

        if (attn_c is not None) and (attn_s is not None):
            Mc = attn_c.mean(dim=1, keepdim=True)  # [B,1,1,1]
            Mc = Mc.expand(-1, 1, attn_s.shape[2], attn_s.shape[3])  # [B,1,H,W]
            M = (Mc * attn_s).squeeze(1)  # [B,H,W]
            M = M / (M.mean() + eps)
            L = L * M

        return L.mean()

    def forward(self, data_dict):

        anchor_mask = self.get_anchor_mask(data_dict,data_dict['st_features_2d'].shape)

        new_anchors = []
        for anchors in self.anchors_root:
            new_anchors.append(anchors[:, anchor_mask, ...])

        self.anchors = new_anchors

        for i in range(self.num_frames):
            if i==0:
                st_features_2d = data_dict['st_features_2d']

                cls_preds = self.conv_cls(st_features_2d)
                box_preds = self.conv_box(st_features_2d)

                cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]  # [N, H, W, C]
                box_preds = box_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]  # [N, H, W, C]

                self.forward_ret_dict['cls_preds'] = cls_preds
                self.forward_ret_dict['box_preds'] = box_preds

                if self.conv_dir_cls is not None:
                    dir_cls_preds = self.conv_dir_cls(st_features_2d)
                    dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]
                    self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
                else:
                    dir_cls_preds = None

            else:
                if 'st_features_2d'+str(-i) not in data_dict:
                    continue
                st_features_2d = data_dict['st_features_2d'+str(-i)]

                cls_preds2 = self.conv_cls(st_features_2d)
                box_preds2 = self.conv_box(st_features_2d)


                cls_preds2 = cls_preds2.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
                box_preds2 = box_preds2.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]


                self.forward_ret_dict['cls_preds'+str(-i)] = cls_preds2
                self.forward_ret_dict['box_preds'+str(-i)] = box_preds2

                if self.conv_dir_cls is not None:
                    dir_cls_preds2 = self.conv_dir_cls(st_features_2d)
                    dir_cls_preds2 = dir_cls_preds2.permute(0, 2, 3, 1).contiguous()
                    self.forward_ret_dict['dir_cls_preds'+str(-i)] = dir_cls_preds2
                else:
                    dir_cls_preds2 = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)
            data_dict['gt_ious'] = targets_dict['gt_ious']

            # ====== 计算语义蒸馏损失 SDL ======
            device = cls_preds.device  # 使用当前 head 的设备

            # 从 data_dict 中取出 LiDAR / Image 语义 logits
            # 这里继续沿用你原来的名字，如果你之后改成其他命名，只要同步改这里即可
            rgb_logits = data_dict.get('rgb_features', None)
            lidar_logits = data_dict.get('lidar_features', None)

            # 从 CAFCAMModule 输出中取出最后一层注意力（可选）
            attn_c = None
            attn_s = None
            if 'cam_mc_list' in data_dict and 'cam_ms_list' in data_dict:
                mc_list = data_dict['cam_mc_list']  # List[Tensor]
                ms_list = data_dict['cam_ms_list']
                if isinstance(mc_list, (list, tuple)) and len(mc_list) > 0:
                    attn_c = mc_list[-1]
                if isinstance(ms_list, (list, tuple)) and len(ms_list) > 0:
                    attn_s = ms_list[-1]

            # ==========================
            # SDL 开关（消融逻辑）
            # ==========================
            if not self.model_cfg.get('USE_SDL', True):
                # 消融：不使用 SDL，loss 设为 0
                semantic_distill_loss = torch.zeros(1, device=device)
            else:
                # 正常训练 SDL
                if (rgb_logits is not None) and (lidar_logits is not None):
                    semantic_distill_loss = self.compute_semantic_distillation_loss(
                        rgb_logits, lidar_logits, attn_c=attn_c, attn_s=attn_s
                    )
                else:
                    semantic_distill_loss = torch.zeros(1, device=device)

            # if (rgb_logits is not None) and (lidar_logits is not None):
            #     semantic_distill_loss = self.compute_semantic_distillation_loss(
            #         rgb_logits, lidar_logits, attn_c=attn_c, attn_s=attn_s
            #     )
            # else:
            #     semantic_distill_loss = torch.zeros(1, device=device)

            # 存到 data_dict 和 forward_ret_dict，方便后续 get_loss 使用
            data_dict['semantic_distill_loss'] = semantic_distill_loss
            self.forward_ret_dict['semantic_distill_loss'] = semantic_distill_loss

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        if self.model_cfg.get('NMS_CONFIG', None) is not None:
            self.proposal_layer(
                data_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
            )

        return data_dict
