from .detector3d_template import Detector3DTemplate
from pcdet.models.backbones_2d import image_fpn, base_bev_backbone
from pcdet.models.roi_heads.cam_modules import cmme, fusion, cam


class CaEPNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        # 加载多模态模块
        in_channels_list = model_cfg.IMAGE_FPN.IN_CHANNELS_LIST
        out_channels = model_cfg.IMAGE_FPN.OUT_CHANNELS
        self.image_fpn = image_fpn.ImageFPN(in_channels_list=in_channels_list, out_channels=out_channels)
        self.cmme = cmme.CascadeMultiModalEnhancer(model_cfg.CMME)
        self.cam = cam.CascadeAttentionModule(
            in_channels=model_cfg.CAM.IN_CHANNELS,
            attention_channels=model_cfg.CAM.ATTENTION_CHANNELS,
            num_stages=model_cfg.CAM.NUM_STAGES,
            use_transformer=model_cfg.CAM.USE_TRANSFORMER
        )
        self.fusion_module = fusion.ModalFusionBlock(
            in_channels=model_cfg.FUSION.IN_CHANNELS,
            out_channels=model_cfg.FUSION.OUT_CHANNELS,
            use_transformer=model_cfg.FUSION.USE_TRANSFORMER
        )

    # 覆盖 build_backbone_2d 方法，手动指定正确的输入通道（如256）
    def build_backbone_2d(self, model_info_dict):
        backbone_2d_module = base_bev_backbone.BaseBEVBackbone(
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=256,  # 🚨 这里手动指定融合后的通道数，比如 LiDAR 128 + 图像 128 = 256
            num_frames=self.num_frames
        )
        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        return backbone_2d_module, model_info_dict

    def forward(self, batch_dict):
        # 原始网络模块
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        # 图像特征提取与融合
        if 'images' in batch_dict:
            image_feat = self.image_fpn(batch_dict['images'])
            image_feat = self.cmme(image_feat)

            point_feat = batch_dict.get("encoded_spconv_tensor", None)
            if point_feat is not None:
                fused_feat = self.fusion_module(point_feat, image_feat)
                fused_feat = self.cam(fused_feat)
                batch_dict["encoded_spconv_tensor"] = fused_feat

        # 推理或训练
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts, batch_dict

    def get_training_loss(self):
        disp_dict = {}
        tb_dict = {}
        loss = 0

        if hasattr(self, 'dense_head') and self.dense_head is not None:
            loss_dense, tb_dict_dense = self.dense_head.get_loss()
            loss += loss_dense
            tb_dict.update(tb_dict_dense)

        if hasattr(self, 'roi_head') and self.roi_head is not None:
            loss_rcnn, tb_dict_rcnn = self.roi_head.get_loss()
            loss += loss_rcnn
            tb_dict.update(tb_dict_rcnn)

        return loss, tb_dict, disp_dict