import torch
import scipy.linalg as linalg
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy

from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from .make_model import weights_init_classifier, weights_init_kaiming
from .backbones.vit_pytorch import trunc_normal_
from .neck.decoder import PartDecoder1
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .part_model import PixelToPartClassifier, GlobalWeightedPooling

class VisibilityPredictor(nn.Module):
    def __init__(self, embed_dim, out_channel=1):
        super(VisibilityPredictor, self).__init__()
        self.bn = nn.BatchNorm1d(embed_dim)
        self.classifier = nn.Linear(embed_dim, out_channel, bias=False)
        self._init_params()
    
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.bn(x).permute(0,2,1)
        return self.classifier(x)
    
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)  # ResNet = 0.01, Bof and ISP-reid = 0.001
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class build_transformer_tsd(nn.Module):
    __factory_dict = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    }
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer_tsd, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = self.__factory_dict[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                        local_feature=True,
                                                        )
        self.in_planes = self.base.embed_dim
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.num_x = self.base.patch_embed.num_x
        self.num_y = self.base.patch_embed.num_y

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        ######## decoder ############
        embed_dim = self.base.embed_dim
        self.num_parts = cfg.MODEL.NUM_PARTS
        self.memory_post_norm = nn.LayerNorm(embed_dim)
        self.part_decoder = PartDecoder1(embed_dim=embed_dim, num_heads=8, attn_dropout=0.0, feedforward_dim=4*embed_dim, ffn_dropout=0.0, num_parts=self.num_parts, return_intermediate=True)
        self.query_embedding = nn.Embedding(self.num_parts+1, embed_dim)
        # self.mask_bn = nn.BatchNorm1d(embed_dim)
        trunc_normal_(self.query_embedding.weight, std=embed_dim**(-0.5))
        ######## decoder head #######
        self.bottleneck_part = nn.BatchNorm1d(embed_dim*self.num_parts)
        self.bottleneck_part.bias.requires_grad_(False)
        self.bottleneck_part.apply(weights_init_kaiming)
        self.classifier_part = nn.Linear(embed_dim*self.num_parts, self.num_classes, bias=False)
        self.classifier_part.apply(weights_init_classifier)
        ######## visible predictor #####
        self.visb_pred = VisibilityPredictor(embed_dim=embed_dim, out_channel=1)
        ######## part pixel classifier ############
        self.pixel_classifier = PixelToPartClassifier(embed_dim, self.num_parts+1)
        self.part_pooling = GlobalWeightedPooling()
        ######## fore head #######
        self.with_fore_head = cfg.MODEL.WITH_FORE_HEAD
        if self.with_fore_head:
            self.fg_pooling = GlobalWeightedPooling()
            self.bottleneck_fg = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_fg.bias.requires_grad_(False)
            self.bottleneck_fg.apply(weights_init_kaiming)
            self.classifier_fg = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_fg.apply(weights_init_classifier)
        ######## part head #######
        self.bottleneck_partm = nn.BatchNorm1d(embed_dim*self.num_parts)
        self.bottleneck_partm.bias.requires_grad_(False)
        self.bottleneck_partm.apply(weights_init_kaiming)
        self.classifier_partm = nn.Linear(embed_dim*self.num_parts, self.num_classes, bias=False)
        self.classifier_partm.apply(weights_init_classifier)

    def mask_generator(self, mask):
        mask_targets = nn.functional.interpolate(mask,
                                                [self.num_y, self.num_x],
                                                mode='bilinear',
                                                align_corners=True)
        mask_targets = mask_targets.argmax(dim=1).flatten(-2, -1)  # (B, N)
        mask_targets_onehot = F.one_hot(mask_targets, self.num_parts+1)  # (B, N, P+1)
        visb_target = mask_targets_onehot.amax(dim=1)  # (B, P+1)
        mask_targets_onehot = mask_targets_onehot.permute(0, 2, 1)
        return mask_targets_onehot, visb_target, mask_targets

    def forward(self, x, label=None, mask=None, cam_label= None, view_label=None, return_attn=False):
        share_memory = self.base(x, cam_label=cam_label, view_label=view_label)  # B, N, C)

        ######### global branch ###########
        b1_feat = self.b1(share_memory)
        global_feat = b1_feat[:, 0]
        global_feat_bn = self.bottleneck(global_feat)

        ######## parsing pred #############
        bs, _, _ = share_memory.shape
        spatial_features = b1_feat[:, 1:]
        pixels_cls_logits = self.pixel_classifier(spatial_features)  # (B, N, P+1)
        pixels_parts_probabilities = F.softmax(pixels_cls_logits, dim=-1)
        # bg_masks = pixels_parts_probabilities[:, :, 0]
        parts_masks = pixels_parts_probabilities[:, :, 1:]
 
        ######### fg branch ###########
        if self.with_fore_head:
            fg_masks = parts_masks.sum(dim=-1, keepdim=True)
            # fg_masks = parts_masks.max(dim=-1)[0]
            fg_feature = self.fg_pooling(spatial_features, fg_masks)  # B, 1, C
            fg_feature_bn = self.bottleneck_fg(fg_feature.squeeze(1))  # B, C
        ######### parts branch ###########
        part_feature_m = self.part_pooling(spatial_features, parts_masks)  # (B, P, C)
        part_feature_m_cat = part_feature_m.flatten(1, 2)
        part_feature_m_cat_bn = self.bottleneck_partm(part_feature_m_cat)
        ######## local branch #############
        bs, _, _ = share_memory.shape
        memory = self.memory_post_norm(share_memory)  #
        query_embed = self.query_embedding.weight
        query_embed_repeat = query_embed.unsqueeze(0).repeat(bs, 1, 1)
        # 
        decoder_input = memory[:, 1:]
        cls_token = memory[:, :1].detach()
        query_input = torch.cat([cls_token, query_embed_repeat], dim=1)  # TODO: detach cls token
        part_output = self.part_decoder(query=query_input, key=decoder_input, value=decoder_input)  # List[dict,]  part_output[-1]['query']: (B, P, C)
        part_feature = part_output['query'][:, 1:]
        part_feature_cat = part_feature.flatten(1,2)
        part_feature_cat_bn = self.bottleneck_part(part_feature_cat)

        # pred visible score
        visb_score = self.visb_pred(part_feature)[:, :, 0]  # (B, P, 1)

        if self.training:
            mask, part_visb_target, mask_targets = self.mask_generator(mask)
            attn_masks = torch.rand_like(pixels_parts_probabilities)
            attn_masks = (attn_masks < pixels_parts_probabilities).to(pixels_parts_probabilities.dtype).permute(0, 2, 1)
            part_output_aux = self.part_decoder(query=query_input, key=decoder_input, value=decoder_input, attn_masks=attn_masks)
            # part_output_aux = self.part_decoder(query=query_input, key=decoder_input, value=decoder_input, attn_masks=mask)
            part_feature_aux = part_output_aux['query'][:, 1:]
            part_feature_aux_cat = part_feature_aux.flatten(1,2)
            part_feature_aux_cat_bn = self.bottleneck_part(part_feature_aux_cat)

            ######### parts branch with detach mask ###########
            part_feature_md = self.part_pooling(spatial_features, parts_masks.detach())  # (B, P, C)

            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score_global = self.classifier(global_feat_bn, label)
                cls_score_part = self.classifier_part(part_feature_cat_bn, label)
                cls_score_part_aux = self.classifier_part(part_feature_aux_cat_bn, label)
                cls_score_part_m = self.classifier_partm(part_feature_m_cat_bn, label)
                if self.with_fore_head:
                    cls_score_fg = self.classifier_fg(fg_feature_bn, label)
            else:
                cls_score_global = self.classifier(global_feat_bn)
                cls_score_part = self.classifier_part(part_feature_cat_bn)
                cls_score_part_aux = self.classifier_part(part_feature_aux_cat_bn)
                cls_score_part_m = self.classifier_partm(part_feature_m_cat_bn)
                if self.with_fore_head:
                    cls_score_fg = self.classifier_fg(fg_feature_bn)

            cls_score = {'global': cls_score_global, 
                         'cat': cls_score_part,
                         'cat_m': cls_score_part_m,
                         'aux': cls_score_part_aux,
                         }
            if self.with_fore_head:
                cls_score.update({'fg': cls_score_fg})
            feat_dict = {
                        'global': global_feat,
                         'part': part_feature,
                         'part_m': part_feature_m,
                         'part_md': part_feature_md,
                         'aux': part_feature_aux,
                         }
            return cls_score, feat_dict, visb_score, part_visb_target[:, 1:].to(global_feat_bn.dtype), pixels_cls_logits, mask_targets
        else:
            visb_score = torch.sigmoid(visb_score)  # (B, P)
            fore_part_visb = (visb_score > 0.5).to(part_feature_cat.dtype)
            part_feature_bn = part_feature_cat_bn.view(bs, self.num_parts, -1)
            # for cat config
            global_visb = fore_part_visb.new_ones(bs, 1)
            global_feat_bn = global_feat_bn.view(bs, 1, -1)
            final_feat_bn = torch.cat([global_feat_bn, part_feature_bn], dim=1)
            visb = torch.cat([global_visb, fore_part_visb], dim=1)

            if return_attn:
                return final_feat_bn, visb, part_output['cross_attn']

            return final_feat_bn, visb

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if ('classifier' in i) and ('vis' not in i):
                continue
            else:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))