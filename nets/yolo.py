#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
from nets.losses import PromptConsistencyLoss
from nets.restoration import Encoder, Decoder, UpSample, CAB

class HardPromptEncoder:
    """CLIP硬提示编码器"""
    def __init__(self, model_type="ViT-B/32", device=None):
        import clip
        self.clip = clip
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"正在加载 CLIP ({model_type})...")
        self.model, _ = clip.load(model_type, device=self.device)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, texts, normalize=True):
        if isinstance(texts, str):
            texts = [texts]

        current_device = next(self.model.parameters()).device
        text_tokens = self.clip.tokenize(texts, truncate=True).to(current_device)

        with torch.no_grad():
            self.model.to(current_device)
            text_features = self.model.encode_text(text_tokens)
            if normalize:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.float().to(current_device)


class HardPromptRefiner(nn.Module):
    """硬提示精炼器"""
    def __init__(self, clip_dim=512, type_dim=3, hidden_dim=512):
        super().__init__()
        self.type_net = nn.Sequential(
            nn.Linear(type_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, clip_dim)
        )

    def forward(self, text, category):
        text = F.normalize(text, dim=-1)
        return F.normalize(text + self.type_net(category), dim=-1)


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, in_channels=[256, 512, 1024], act="silu", depthwise=False, ):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels=int(in_channels[i] * width), out_channels=int(256 * width), ksize=1, stride=1, act=act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=num_classes, kernel_size=1, stride=1, padding=0)
            )

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, inputs):
        #   inputs
        outputs = []
        for k, x in zip(range(len(inputs)), inputs):
            x = self.stems[k](x)
            cls_feat = self.cls_convs[k](x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = self.reg_convs[k](x)
            reg_output = self.reg_preds[k](reg_feat)

            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs


class MBFFB(nn.Module):
    def __init__(self, n_feats, w_d, w_r):
        super().__init__()
        # 20x20x1024
        self.cab1 = CAB(n_feats * 16, reduction=8)
        self.up1 = UpSample(n_feats * 16, s_factor=2)
        # 40x40x512
        self.cab2 = CAB(n_feats * 8, reduction=8)
        self.up2 = UpSample(n_feats * 8, s_factor=2)
        # 80x80x256
        self.cab3 = CAB(n_feats * 4, reduction=8)

        self.w_d = nn.Parameter(torch.FloatTensor(w_d), requires_grad=True)
        self.w_r = nn.Parameter(torch.FloatTensor(w_r), requires_grad=True)

    def forward(self, x_d, x_r):
        x_d, x_r = x_d[::-1], x_r[::-1]
        x_20 = self.cab1(self.w_d[0] * x_d[0] + self.w_r[0] * x_r[0])
        x_40 = self.up1(x_20)
        x_40 = self.cab2(x_40 + self.w_d[1] * x_d[1] + self.w_r[1] * x_r[1])
        x_80 = self.up2(x_40)
        x_80 = self.cab3(x_80 + self.w_d[2] * x_d[2] + self.w_r[2] * x_r[2])
        return x_80, x_40, x_20


class YOLOPAFPN(nn.Module):
    def __init__(self, depth=1.0, width=1.0, base_channels=32, in_features=("dark3", "dark4", "dark5"), in_channels=[256, 512, 1024], depthwise=False, act="silu"):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)

        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.bu_conv2 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.bu_conv1 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.fusion = MBFFB(n_feats=base_channels, w_d=[1.0, 1.0, 1.0], w_r=[1.0, 1.0, 1.0])

    def forward(self, input_, res_feats, inter):
        out_features = self.backbone(input_, inter)
        det_features = [out_features[f] for f in self.in_features]
        fusion_features = self.fusion(det_features, res_feats)

        feat1, feat2, feat3 = fusion_features  #
        P5 = self.lateral_conv0(feat3)
        P5_upsample = self.upsample(P5)
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        P5_upsample = self.C3_p4(P5_upsample)

        P4 = self.reduce_conv1(P5_upsample)
        P4_upsample = self.upsample(P4)
        P4_upsample = torch.cat([P4_upsample, feat1], 1)
        P3_out = self.C3_p3(P4_upsample)

        P3_downsample = self.bu_conv2(P3_out)
        P3_downsample = torch.cat([P3_downsample, P4], 1)
        P4_out = self.C3_n3(P3_downsample)

        P4_downsample = self.bu_conv1(P4_out)
        P4_downsample = torch.cat([P4_downsample, P5], 1)
        P5_out = self.C3_n4(P4_downsample)
        return [P3_out, P4_out, P5_out], feat1


class YoloBody(nn.Module):
    def __init__(self, num_classes, phi='s', use_prompt_fusion=True, input_size=640):
        super().__init__()
        if phi != 's':
            raise ValueError('This implementation uses the YOLOX-s backbone.')
        self.use_prompt_fusion = use_prompt_fusion
        self.clip_encoder = None
        self.encoder = Encoder(in_channels=3, n_feats=32, reduction=8)
        self.backbone = YOLOPAFPN(.33, .5, base_channels=32)
        self.head = YOLOXHead(num_classes, .5)
        self.decoder = Decoder(out_channel=3, n_feats=32, reduction=8, use_prompt_fusion=use_prompt_fusion, input_size=input_size)
        if use_prompt_fusion:
            self.hard_prompt_refiner = HardPromptRefiner()
            self.prompt_consistency = PromptConsistencyLoss([64, 32])

    def _init_clip_encoder(self):
        if self.use_prompt_fusion and self.clip_encoder is None:
            self.clip_encoder = HardPromptEncoder(device=next(self.parameters()).device)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['clip_encoder'] = None
        return state

    def forward(self, x, weather_texts=None, category_vectors=None, text_features=None):
        hard_prompt = None
        if self.use_prompt_fusion:
            if text_features is not None:
                text = text_features.detach().to(device=x.device, dtype=x.dtype)
                if text.shape != (len(x), 512) or not torch.isfinite(text).all():
                    raise ValueError('text_features must be a finite [batch, 512] tensor.')
            else:
                if weather_texts is None or len(weather_texts) != len(x) or any(not t.strip() for t in weather_texts):
                    raise ValueError('Provide one degradation description per image.')
                self._init_clip_encoder()
                text = self.clip_encoder.encode(weather_texts).to(device=x.device, dtype=x.dtype)
            if category_vectors is None:
                raise ValueError('Provide one-hot category_vectors in [fog, rain, dark] order.')
            category = torch.as_tensor(category_vectors, dtype=x.dtype, device=x.device)
            if category.shape != (len(x), 3) or not torch.isfinite(category).all():
                raise ValueError('category_vectors must have shape [batch, 3].')
            if not (((category == 0) | (category == 1)).all() and (category.sum(-1) == 1).all()):
                raise ValueError('category_vectors must be one-hot.')
            hard_prompt = self.hard_prompt_refiner(text, category)
        features, soft_prompts = self.decoder.prepare_features(self.encoder(x), hard_prompt)
        # The same guided skip features condition detection and restoration.
        pyramid, decoder_input = self.backbone(x, features, features)
        detected = self.head(pyramid)
        restored = self.decoder(decoder_input, features, x)
        if self.training:
            ppc = self.prompt_consistency(soft_prompts) if soft_prompts else x.new_zeros(())
            return detected, restored, ppc
        return detected, restored
