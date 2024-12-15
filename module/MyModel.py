import torch.nn as nn
import torch.nn.functional as F
import torch
import ever as er
from module.resnet import ResNetEncoder


class PPMBilinear(nn.Module):
    def __init__(
        self,
        num_classes=7,
        fc_dim=2048,
        use_aux=False,
        pool_scales=(1, 2, 3, 6),
        norm_layer=nn.BatchNorm2d
    ):
        super(PPMBilinear, self).__init__()

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True)
                )
            )
        self.ppm = nn.ModuleList(self.ppm)

        self.use_aux = use_aux
        if self.use_aux:
            self.cbr_deepsup = nn.Sequential(
                nn.Conv2d(fc_dim // 2, fc_dim // 4, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(fc_dim // 4),
                nn.ReLU(inplace=True),
            )
            self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_classes, 1, 1, 0)
            self.dropout_deepsup = nn.Dropout2d(0.1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim + len(pool_scales) * 512, 512, kernel_size=3, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, conv_out):
        input_size = conv_out.size()
        ppm_out = [conv_out]
        for pool_scale in self.ppm:
            ppm_out.append(F.interpolate(
                pool_scale(conv_out),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)
            )

        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_aux and self.training:
            conv4 = conv_out[-2]
            _ = self.cbr_deepsup(conv4)
            _ = self.dropout_deepsup(_)
            _ = self.conv_last_deepsup(_)
            return x
        else:
            return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, height, width = x.size()

        avg_pool = torch.mean(x, dim=(2, 3))  # [B, C]
        max_pool, _ = torch.max(x.view(batch, channels, -1), dim=2)  # [B, C]
        pool_sum = avg_pool + max_pool  # [B, C]

        attention = self.fc1(pool_sum)  # [B, C//reduction]
        attention = self.relu(attention)  # [B, C//reduction]
        attention = self.fc2(attention)  # [B, C]
        attention = self.sigmoid(attention).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]

        return x * attention

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, height, width = x.size()

        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        pool = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, H, W]

        attention = self.conv(pool)  # [B, 1, H, W]
        attention = self.sigmoid(attention)  # [B, 1, H, W]

        return x * attention

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, channels, height, width = x.size()

        query = self.query_conv(x).view(batch, -1, height * width).permute(0, 2, 1)  # [B, N, C]
        key = self.key_conv(x).view(batch, -1, height * width)  # [B, N, C]
        attention = self.softmax(torch.bmm(query, key))  # [B, N, N]
        value = self.value_conv(x).view(batch, -1, height * width)  # [B, C, N]

        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, N]
        out = out.view(batch, channels, height, width)

        return out + x  # Residual connection


class Deeplabv2(er.ERModule):
    def __init__(self, config):
        super(Deeplabv2, self).__init__(config)

        self.encoder = ResNetEncoder(self.config.backbone)

        if self.config.multi_layer:
            if self.config.cascade:
                if self.config.use_ppm:
                    self.layer5 = PPMBilinear(**self.config.ppm1)
                    self.layer6 = PPMBilinear(**self.config.ppm2)
                else:
                    self.layer5 = self._make_pred_layer(Classifier_Module, self.config.inchannels // 2, [6, 12, 18, 24],[6, 12, 18, 24], self.config.num_classes)
                    self.layer6 = self._make_pred_layer(Classifier_Module, self.config.inchannels, [6, 12, 18, 24],[6, 12, 18, 24], self.config.num_classes)
            else:
                if self.config.use_ppm:
                    self.layer5 = PPMBilinear(**self.config.ppm)
                    self.layer6 = PPMBilinear(**self.config.ppm)
                else:
                    self.layer5 = self._make_pred_layer(Classifier_Module, self.config.inchannels, [6, 12, 18, 24],[6, 12, 18, 24], self.config.num_classes)
                    self.layer6 = self._make_pred_layer(Classifier_Module, self.config.inchannels, [6, 12, 18, 24],[6, 12, 18, 24], self.config.num_classes)

            self.channel_attention = ChannelAttention(self.config.inchannels)
            self.spatial_attention = SpatialAttention()
            self.self_attention = SelfAttention(self.config.inchannels)
            self.fusion_conv = nn.Conv2d(self.config.inchannels * 3, self.config.inchannels, kernel_size=1, bias=False)
        else:
            if self.config.use_ppm:
                self.cls_pred = PPMBilinear(**self.config.ppm)
            else:
                self.cls_pred = self._make_pred_layer(Classifier_Module, self.config.inchannels, [6, 12, 18, 24],[6, 12, 18, 24], self.config.num_classes)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.config.multi_layer:
            if self.config.cascade:
                feat1, feat2 = self.encoder(x)[-2:]
                x1 = self.layer5(feat1)
                x2 = self.layer6(feat2)

                Fc1 = self.channel_attention(feat1)
                Fs1 = self.spatial_attention(feat1)
                Fa1 = self.self_attention(feat1)

                feat1 = torch.cat([Fc1, Fs1, Fa1], dim=1)
                feat1 = self.fusion_conv(feat1)

                Fc2 = self.channel_attention(feat2)
                Fs2 = self.spatial_attention(feat2)
                Fa2 = self.self_attention(feat2)

                feat2 = torch.cat([Fc2, Fs2, Fa2], dim=1)
                feat2 = self.fusion_conv(feat2)

                if self.training:
                    return x1, feat1, x2, feat2
                else:
                    x1 = F.interpolate(x1, (H, W), mode='bilinear', align_corners=True)
                    x2 = F.interpolate(x2, (H, W), mode='bilinear', align_corners=True)
                    return (x1.softmax(dim=1) + x2.softmax(dim=1)) / 2
            else:
                feat = self.encoder(x)[-1]
                x1 = self.layer5(feat)
                x2 = self.layer6(feat)

                Fc = self.channel_attention(feat)
                Fs = self.spatial_attention(feat)
                Fa = self.self_attention(feat)

                feat = torch.cat([Fc, Fs, Fa], dim=1)
                feat = self.fusion_conv(feat)

                if self.training:
                    return x1, x2, feat
                else:
                    x1 = F.interpolate(x1, (H, W), mode='bilinear', align_corners=True)
                    x2 = F.interpolate(x2, (H, W), mode='bilinear', align_corners=True)
                    return (x1.softmax(dim=1) + x2.softmax(dim=1)) / 2

        else:
            feat = self.encoder(x)[-1]
            x = self.cls_pred(feat)
            if self.training:
                return x, feat
            else:
                x = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
                return x.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            backbone=dict(
                resnet_type='resnet50',
                output_stride=16,
                pretrained=True,
            ),
            multi_layer=False,
            cascade=False,
            use_ppm=False,
            ppm=dict(
                num_classes=7,
                use_aux=False,
                norm_layer=nn.BatchNorm2d,
            ),
            inchannels=2048,
            num_classes=7
        ))


class Classifier_Module(nn.Module):
    def __init__(
        self,
        inplanes,
        dilation_series,
        padding_series,
        num_classes
    ):
        super(Classifier_Module, self).__init__()

        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True)
            )
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out