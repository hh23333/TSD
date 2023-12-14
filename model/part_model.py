import torch
import torch.nn as nn

class PixelToPartClassifier(nn.Module):
    def __init__(self, in_channel, out_channel, with_bn=True):
        super(PixelToPartClassifier, self).__init__()
        self.with_bn = with_bn
        if self.with_bn:
            self.bn = nn.BatchNorm1d(in_channel)
        self.classifier = nn.Linear(in_channel, out_channel, bias=False)
        self._init_params()

    def forward(self, x):
        if self.with_bn:
            x = x.permute(0,2,1)
            x = self.bn(x).permute(0,2,1)
        
        return self.classifier(x)  # (B, N, P)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)  # ResNet = 0.01, Bof and ISP-reid = 0.001
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class GlobalWeightedPooling(nn.Module):
    def __init__(self):
        super().__init__()
        # self.global_pooling = nn.AdaptiveAvgPool1d((1, 1))
    def forward(self, features, part_masks):
        '''
        features: B,N,C
        part_masks: B,N,P
        '''
        part_masks = torch.unsqueeze(part_masks, 2) # B,N,1,P
        features = torch.unsqueeze(features, 3)  # B,N,C,1
        parts_features = torch.mul(part_masks, features)  # B,N,C,P
        parts_features = torch.sum(parts_features, dim=1)  # B, C, P
        part_masks_sum = torch.sum(part_masks, dim=1)  # B, 1, P
        part_masks_sum = torch.clamp(part_masks_sum, min=1e-6)
        parts_features_avg = torch.div(parts_features, part_masks_sum)
        parts_features = parts_features_avg.permute(0,2,1)  # B, P, C
        return parts_features