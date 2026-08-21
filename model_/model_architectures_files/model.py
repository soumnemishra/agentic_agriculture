import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import config
from model_blocks import CondConv2D, SSEBlock, InceptionBlock, PatchTokenizer

class CondConViT_V2(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, dropout_rate=0.4):
        super(CondConViT_V2, self).__init__()

        # --- 1. THE BACKBONE (MobileNetV2) ---
        base_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        return_nodes = {
            'features.1': 'x1',    # Spatial: 112x112, Channels: 16
            'features.3': 'x2',    # Spatial: 56x56, Channels: 24
            'features.6': 'x_inc'  # Spatial: 28x28, Channels: 32
        }
        self.backbone = create_feature_extractor(base_model, return_nodes=return_nodes)

        for param in self.backbone.parameters():
            param.requires_grad = False

        # --- 2. MULTI-SCALE FEATURE EXTRACTION ---
        self.cond1_1 = CondConv2D(in_channels=16, out_channels=16, kernel_size=3, stride=2)
        self.cond1_2 = CondConv2D(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.cond1_3 = CondConv2D(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.conv1_final = nn.Conv2d(32, 29, kernel_size=4, stride=1, padding=0)
        self.sse1 = SSEBlock(in_channels=29, ratio=4)
        self.p1_proj = nn.Conv2d(32, 29, kernel_size=1)

        self.cond2_1 = CondConv2D(in_channels=24, out_channels=16, kernel_size=3, stride=2)
        self.cond2_2 = CondConv2D(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.conv2_final = nn.Conv2d(32, 29, kernel_size=4, stride=1, padding=0)
        self.sse2 = SSEBlock(in_channels=29, ratio=4)
        self.p2_proj = nn.Conv2d(32, 29, kernel_size=1)

        self.inception = InceptionBlock(in_channels=32, nb_filter=32)
        self.cond3_1 = CondConv2D(in_channels=128, out_channels=32, kernel_size=3, stride=2)
        self.conv3_final = nn.Conv2d(32, 29, kernel_size=4, stride=1, padding=0)
        self.sse3 = SSEBlock(in_channels=29, ratio=4)
        self.p3_proj = nn.Conv2d(32, 29, kernel_size=1)

        # --- 3. VISION TRANSFORMER PATH (HIGH DROPOUT) ---
        self.tokenizer = PatchTokenizer(in_channels=128, patch_size=7, embed_dim=32, img_size=28)
        self.vit_dropout = nn.Dropout(dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=2,
            dim_feedforward=64,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.final_norm = nn.LayerNorm(32)

        self.vit_projection = nn.Conv2d(32, 29, kernel_size=1)

        # --- 4. FUSION & CLASSIFICATION ---
        self.fusion_sse = SSEBlock(in_channels=29, ratio=4)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(32, num_classes) 

    def forward(self, x):
        features = self.backbone(x)
        x1, x2, x_inc = features['x1'], features['x2'], features['x_inc']

        p1 = self.p1_proj(self.sse1(self.conv1_final(self.cond1_3(self.cond1_2(self.cond1_1(x1))))))
        p2 = self.p2_proj(self.sse2(self.conv2_final(self.cond2_2(self.cond2_1(x2)))))

        inc_out = self.inception(x_inc)
        p3 = self.p3_proj(self.sse3(self.conv3_final(self.cond3_1(inc_out))))

        tokens = self.tokenizer(inc_out)
        tokens = self.vit_dropout(tokens)
        vit_out = self.transformer(tokens)
        vit_out = self.final_norm(vit_out)

        b = vit_out.size(0)
        vit_spatial = vit_out.transpose(1, 2).view(b, 32, 4, 4)

        vit_spatial = F.interpolate(vit_spatial, size=(11, 11), mode='bilinear', align_corners=False)
        vit_spatial = self.vit_projection(vit_spatial)

        merged = p1 + p2 + p3 + vit_spatial
        merged = self.fusion_sse(merged)

        out = self.global_pool(merged)
        out = torch.flatten(out, 1)

        return self.classifier(out)

def get_model():
    model = CondConViT_V2().to(config.DEVICE)
    return model
