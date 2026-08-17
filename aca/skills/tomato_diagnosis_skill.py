"""
ACA Tomato Disease Diagnosis Skill
===================================

Encapsulates deep learning vision perception for tomato leaf disease diagnosis
using the CondConViT_V2 model architecture loaded from best_model_v5.pth.

Strict VRAM Management:
- Model loaded to GPU (cuda:0) if available, otherwise CPU.
- Uses `torch.no_grad()` and batch_size=1 during execution.
- Clears CUDA cache immediately following inference to preserve VRAM for LLM reasoning.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from aca.logging_config import get_logger
from aca.skills.base_skill import BaseSkill, SkillParameter, SkillResult, SkillSchema
from aca.tools.registry import ToolRegistry

logger = get_logger("skills.tomato_diagnosis")

TOMATO_CLASSES: List[str] = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites Two-spotted_spider_mite",
    "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "healthy",
    "powdery_mildew",
]


# ── PyTorch CondConViT_V2 Architecture ────────────────────────────────────────

class Routing(nn.Module):
    """Routing layer with temperature-scaled Softmax for expert kernels."""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.2, temperature: float = 30.0):
        super().__init__()
        self.temperature = temperature
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_channels, out_channels)
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x) / self.temperature
        return F.softmax(x, dim=1)


class CondConv2D(nn.Module):
    """Conditional Convolution with multiple expert kernels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        num_experts: int = 3,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.routing = Routing(in_channels, out_channels=num_experts)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            for _ in range(num_experts)
        ])
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            if bias:
                nn.init.constant_(conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routing_weights = self.routing(x)
        output = 0
        for i in range(self.num_experts):
            weight_i = routing_weights[:, i].view(-1, 1, 1, 1)
            output += weight_i * self.convs[i](x)
        return output


class SeparableConv2d(nn.Module):
    """Depthwise-separable convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Any, stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        if isinstance(kernel_size, int):
            pad = kernel_size // 2
        else:
            pad = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=pad, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class InceptionBlock(nn.Module):
    """Multi-scale cross separable Inception block."""

    def __init__(self, in_channels: int, nb_filter: int):
        super().__init__()
        self.branch1x1 = SeparableConv2d(in_channels, nb_filter, kernel_size=1)
        self.branch3x3_base = SeparableConv2d(in_channels, nb_filter, kernel_size=1)
        self.branch3x3_1 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(3, 1))
        self.branch3x3_2 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(1, 3))

        self.branch5x5_base = SeparableConv2d(in_channels, nb_filter, kernel_size=1)
        self.branch5x5_1 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(3, 1))
        self.branch5x5_2 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(1, 3))
        self.branch5x5_final1 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(3, 1))
        self.branch5x5_final2 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(1, 3))

        self.branchpool_max = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branchpool_conv = SeparableConv2d(in_channels, nb_filter, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1x1(x)
        b3_base = self.branch3x3_base(x)
        b3 = self.branch3x3_1(b3_base) + self.branch3x3_2(b3_base)
        b5_base = self.branch5x5_base(x)
        b5_mid = self.branch5x5_1(b5_base) + self.branch5x5_2(b5_base)
        b5 = self.branch5x5_final1(b5_mid) + self.branch5x5_final2(b5_mid)
        bp = self.branchpool_conv(self.branchpool_max(x))
        return torch.cat([b1, b3, b5, bp], dim=1)


class PatchTokenizer(nn.Module):
    """2D Patch Tokenizer with positional embeddings."""

    def __init__(self, in_channels: int, patch_size: int, embed_dim: int, img_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.position_embedding
        return x


class SSEBlock(nn.Module):
    """Squeeze-and-Excitation block with Spatial Statistics."""

    def __init__(self, in_channels: int, ratio: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        se = self.avg_pool(x).view(b, c)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se).view(b, c, 1, 1)
        se_out = x * se
        spatial_mean = torch.mean(x, dim=1, keepdim=True)
        spatial_std = torch.std(x, dim=1, keepdim=True)
        spatial_max = torch.max(x, dim=1, keepdim=True)[0]
        return torch.cat([se_out, spatial_mean, spatial_std, spatial_max], dim=1)


class CondConViT_V2(nn.Module):
    """CondConViT_V2 spatial vision architecture."""

    def __init__(self, num_classes: int = 11, dropout_rate: float = 0.4):
        super().__init__()
        base_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        return_nodes = {
            "features.1": "x1",
            "features.3": "x2",
            "features.6": "x_inc",
        }
        self.backbone = create_feature_extractor(base_model, return_nodes=return_nodes)
        for param in self.backbone.parameters():
            param.requires_grad = False

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

        self.tokenizer = PatchTokenizer(in_channels=128, patch_size=7, embed_dim=32, img_size=28)
        self.vit_dropout = nn.Dropout(dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=2,
            dim_feedforward=64,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.final_norm = nn.LayerNorm(32)
        self.vit_projection = nn.Conv2d(32, 29, kernel_size=1)

        self.fusion_sse = SSEBlock(in_channels=29, ratio=4)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        x1, x2, x_inc = features["x1"], features["x2"], features["x_inc"]

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

        vit_spatial = F.interpolate(vit_spatial, size=(11, 11), mode="bilinear", align_corners=False)
        vit_spatial = self.vit_projection(vit_spatial)

        merged = p1 + p2 + p3 + vit_spatial
        merged = self.fusion_sse(merged)

        out = self.global_pool(merged)
        out = torch.flatten(out, 1)

        return self.classifier(out)


# ── BaseSkill Implementation ──────────────────────────────────────────────────

class TomatoDiagnosisSkill(BaseSkill):
    """
    Skill for tomato plant leaf disease diagnosis using CondConViT_V2.

    Loads weights from `best_model_v5.pth` and executes inference with strict
    memory boundaries (`torch.no_grad()`, `batch_size=1`, CUDA cache clearing).
    """

    def __init__(
        self,
        model_path: str = r"D:\agentic_agriculture\model_\best_model_v5.pth",
        device: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.device = torch.device(
            device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.classes = TOMATO_CLASSES
        self._model: Optional[CondConViT_V2] = None

        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._load_model()

    def _load_model(self) -> None:
        """Initializes architecture and loads pre-trained weights."""
        try:
            logger.info("Initializing CondConViT_V2 model on device: %s", self.device)
            model = CondConViT_V2(num_classes=len(self.classes))

            if os.path.exists(self.model_path):
                state_dict = torch.load(self.model_path, map_location=self.device)
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                model.load_state_dict(state_dict)
                logger.info("Successfully loaded weights from %s", self.model_path)
            else:
                logger.warning(
                    "Model weights not found at %s. Running in uninitialized mode.",
                    self.model_path,
                )

            model.to(self.device)
            model.eval()
            self._model = model
        except Exception as e:
            logger.exception("Failed to load TomatoDiagnosisSkill vision model: %s", e)
            raise

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            name="tomato_diagnosis_skill",
            description="Diagnoses tomato leaf disease and health state from leaf image frame using CondConViT_V2 model.",
            parameters=[
                SkillParameter(
                    name="image_path",
                    description="Path to leaf image file or synthetic image marker.",
                    param_type="str",
                    required=False,
                    default="",
                )
            ],
            tools_required=[],
            estimated_duration_seconds=0.1,
        )

    def execute(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        image_path: str = "",
        **kwargs: Any,
    ) -> SkillResult:
        """
        Executes vision inference on an input image tensor/path.

        Enforces VRAM budget:
            1. `torch.no_grad()` evaluation mode.
            2. Single image batch (`batch_size=1`).
            3. `torch.cuda.empty_cache()` post-execution cleanup.
        """
        start_time = time.perf_counter()

        try:
            # Load and preprocess image
            if image_path and os.path.exists(image_path):
                raw_img = Image.open(image_path).convert("RGB")
                input_tensor = self._transform(raw_img).unsqueeze(0).to(self.device)
            else:
                logger.debug("No valid image file provided; generating synthetic 224x224 RGB input frame.")
                input_tensor = torch.randn(1, 3, 224, 224, device=self.device)

            # Strict no-grad inference
            with torch.no_grad():
                logits = self._model(input_tensor)
                probs = F.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)

                predicted_class = self.classes[pred_idx.item()]
                confidence = float(conf.item())

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            # VRAM Garbage collection & CUDA Cache flush
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            result_data = {
                "predicted_class": predicted_class,
                "confidence": round(confidence, 4),
                "inference_time_ms": round(elapsed_ms, 2),
            }

            logger.info(
                "TomatoDiagnosisSkill output: %s (confidence=%.2f, time=%.2f ms)",
                predicted_class,
                confidence,
                elapsed_ms,
            )

            return SkillResult(
                success=True,
                data=result_data,
                metadata={"device": str(self.device), "num_classes": len(self.classes)},
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.exception("TomatoDiagnosisSkill execution failed: %s", e)

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            return SkillResult(
                success=False,
                error=str(e),
                metadata={"inference_time_ms": round(elapsed_ms, 2)},
            )
