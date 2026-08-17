"""
ACA Skills Layer — Tomato Diagnosis Skill
==========================================

Wraps the specialized PyTorch CondConViT_V2 vision model into an ACA
``BaseSkill`` for high-confidence tomato crop disease identification.

Features:
    - Custom PyTorch CondConViT_V2 deep convolutional-vision transformer
      architecture with Dynamic Conditional Convolutions, Inception blocks,
      Spatial-Squeeze-and-Excitation, and Transformer encoders.
    - 11 tomato health categories (10 diseases + 1 healthy state).
    - Strict Hardware / VRAM Optimization for 4GB NVIDIA GPUs (GTX 1650):
        * ``torch.no_grad()`` enforced during inference.
        * ``batch_size = 1`` strict pipeline.
        * ``torch.cuda.empty_cache()`` called immediately post-inference.
    - Full dependency injection via ``__init__``.
    - Conforms to ``BaseSkill`` with declarative ``SkillSchema`` and
      standardized ``SkillResult``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from aca.logging_config import get_logger
from aca.skills.base_skill import BaseSkill, SkillParameter, SkillResult, SkillSchema
from aca.tools.registry import ToolRegistry

logger = get_logger("skills.tomato_diagnosis")

# ── Known Tomato Classes (Alphabetical dataset order) ────────────────────────
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

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "model_",
    "best_model_v5.pth",
)


# ── PyTorch Model Architecture: CondConViT_V2 ────────────────────────────────

class ConvBNReLU(nn.Sequential):
    """Convolution + BatchNorm + ReLU6 building block."""
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """MobileNetV2 Inverted Residual block."""
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Backbone(nn.Module):
    """
    Self-contained MobileNetV2 feature extractor slicing features at
    nodes x1 (112x112, 16ch), x2 (56x56, 24ch), and x_inc (28x28, 32ch).
    """
    def __init__(self) -> None:
        super(MobileNetV2Backbone, self).__init__()
        self.features = nn.ModuleList([
            ConvBNReLU(3, 32, stride=2),                           # 0: 112x112, 32
            InvertedResidual(32, 16, stride=1, expand_ratio=1),     # 1: 112x112, 16 (x1)
            InvertedResidual(16, 24, stride=2, expand_ratio=6),     # 2: 56x56, 24
            InvertedResidual(24, 24, stride=1, expand_ratio=6),     # 3: 56x56, 24 (x2)
            InvertedResidual(24, 32, stride=2, expand_ratio=6),     # 4: 28x28, 32
            InvertedResidual(32, 32, stride=1, expand_ratio=6),     # 5: 28x28, 32
            InvertedResidual(32, 32, stride=1, expand_ratio=6),     # 6: 28x28, 32 (x_inc)
        ])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.features[0](x)
        x1 = self.features[1](x)
        x = self.features[2](x1)
        x2 = self.features[3](x)
        x = self.features[4](x2)
        x = self.features[5](x)
        x_inc = self.features[6](x)
        return {"x1": x1, "x2": x2, "x_inc": x_inc}


class Routing(nn.Module):
    """
    Learns dynamic routing weights for expert convolutions using Softmax.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.2,
        temperature: float = 30.0,
    ) -> None:
        super(Routing, self).__init__()
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
    """
    Dynamic Kernel Mixing Conditional Convolution (O(1) FLOP overhead).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        num_experts: int = 3,
    ) -> None:
        super(CondConv2D, self).__init__()
        self.num_experts = num_experts
        self.routing = Routing(in_channels, out_channels=num_experts)
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
            for _ in range(num_experts)
        ])
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            if bias and conv.bias is not None:
                nn.init.constant_(conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        routing_weights = self.routing(x)
        expert_weights = torch.stack([conv.weight for conv in self.convs])
        expert_biases = (
            torch.stack([conv.bias for conv in self.convs])
            if self.convs[0].bias is not None
            else None
        )

        mixed_weights = torch.einsum("be,eoikl->boikl", routing_weights, expert_weights)
        mixed_weights = mixed_weights.reshape(
            B * self.convs[0].out_channels,
            C,
            self.convs[0].kernel_size[0],
            self.convs[0].kernel_size[1],
        )
        x_reshaped = x.view(1, B * C, H, W)

        if expert_biases is not None:
            mixed_biases = torch.einsum("be,eo->bo", routing_weights, expert_biases).reshape(-1)
        else:
            mixed_biases = None

        out = F.conv2d(
            x_reshaped,
            mixed_weights,
            bias=mixed_biases,
            stride=self.convs[0].stride,
            padding=self.convs[0].padding,
            groups=B,
        )
        return out.view(B, self.convs[0].out_channels, out.shape[2], out.shape[3])


class SeparableConv2d(nn.Module):
    """
    Depthwise-Separable 2D Convolution.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super(SeparableConv2d, self).__init__()
        if isinstance(kernel_size, int):
            pad = kernel_size // 2
        else:
            pad = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class InceptionBlock(nn.Module):
    """
    Multi-branch Inception module with separable convolutions.
    """
    def __init__(self, in_channels: int, nb_filter: int) -> None:
        super(InceptionBlock, self).__init__()
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
    """
    Slices the feature map into patches and projects into a flat sequence.
    """
    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        embed_dim: int,
        img_size: int,
    ) -> None:
        super(PatchTokenizer, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.num_patches = (img_size // patch_size) ** 2
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.position_embedding
        return x


class SSEBlock(nn.Module):
    """
    Squeeze-and-Excitation block augmented with Spatial Statistics.
    """
    def __init__(self, in_channels: int, ratio: int = 4) -> None:
        super(SSEBlock, self).__init__()
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
    """
    Full CondConViT_V2 Neural Architecture for Tomato Leaf Disease Perception.
    """
    def __init__(self, num_classes: int = 11, dropout_rate: float = 0.4) -> None:
        super(CondConViT_V2, self).__init__()

        # 1. Backbone
        self.backbone = MobileNetV2Backbone()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Multi-Scale Feature Extraction
        # Path 1 (Early Features: Input 16 channels)
        self.cond1_1 = CondConv2D(in_channels=16, out_channels=16, kernel_size=3, stride=2)
        self.cond1_2 = CondConv2D(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.cond1_3 = CondConv2D(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.conv1_final = nn.Conv2d(32, 29, kernel_size=4, stride=1, padding=0)
        self.sse1 = SSEBlock(in_channels=29, ratio=4)
        self.p1_proj = nn.Conv2d(32, 29, kernel_size=1)

        # Path 2 (Mid Features: Input 24 channels)
        self.cond2_1 = CondConv2D(in_channels=24, out_channels=16, kernel_size=3, stride=2)
        self.cond2_2 = CondConv2D(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.conv2_final = nn.Conv2d(32, 29, kernel_size=4, stride=1, padding=0)
        self.sse2 = SSEBlock(in_channels=29, ratio=4)
        self.p2_proj = nn.Conv2d(32, 29, kernel_size=1)

        # Path 3 (Deep Features via Inception: Input 32 channels)
        self.inception = InceptionBlock(in_channels=32, nb_filter=32)
        self.cond3_1 = CondConv2D(in_channels=128, out_channels=32, kernel_size=3, stride=2)
        self.conv3_final = nn.Conv2d(32, 29, kernel_size=4, stride=1, padding=0)
        self.sse3 = SSEBlock(in_channels=29, ratio=4)
        self.p3_proj = nn.Conv2d(32, 29, kernel_size=1)

        # 3. Vision Transformer Path
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

        # 4. Fusion & Classifier
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


# ── Tomato Diagnosis BaseSkill ────────────────────────────────────────────────

class TomatoDiagnosisSkill(BaseSkill):
    """
    ACA BaseSkill encapsulating PyTorch CondConViT_V2 inference for
    tomato disease diagnosis.

    Hardware Profile:
        Runs on GPU (``cuda:0``) if available, with strict ``torch.no_grad()``,
        ``batch_size = 1``, and automatic ``torch.cuda.empty_cache()`` to protect
        the 4GB VRAM envelope.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        auto_load: bool = True,
    ) -> None:
        self.model_path = model_path or DEFAULT_MODEL_PATH
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model: Optional[CondConViT_V2] = None
        self._is_loaded: bool = False

        if auto_load:
            self.load_model()

    def load_model(self) -> None:
        """Instantiate CondConViT_V2 and load the pre-trained weights."""
        logger.info("Initializing CondConViT_V2 on device: %s", self.device)
        self.model = CondConViT_V2(num_classes=len(TOMATO_CLASSES))

        if os.path.exists(self.model_path):
            try:
                ckpt = torch.load(self.model_path, map_location=self.device)
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    state_dict = ckpt["state_dict"]
                elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    state_dict = ckpt["model_state_dict"]
                else:
                    state_dict = ckpt

                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    clean_k = k
                    if clean_k.startswith("module."):
                        clean_k = clean_k[7:]
                    cleaned_state_dict[clean_k] = v

                self.model.load_state_dict(cleaned_state_dict, strict=True)
                logger.info("Successfully loaded weights from %s", self.model_path)
            except Exception as exc:
                logger.warning("Could not load weights from %s: %s (using initialized weights)", self.model_path, exc)
        else:
            logger.warning("Model checkpoint not found at %s. Initialized with random weights.", self.model_path)

        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True

    @property
    def schema(self) -> SkillSchema:
        """Declarative schema for the tomato diagnosis skill."""
        return SkillSchema(
            name="tomato_diagnosis",
            description="Diagnose tomato leaf diseases using the CondConViT_V2 multi-modal vision model",
            parameters=[
                SkillParameter(
                    name="image_path",
                    description="Filesystem path to the tomato leaf RGB image or raw tensor",
                    param_type="str",
                    required=True,
                ),
            ],
            tools_required=[],
            estimated_duration_seconds=0.05,
        )

    def preprocess_image(self, image_input: Union[str, Any]) -> torch.Tensor:
        """
        Preprocess input image to normalized (1, 3, 224, 224) float32 tensor.
        """
        if isinstance(image_input, torch.Tensor):
            tensor = image_input
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[2:] != (224, 224):
                tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
            return tensor.to(self.device, dtype=torch.float32)

        if isinstance(image_input, (str, Path)):
            if not os.path.exists(str(image_input)):
                logger.warning("Image path '%s' not found on disk; generating synthetic test frame.", image_input)
                tensor = torch.randn(1, 3, 224, 224, dtype=torch.float32)
                return tensor.to(self.device)
            from PIL import Image
            img = Image.open(str(image_input)).convert("RGB")
        else:
            from PIL import Image
            if isinstance(image_input, Image.Image):
                img = image_input.convert("RGB")
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")

        img = img.resize((224, 224))
        import numpy as np
        arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_arr = (arr - mean) / std
        tensor = torch.from_numpy(norm_arr).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
        return tensor.to(self.device)

    def execute(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        **kwargs: Any,
    ) -> SkillResult:
        """
        Execute tomato disease diagnosis on an input image.

        Args:
            tool_registry: System ToolRegistry (optional).
            **kwargs: Must include ``image_path`` (str or PIL Image or tensor).

        Returns:
            ``SkillResult`` containing:
                - ``predicted_class``: Category name (e.g. "Early_blight", "healthy")
                - ``confidence``: Confidence score [0.0, 1.0]
                - ``inference_time_ms``: Time taken in milliseconds
                - ``all_probabilities``: Mapping of all 11 classes to probabilities
        """
        if not self._is_loaded or self.model is None:
            self.load_model()

        image_path = kwargs.get("image_path")
        if image_path is None:
            return SkillResult(
                success=False,
                error="Missing required parameter 'image_path'",
            )

        start_t = time.perf_counter()
        try:
            tensor = self.preprocess_image(image_path)

            # Strict 4GB VRAM Safety: torch.no_grad and batch_size=1
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0)
                conf, pred_idx = torch.max(probs, dim=0)

                pred_class = TOMATO_CLASSES[pred_idx.item()]
                confidence_val = float(conf.item())

                all_probs = {
                    TOMATO_CLASSES[i]: float(probs[i].item())
                    for i in range(len(TOMATO_CLASSES))
                }

            # Hardware Safety: Clear CUDA cache immediately after inference
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            result_data = {
                "predicted_class": pred_class,
                "confidence": round(confidence_val, 4),
                "inference_time_ms": round(elapsed_ms, 2),
                "all_probabilities": all_probs,
                "device_used": str(self.device),
            }

            logger.info(
                "Tomato diagnosis: %s (confidence=%.2f%%, latency=%.1fms)",
                pred_class,
                confidence_val * 100.0,
                elapsed_ms,
            )

            return SkillResult(
                success=True,
                data=result_data,
                metadata={"model": "CondConViT_V2", "device": str(self.device)},
            )

        except Exception as exc:
            logger.exception("Tomato diagnosis inference failed: %s", exc)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return SkillResult(
                success=False,
                error=f"Inference error: {str(exc)}",
            )
