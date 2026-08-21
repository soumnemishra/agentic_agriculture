from typing import cast
import torch
import torch.nn as nn
import torch.nn.functional as F

class Routing(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2, temperature=30):
        super(Routing, self).__init__()
        self.temperature = temperature
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_channels, out_channels)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x) / self.temperature
        return F.softmax(x, dim=1)

class CondConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=1, bias: bool = True, num_experts: int = 3):
        super(CondConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        self.num_experts = num_experts
        self.routing = Routing(in_channels, out_channels=num_experts)
        convs = []
        for _ in range(num_experts):
            conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias
            )
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0.0)
            convs.append(conv)
        self.convs = nn.ModuleList(convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        routing_weights = self.routing(x)
        expert_weights = torch.stack([cast(torch.Tensor, conv.weight) for conv in self.convs])
        expert_biases = (
            torch.stack([cast(torch.Tensor, conv.bias) for conv in self.convs])
            if self.use_bias
            else None
        )
        
        mixed_weights = torch.einsum('be,eoikl->boikl', routing_weights, expert_weights)
        mixed_weights = mixed_weights.reshape(B * self.out_channels, C, self.kernel_size[0], self.kernel_size[1])
        x_reshaped = x.reshape(1, B * C, H, W)
        
        if expert_biases is not None:
            mixed_biases = torch.einsum('be,eo->bo', routing_weights, expert_biases).reshape(-1)
        else:
            mixed_biases = None
            
        out = F.conv2d(x_reshaped, mixed_weights, bias=mixed_biases,
                       stride=self.stride, padding=self.padding, groups=B)
        return out.reshape(B, self.out_channels, out.shape[2], out.shape[3])

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SeparableConv2d, self).__init__()
        if isinstance(kernel_size, int):
            pad = kernel_size // 2
        else:
            pad = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=pad, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, nb_filter):
        super(InceptionBlock, self).__init__()
        self.branch1x1 = SeparableConv2d(in_channels, nb_filter, kernel_size=1)
        
        self.branch3x3_base = SeparableConv2d(in_channels, nb_filter, kernel_size=1)
        self.branch3x3_1 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(3,1))
        self.branch3x3_2 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(1,3))
        
        self.branch5x5_base = SeparableConv2d(in_channels, nb_filter, kernel_size=1)
        self.branch5x5_1 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(3,1))
        self.branch5x5_2 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(1,3))
        self.branch5x5_final1 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(3,1))
        self.branch5x5_final2 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(1,3))
        
        self.branchpool_max = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branchpool_conv = SeparableConv2d(in_channels, nb_filter, kernel_size=1)

    def forward(self, x):
        b1 = self.branch1x1(x)
        
        b3_base = self.branch3x3_base(x)
        b3 = self.branch3x3_1(b3_base) + self.branch3x3_2(b3_base)
        
        b5_base = self.branch5x5_base(x)
        b5_mid = self.branch5x5_1(b5_base) + self.branch5x5_2(b5_base)
        b5 = self.branch5x5_final1(b5_mid) + self.branch5x5_final2(b5_mid)
        
        bp = self.branchpool_conv(self.branchpool_max(x))
        return torch.cat([b1, b3, b5, bp], dim=1)

class PatchTokenizer(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, img_size):
        super(PatchTokenizer, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.position_embedding
        return x

class SSEBlock(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(SSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
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
