
from google.colab import drive
drive.mount('/content/drive')
# mounting the google drive
# ==========================================
# CELL 1: Environment & Standard Library
# ==========================================
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PyTorch Ecosystem
# ==========================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.utils.class_weight import compute_class_weight

# Set device agnostic code (uses GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Computation device set to: {device}")

tomato_dir=r"/content/drive/MyDrive/plant_disease_dataset/data_/Tomato_"
print(f"Checking contents of: {tomato_dir}")

# List contents of the base dataset path (e.g., train, validation, test folders)
if os.path.exists(tomato_dir):
    print(f"Contents found at {tomato_dir}:")
    for item in os.listdir(tomato_dir):
        item_path = os.path.join(tomato_dir, item)
        print(f"- {item} {'(Directory)' if os.path.isdir(item_path) else '(File)'}")
else:
    print(f"Base dataset path does not exist: {tomato_dir}")

train = os.path.join(tomato_dir, 'train') # Assuming 'train' is the directory with class folders


print(f"Folders in '{train}':")

# List all entries in the train_dir
for item in os.listdir(train):
    item_path = os.path.join(train, item)
    # Check if the item is a directory (a class folder)
    if os.path.isdir(item_path):
        print(f"- {item}")




if os.path.exists(train):
    print(f"\nCounting files in each class directory within: {train}")
    class_counts = {}
    for class_name in os.listdir(train):
        class_path = os.path.join(train, class_name)
        if os.path.isdir(class_path):
            # Count only image files (e.g., .jpg, .jpeg, .png)
            num_files = len([name for name in os.listdir(class_path) if name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
            class_counts[class_name] = num_files

    # Sort for consistent output
    sorted_class_counts = sorted(class_counts.items())

    for class_name, count in sorted_class_counts:
        print(f"{class_name}: {count} files")
else:
    print(f"The directory '{train}' does not exist. Please check the dataset structure.")
import shutil
import os
import tensorflow as tf

# 1. Define paths
drive_train_dir = '/content/drive/MyDrive/plant_disease_dataset/data_/Tomato_/train'
drive_valid_dir = '/content/drive/MyDrive/plant_disease_dataset/data_/Tomato_/valid'

local_train_dir = '/content/local_data/train'
local_valid_dir = '/content/local_data/valid'

# 2. Copy from Drive to Local Runtime
def sync_local_data(src, dst):
    if os.path.exists(src):
        if not os.path.exists(dst):
            print(f"Copying {src} to local runtime...")
            shutil.copytree(src, dst)
        else:
            print(f"Local data already exists at {dst}")

sync_local_data(drive_train_dir, local_train_dir)
sync_local_data(drive_valid_dir, local_valid_dir)
import os

def get_directory_structure_and_counts(base_dir):
    structure = {}
    if not os.path.exists(base_dir):
        return None

    for class_name in sorted(os.listdir(base_dir)):
        class_path = os.path.join(base_dir, class_name)
        if os.path.isdir(class_path):
            num_files = len([name for name in os.listdir(class_path) if name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
            structure[class_name] = num_files
    return structure

# Define the two paths to compare
local_train_path = '/content/local_data/train'
drive_train_path = os.path.join(tomato_dir, 'train')

print(f"Analyzing: {local_train_path}")
local_structure = get_directory_structure_and_counts(local_train_path)

print(f"\nAnalyzing: {drive_train_path}")
drive_structure = get_directory_structure_and_counts(drive_train_path)


print("\n--- Comparison Results ---")
if local_structure is None:
    print(f"Error: '{local_train_path}' does not exist.")
elif drive_structure is None:
    print(f"Error: '{drive_train_path}' does not exist. (Check 'tomato_dir' variable and its contents)")
else:
    # Compare class names
    local_classes = set(local_structure.keys())
    drive_classes = set(drive_structure.keys())

    if local_classes == drive_classes:
        print(" Class directories are identical in both paths.")

        # Compare file counts for each class
        differences_found = False
        for class_name in sorted(local_classes):
            local_count = local_structure.get(class_name, 0)
            drive_count = drive_structure.get(class_name, 0)
            if local_count != drive_count:
                print(f"   Mismatch in '{class_name}': Local has {local_count} files, Drive has {drive_count} files.")
                differences_found = True
            else:
                print(f"   '{class_name}': {local_count} files (Match)")

        if not differences_found:
            print("\n All class file counts match between local and Google Drive training directories!")
        else:
            print("\n Differences in file counts were found. Data inconsistency detected.")
    else:
        print(" Class directories are NOT identical between the two paths.")
        print(f"  Classes unique to local: {local_classes - drive_classes}")
        print(f"  Classes unique to Drive: {drive_classes - local_classes}")
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2

class AddGaussianNoise(object):
    """Custom transform to simulate drone camera sensor noise with a probability."""
    def __init__(self, std=0.03, p=0.3): # 30% chance to apply noise
        self.std = std
        self.p = p

    def __call__(self, tensor):
        # Only apply noise 'p' percent of the time
        if torch.rand(1).item() < self.p:
            noise = torch.randn(tensor.size()) * self.std
            return torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor

# The updated pipeline that mathematically alters pixels to force invariant feature learning
drone_augmentation_v2 = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),

    # 50% chance to rotate or scale
    v2.RandomApply([v2.RandomRotation(degrees=72)], p=0.5),
    v2.RandomApply([v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2))], p=0.5),

    # 30% chance to mess with the lighting
    v2.RandomApply([v2.ColorJitter(brightness=0.1, contrast=0.1)], p=0.3),

    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True), # Scales to [0, 1]

    # 30% chance to add drone static
    AddGaussianNoise(std=0.03, p=0.3)
])

batch_size = 32
img_height = 224
img_width = 224

train_dir = os.path.join(tomato_dir, 'train')
valid_dir = os.path.join(tomato_dir, 'valid')

data_transforms = {
    'train': v2.Compose([
        v2.Resize((img_height, img_width)),
        drone_augmentation_v2,
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'valid': v2.Compose([
        v2.Resize((img_height, img_width)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
    'valid': DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

print(f"✅ Training images: {dataset_sizes['train']}")
print(f"✅ Validation images: {dataset_sizes['valid']}")
# ==========================================
# CELL 4: Computing Class Weights (PyTorch Format)
# ==========================================
print("="*60)
print("COMPUTING CLASS WEIGHTS TO COMBAT IMBALANCE")
print("="*60)

# Create a mock array of labels for sklearn
y_train_mock = []
for i, class_name in enumerate(class_names):
    count = class_counts[class_name]
    y_train_mock.extend([i] * count)

# Compute the weights
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_mock),
    y=y_train_mock
)

# FIXED: Clamp weights to prevent rare classes from destroying precision
weights = np.clip(weights, a_min=None, a_max=1.5)

# Convert the numpy array to a PyTorch Tensor and move it to the GPU/CPU
class_weights_tensor = torch.FloatTensor(weights).to(device)

for i, class_name in enumerate(class_names):
    print(f"{class_name}: Weight = {weights[i]:.4f}")

# Define the Loss function using these weights
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
print("\n Loss function (CrossEntropyLoss) initialized with balanced weights.")
# ==========================================
# CELL 8 & 9: Routing and Conditional Convolutions
# ==========================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class Routing(nn.Module):
    """
    Looks at the incoming feature map and decides which expert convolutions
    should be used and by how much, using a temperature-scaled Softmax.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.2, temperature=30):
        super(Routing, self).__init__()
        self.temperature = temperature

        # Global Average Pooling: (B, C, H, W) -> (B, C, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Dense layer expects just the channel dimension
        self.fc = nn.Linear(in_channels, out_channels)

        # Explicit Initialization: He Normal (kaiming) for weights, 0.0 for bias
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Flatten (B, C, 1, 1) to (B, C)
        x = self.dropout(x)

        # Apply temperature scaling before Softmax
        x = self.fc(x) / self.temperature
        return F.softmax(x, dim=1)


class CondConv2D(nn.Module):
    """
    The factory that routes inputs through multiple expert convolutions.
    FIXED: Uses dynamic kernel mixing via grouped convolutions for O(1) FLOP overhead.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True, num_experts=3):
        super(CondConv2D, self).__init__()
        self.num_experts = num_experts

        self.routing = Routing(in_channels, out_channels=num_experts)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias)
            for _ in range(num_experts)
        ])

        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            if bias:
                nn.init.constant_(conv.bias, 0.0)

    def forward(self, x):
        B, C, H, W = x.shape
        routing_weights = self.routing(x) # Shape: (B, num_experts)

        # 1. Stack expert weights and biases
        expert_weights = torch.stack([conv.weight for conv in self.convs]) # (E, O, I, K, K)
        expert_biases = torch.stack([conv.bias for conv in self.convs]) if self.convs[0].bias is not None else None

        # 2. Dynamically mix kernels per batch element using einsum
        # mixed_weights: (B, O, I, K, K)
        mixed_weights = torch.einsum('be,eoikl->boikl', routing_weights, expert_weights)

        # 3. Reshape for grouped conv: Treat the batch dimension as convolution groups
        mixed_weights = mixed_weights.reshape(B * self.convs[0].out_channels, C, self.convs[0].kernel_size[0], self.convs[0].kernel_size[1])
        x_reshaped = x.reshape(1, B * C, H, W)

        if expert_biases is not None:
            mixed_biases = torch.einsum('be,eo->bo', routing_weights, expert_biases).reshape(-1)
        else:
            mixed_biases = None

        # 4. Run a single convolution on the whole batch
        out = F.conv2d(x_reshaped, mixed_weights, bias=mixed_biases,
                       stride=self.convs[0].stride, padding=self.convs[0].padding, groups=B)

        return out.reshape(B, self.convs[0].out_channels, out.shape[2], out.shape[3])
# ==========================================
# CELL 11: Inception Block with Separable Convs
# ==========================================

class SeparableConv2d(nn.Module):
    """PyTorch equivalent of Keras layers.SeparableConv2D"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SeparableConv2d, self).__init__()

        # Depthwise: groups=in_channels forces the kernel to look at one channel at a time
        # We calculate padding mathematically to match Keras's padding='same'
        if isinstance(kernel_size, int):
            pad = kernel_size // 2
        else:
            pad = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=pad, groups=in_channels, bias=bias)

        # Pointwise: 1x1 convolution to recombine the features
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, nb_filter):
        super(InceptionBlock, self).__init__()

        # Branch 1x1
        self.branch1x1 = SeparableConv2d(in_channels, nb_filter, kernel_size=1)

        # Branch 3x3 (Cross)
        self.branch3x3_base = SeparableConv2d(in_channels, nb_filter, kernel_size=1)
        self.branch3x3_1 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(3,1))
        self.branch3x3_2 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(1,3))

        # Branch 5x5 (Cross)
        self.branch5x5_base = SeparableConv2d(in_channels, nb_filter, kernel_size=1)
        self.branch5x5_1 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(3,1))
        self.branch5x5_2 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(1,3))
        self.branch5x5_final1 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(3,1))
        self.branch5x5_final2 = SeparableConv2d(nb_filter, nb_filter, kernel_size=(1,3))

        # Branch Pool
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

        # Concatenate along the channel dimension (dim=1 in PyTorch, not axis=3)
        return torch.cat([b1, b3, b5, bp], dim=1)
class PatchTokenizer(nn.Module):
    """
    Slices the feature map into patches and projects them into a flat sequence.
    Replaces both Keras `Patches` and `PatchEncoder` classes.
    """
    def __init__(self, in_channels, patch_size, embed_dim, img_size):
        super(PatchTokenizer, self).__init__()
        self.patch_size = patch_size

        # 1. Simultaneous Patch Extraction and Linear Projection
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 2. Calculate the sequence length
        self.num_patches = (img_size // patch_size) ** 2

        # 3. Learnable Positional Embeddings
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        # print("=" * 60)
        # print(f"INPUT TO TOKENIZER: {x.shape}")

        x = self.projection(x)
        # print(f"PROJECTED PATCHES (2D): {x.shape}")

        x = x.flatten(2).transpose(1, 2)
        # print(f"FLATTENED SEQUENCE: {x.shape}")

        x = x + self.position_embedding
        # print(f"FINAL OUTPUT WITH POSITIONS: {x.shape}")
        # print("=" * 60)

        return x
# ==========================================
# CELL 14: SSE Block
# ==========================================

class SSEBlock(nn.Module):
    """Implementation of Squeeze-and-Excitation(SE) block with Spatial Stats."""
    def __init__(self, in_channels, ratio=4):
        super(SSEBlock, self).__init__()

        # Squeeze & Excitation Path
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        # SE Path
        se = self.avg_pool(x).view(b, c) # Squeeze
        se = self.fc1(se)                # Excite
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se).view(b, c, 1, 1) # Reshape to broadcast

        # Multiply input by SE features
        se_out = x * se

        # Spatial Statistics Path (reducing across the Channel dimension, dim=1)
        # keepdim=True ensures the output is (B, 1, H, W) for concatenation
        spatial_mean = torch.mean(x, dim=1, keepdim=True)
        spatial_std = torch.std(x, dim=1, keepdim=True)

        # torch.max returns a tuple of (values, indices), we extract index 0 (values)
        spatial_max = torch.max(x, dim=1, keepdim=True)[0]

        # Concatenate all features along the Channel dimension
        return torch.cat([se_out, spatial_mean, spatial_std, spatial_max], dim=1)
# ==========================================
# CELL 15: Inspecting the Drone Simulator Output
# ==========================================
import matplotlib.pyplot as plt
import numpy as np

print("="*60)
print("INSPECTING THE V2 DATA PIPELINE: DRONE SIMULATOR OUTPUT")
print("="*60)

# Assuming 'dataloaders' is your PyTorch DataLoader dictionary from Phase 1
def visualize_augmentations(dataloader, class_names):
    # Fetch exactly one batch of data
    images, labels = next(iter(dataloader))

    plt.figure(figsize=(15, 12))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)

        # Shift from (C, H, W) to (H, W, C) for Matplotlib
        img = images[i].permute(1, 2, 0).numpy()

        # Images are already transformed to [0, 1] by the DataLoader v2 pipeline
        img_display = np.clip(img, 0, 1)

        # In PyTorch CrossEntropyLoss, labels are integers, not one-hot encoded
        label_index = labels[i].item()
        true_label_name = class_names[label_index]

        plt.imshow(img_display)
        plt.title(f"Label: {true_label_name}", fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# To execute:
visualize_augmentations(dataloaders['train'], class_names)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models.feature_extraction import create_feature_extractor

# ==========================================
# PHASE 1: CONDCONVIT REBUILT IN PYTORCH (STRICT SPATIAL ALIGNMENT)
# ==========================================
class CondConViT_V2(nn.Module):
    def __init__(self, num_classes=11, dropout_rate=0.4):
        super(CondConViT_V2, self).__init__()

        # --- 1. THE BACKBONE (MobileNetV2) ---
        base_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # CORRECTED HOOKS: Ensuring spatial grids are 112, 56, and 28
        return_nodes = {
            'features.1': 'x1',    # Spatial: 112x112, Channels: 16
            'features.3': 'x2',    # Spatial: 56x56, Channels: 24
            'features.6': 'x_inc'  # Spatial: 28x28, Channels: 32
        }
        self.backbone = create_feature_extractor(base_model, return_nodes=return_nodes)

        for param in self.backbone.parameters():
            param.requires_grad = False

        # --- 2. MULTI-SCALE FEATURE EXTRACTION ---

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

        # --- 3. VISION TRANSFORMER PATH (HIGH DROPOUT) ---
        # The Inception block outputs 128 channels (32 * 4 branches). Spatial is perfectly 28x28.
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
        self.classifier = nn.Linear(32, num_classes) # Brilliant fix matching SSE output

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

        # Exact mathematical reshaping to 4x4 spatial grid
        b = vit_out.size(0)
        vit_spatial = vit_out.transpose(1, 2).view(b, 32, 4, 4)

        vit_spatial = F.interpolate(vit_spatial, size=(11, 11), mode='bilinear', align_corners=False)
        vit_spatial = self.vit_projection(vit_spatial)

        # Spatial dimensions naturally match 11x11 now due to correct stride math
        merged = p1 + p2 + p3 + vit_spatial
        merged = self.fusion_sse(merged)

        out = self.global_pool(merged)
        out = torch.flatten(out, 1)

        return self.classifier(out)

# Instantiate the model to verify
model = CondConViT_V2().to(device)
print("✅ CondConViT_V2 successfully built with absolute spatial integrity!")
pip install torchinfo
from torchinfo import summary

# Assuming typical input size for a single image (batch_size=1, channels=3, height, width)
# img_height and img_width are defined in a previous cell (erdwNADbJz2j) as 224.

# Ensure the model is on the correct device before summarizing
model.to(device)
summary(model, input_size=(1, 3, img_height, img_width))
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from tqdm import tqdm

print("=" * 60)
print("PHASE 2 & 3: PYTORCH REGULARIZED TRAINING (RESUMABLE)")
print("=" * 60)

# 1. Setup Checkpoint Directory
checkpoint_dir = "/content/drive/MyDrive/plant_disease_dataset/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
v2_checkpoint_path = os.path.join(checkpoint_dir, "best_model_v7.pth")

# 2. Build optimizer/scheduler BEFORE trying to load, so their state_dicts exist to fill in
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

early_stopping_patience = 8  # was previously missing from the resume cell -> NameError risk
checkpoint_every_n_batches = 100  # mid-epoch safety save; tune to taste (808 batches/epoch here)

# 3. Full-state resume: model + optimizer + scheduler + progress counters
start_epoch = 0
best_val_loss = float('inf')
patience_counter = 0

midepoch_path = v2_checkpoint_path.replace('.pth', '_midepoch.pth')

# If a mid-epoch safety save exists, prefer it (it's newer progress than the last
# fully-completed epoch). We don't resume mid-batch-loop (too fragile) — we just
# restart that same epoch from its beginning with the more recent weights.
if os.path.exists(midepoch_path):
    print(f"Found mid-epoch safety checkpoint: {midepoch_path}")
    checkpoint = torch.load(midepoch_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # re-do the interrupted epoch from scratch
    best_val_loss = checkpoint['best_val_loss']
    patience_counter = checkpoint['patience_counter']
    print(f"Restored mid-epoch state (was {checkpoint['mid_epoch_batch']} batches into "
          f"epoch {start_epoch + 1}). Re-running epoch {start_epoch + 1} from the start.")
elif os.path.exists(v2_checkpoint_path):
    print(f"Resuming training from checkpoint: {v2_checkpoint_path}")
    checkpoint = torch.load(v2_checkpoint_path, map_location=device)

    # Backward-compatible: if it's an old-style checkpoint (just raw weights, not a dict
    # with these keys), fall back to loading only the model weights.
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        print(f"Full state restored. Continuing from epoch {start_epoch + 1}, "
              f"best_val_loss={best_val_loss:.4f}, patience={patience_counter}/{early_stopping_patience}")
    else:
        print("Old-format checkpoint detected (weights only). Loading weights; "
              "optimizer/scheduler/epoch/patience will restart fresh.")
        model.load_state_dict(checkpoint)
else:
    print("No existing checkpoint found. Starting training from scratch.")

best_model_weights = copy.deepcopy(model.state_dict())

epochs = 100
scaler = torch.amp.GradScaler('cuda')  # modern API, no deprecation warning

# 4. Training Loop
for epoch in range(start_epoch, epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print("-" * 20)

    # --- TRAINING ---
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    train_loader = tqdm(dataloaders['train'], desc="Training", leave=False)
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        train_loader.set_postfix({'Loss': f"{loss.item():.4f}"})

        # Mid-epoch safety net: if Colab disconnects mid-epoch, you lose at most
        # `checkpoint_every_n_batches` batches of progress instead of the whole epoch.
        # Saved as "mid-epoch" state, separate from the real end-of-epoch best checkpoint,
        # so it never overwrites your actual best model.
        if (batch_idx + 1) % checkpoint_every_n_batches == 0:
            torch.save({
                'epoch': epoch - 1,  # still mid this epoch, so record the last COMPLETED epoch
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'mid_epoch_batch': batch_idx + 1,
            }, v2_checkpoint_path.replace('.pth', '_midepoch.pth'))

    epoch_train_loss = train_loss / dataset_sizes['train']
    epoch_train_acc = correct_train / total_train

    # --- VALIDATION ---
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    val_loader = tqdm(dataloaders['valid'], desc="Validating", leave=False)
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # FIXED: labels now moved too

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    epoch_val_loss = val_loss / dataset_sizes['valid']
    epoch_val_acc = correct_val / total_val

    print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
    print(f"Val Loss: {epoch_val_loss:.4f}   | Val Acc: {epoch_val_acc:.4f}")

    # --- CALLBACKS ---
    scheduler.step(epoch_val_loss)

    improved = epoch_val_loss < best_val_loss
    if improved:
        best_val_loss = epoch_val_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1

    # Always save FULL state, not just weights, so a future resume is a true resume.
    torch.save({
        'epoch': epoch,
        'model_state_dict': best_model_weights if improved else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
    }, v2_checkpoint_path)

    # This epoch finished cleanly, so the mid-epoch safety file is stale — remove it
    # so future resumes use the proper end-of-epoch checkpoint instead.
    if os.path.exists(midepoch_path):
        os.remove(midepoch_path)

    if improved:
        print(f"✅ Val Loss improved! Saved checkpoint to {v2_checkpoint_path}")
    else:
        print(f"⚠️ Val Loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
        if patience_counter >= early_stopping_patience:
            print(f"\n⏹️ Early stopping triggered at epoch {epoch + 1}.")
            break

# 5. Restore best weights
print("\nLoading best weights into model...")
model.load_state_dict(best_model_weights)
print("✅ Training complete.")
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from tqdm import tqdm

print("="*60)
print("PHASE 2 & 3: PYTORCH REGULARIZED TRAINING INITIATED")
print("="*60)

# 1. Setup Checkpoint Directory
checkpoint_dir = "/content/drive/MyDrive/plant_disease_dataset/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
v2_checkpoint_path = os.path.join(checkpoint_dir, "best_model_v6.pth")

# 2. Optimizer (AdamW), Loss (Class Weights + Label Smoothing)
# UPGRADE: AdamW provides decoupled weight decay for better transformer regularization
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

# 3. Scheduler & Early Stopping
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

early_stopping_patience = 8
patience_counter = 0
best_val_loss = float('inf')
best_model_weights = copy.deepcopy(model.state_dict())

epochs = 100

# UPGRADE: Initialize AMP Scaler to save VRAM and speed up training
scaler = torch.cuda.amp.GradScaler()

# 4. Training Loop
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    print("-" * 20)

    # --- TRAINING ---
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    train_loader = tqdm(dataloaders['train'], desc="Training", leave=False)
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # UPGRADE: Autocast for Mixed Precision
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # UPGRADE: Scaled backward pass
        scaler.scale(loss).backward()

        # UPGRADE: Gradient Clipping to prevent routing instability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        train_loader.set_postfix({'Loss': f"{loss.item():.4f}"})

    epoch_train_loss = train_loss / dataset_sizes['train']
    epoch_train_acc = correct_train / total_train

    # --- VALIDATION ---
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    val_loader = tqdm(dataloaders['valid'], desc="Validating", leave=False)
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Use autocast in validation as well for speed
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    epoch_val_loss = val_loss / dataset_sizes['valid']
    epoch_val_acc = correct_val / total_val

    print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
    print(f"Val Loss: {epoch_val_loss:.4f}   | Val Acc: {epoch_val_acc:.4f}")

    # --- CALLBACKS ---
    scheduler.step(epoch_val_loss)

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        torch.save(best_model_weights, v2_checkpoint_path)
        print(f"✅ Val Loss improved! Saved weights to {v2_checkpoint_path}")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"⚠️ Val Loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
        if patience_counter >= early_stopping_patience:
            print(f"\n⏹️ Early stopping triggered at epoch {epoch+1}.")
            break

# 5. Restore best weights
print("\nLoading best weights into model...")
model.load_state_dict(best_model_weights)
print("✅ Training complete. Baseline established.")
import torch

# 1. Instantiate the model EXACTLY as you did before training.
# We know from your dataset that you have 11 classes.
# (If your CondConViT_V2 init requires other arguments like embed_dim, add them here)
model = CondConViT_V2(num_classes=11)
model = model.to(device)

# 2. Safely load your local Windows path using a raw string (r"...")
print("Loading weights from local drive...")
weights_path = r'D:\agentic_agriculture\model_\best_model_v2.pth'

# 3. Load the state dictionary to the correct device
state_dict = torch.load(weights_path, map_location=device)

# 4. Apply weights to the model with strict checking enabled
# strict=True forces PyTorch to check that every single layer matches exactly.
load_result = model.load_state_dict(state_dict, strict=True)

print(load_result)

# 5. Set to evaluation mode for inference
model.eval()
print("Model successfully loaded and set to evaluation mode!")
generate_forensics(model, dataloaders['valid'], class_names, device)
import os
import zipfile
from tqdm.notebook import tqdm

source_dir = '/kaggle/working/checkpoints'
output_zip = 'model_download.zip'

# 1. Get a list of all files to compress
file_list = []
for root, dirs, files in os.walk(source_dir):
    for file in files:
        file_list.append(os.path.join(root, file))

# 2. Zip the files with a progress bar
with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_path in tqdm(file_list, desc="Zipping model files"):
        # This keeps the correct folder structure inside the zip
        arcname = os.path.relpath(file_path, start=source_dir)
        zipf.write(file_path, arcname)

print("Zip file created successfully!")

# 3. Create the download link
from IPython.display import FileLink
FileLink('model_download.zip')

import torch
from torch.utils.data import DataLoader
import numpy as np

# ---------- Embedding Extractor ----------
class EmbeddingModel(nn.Module):
    """Wraps CondConViT_V2 to return the 32‑dim embedding instead of logits."""
    def __init__(self, original_model):
        super().__init__()
        self.original = original_model

    def forward(self, x):
        # Replicate the forward until the classifier
        features = self.original.backbone(x)
        x1, x2, x_inc = features['x1'], features['x2'], features['x_inc']

        p1 = self.original.p1_proj(self.original.sse1(self.original.conv1_final(self.original.cond1_3(self.original.cond1_2(self.original.cond1_1(x1))))))
        p2 = self.original.p2_proj(self.original.sse2(self.original.conv2_final(self.original.cond2_2(self.original.cond2_1(x2)))))
        inc_out = self.original.inception(x_inc)
        p3 = self.original.p3_proj(self.original.sse3(self.original.conv3_final(self.original.cond3_1(inc_out))))

        tokens = self.original.tokenizer(inc_out)
        tokens = self.original.vit_dropout(tokens)
        vit_out = self.original.transformer(tokens)
        vit_out = self.original.final_norm(vit_out)
        b = vit_out.size(0)
        vit_spatial = vit_out.transpose(1, 2).view(b, 32, 4, 4)
        vit_spatial = F.interpolate(vit_spatial, size=(11, 11), mode='bilinear', align_corners=False)
        vit_spatial = self.original.vit_projection(vit_spatial)

        merged = p1 + p2 + p3 + vit_spatial
        merged = self.original.fusion_sse(merged)

        # Pool to 32‑dim vector – this is your embedding
        embedding = self.original.global_pool(merged).flatten(1)
        return embedding

# Load the best model weights
model.load_state_dict(torch.load(r'D:\agentic_agriculture\model_\best_model_v2.pth', map_location=device))
model.eval()

embedding_model = EmbeddingModel(model).to(device)
embedding_model.eval()

# ---------- Collect embeddings and labels ----------
all_embeddings = []
all_labels = []

with torch.no_grad():
    for inputs, labels in dataloaders['valid']:
        inputs = inputs.to(device)
        emb = embedding_model(inputs)
        all_embeddings.append(emb.cpu().numpy())
        all_labels.append(labels.numpy())

embeddings = np.concatenate(all_embeddings, axis=0)   # shape (N, 32)
labels = np.concatenate(all_labels, axis=0)            # shape (N,)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Labels shape: {labels.shape}")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Optional: install umap
# !pip install umap-learn
import umap

# Map class indices to names
idx_to_class = {i: name for i, name in enumerate(class_names)}

# ---------- 2D Projections ----------
# Standardize embeddings (optional, helps PCA/t-SNE)
scaler = StandardScaler()
emb_scaled = scaler.fit_transform(embeddings)

# PCA
pca = PCA(n_components=2)
pca_emb = pca.fit_transform(emb_scaled)

# t-SNE (perplexity ~30, you can tune)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_emb = tsne.fit_transform(emb_scaled)

# UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
umap_emb = umap_reducer.fit_transform(emb_scaled)

# Plot function
def plot_projection(proj, title, labels, class_names):
    plt.figure(figsize=(12, 10))
    for i, name in enumerate(class_names):
        idx = labels == i
        plt.scatter(proj[idx, 0], proj[idx, 1], label=name, alpha=0.6, s=5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=4)
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_projection(pca_emb, 'PCA of 32‑dim Embeddings', labels, class_names)
plot_projection(tsne_emb, 't‑SNE of 32‑dim Embeddings', labels, class_names)
plot_projection(umap_emb, 'UMAP of 32‑dim Embeddings', labels, class_names)
# ---------- Cosine Similarity Matrix ----------
# Compute class prototypes (mean embedding per class)
class_means = np.array([embeddings[labels == i].mean(axis=0) for i in range(num_classes)])

# Cosine similarity between class prototypes
cos_sim = cosine_similarity(class_means)

plt.figure(figsize=(10, 8))
sns.heatmap(cos_sim, annot=True, fmt=".2f", xticklabels=class_names, yticklabels=class_names,
            cmap='coolwarm', vmin=-1, vmax=1, square=True)
plt.title('Cosine Similarity Between Class Prototype Embeddings')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Print the most similar/different pairs
# (optional analysis)
# ---------- Intra‑class variance & Inter‑class distances ----------
intra_vars = []
for i in range(num_classes):
    class_emb = embeddings[labels == i]
    intra_vars.append(np.mean(np.var(class_emb, axis=0)))

inter_dists = []
for i in range(num_classes):
    for j in range(i+1, num_classes):
        inter_dists.append(np.linalg.norm(class_means[i] - class_means[j]))

print("Intra‑class variance (mean per class):", intra_vars)
print(f"Average intra‑class variance: {np.mean(intra_vars):.4f}")
print(f"Average inter‑class distance: {np.mean(inter_dists):.4f}")

# Ratio: larger is better (more separation relative to spread)
ratio = np.mean(inter_dists) / np.mean(intra_vars) if np.mean(intra_vars) > 0 else float('inf')
print(f"Inter/intra ratio: {ratio:.2f}")
import os

# Memory footprint helper
def get_model_size(model):
    """Return model size in MB."""
    torch.save(model.state_dict(), "temp.pth")
    size_mb = os.path.getsize("temp.pth") / (1024 * 1024)
    if os.path.exists("temp.pth"):
        os.remove("temp.pth")
    return size_mb

print(f"FP32 model size: {get_model_size(model_cpu):.2f} MB")
print(f"FP16 model size: {get_model_size(model_fp16):.2f} MB")
# Get all predictions and compare
model.eval()
all_preds = []
all_targets = []
misclassified_idx = []

with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(dataloaders['valid']):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.numpy())

        # Track indices of misclassified images (global index)
        batch_size = inputs.size(0)
        start_idx = batch_idx * 32
        for i in range(batch_size):
            if preds[i] != labels[i]:
                misclassified_idx.append(start_idx + i)

# Filter for powdery mildew class index
pm_class_idx = class_names.index('powdery_mildew')

# FIXED: Use all_targets instead of the batch-specific 'labels' variable
pm_mis = [idx for idx in misclassified_idx if all_targets[idx] == pm_class_idx]

print(f"Total misclassified across all classes: {len(misclassified_idx)}")
print(f"Powdery mildew specifically misclassified: {len(pm_mis)}")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

# Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        # Using register_full_backward_hook for modern PyTorch versions
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy()

# Target the fusion_sse layer (the rich spatial features right before pooling)
target_layer = model.fusion_sse
gradcam = GradCAM(model, target_layer)

# Visualize the misclassified powdery mildew samples
num_show = min(5, len(pm_mis))
fig, axes = plt.subplots(num_show, 2, figsize=(8, 3*num_show))

# Handle the edge case if there's only 1 misclassified image to show
if num_show == 1:
    axes = [axes]

for i, idx in enumerate(pm_mis[:num_show]):
    # Get the image directly from the dataset (cell 3)
    img_tensor, true_label = image_datasets['valid'][idx]
    img_tensor_batch = img_tensor.unsqueeze(0).to(device)

    # Original image for display
    pil_img = to_pil_image(img_tensor)

    # Generate Grad-CAM for the TRUE class to see why it missed it
    cam = gradcam.generate(img_tensor_batch, class_idx=pm_class_idx)

    # Overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (224, 224))

    overlay = heatmap * 0.4 + np.array(pil_img) * 0.6
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    axes[i, 0].imshow(pil_img)
    axes[i, 0].set_title(f"True: {class_names[true_label]}")
    axes[i, 0].axis('off')

    axes[i, 1].imshow(overlay)
    axes[i, 1].set_title(f"Grad-CAM Focus")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()