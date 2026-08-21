
# this acts as the central brain for project settings 
# of the model architecture
# 
# it consits of the paths , hyperparameters and device configurations  

import os
import torch

# Paths
BASE_DIR = r"D:\agentic_agriculture\model_\model_architectures_files"
DATA_DIR = r"D:\agentic_agriculture\datasets\soyanet_cleaned"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
EPOCHS = 5  # Reduced to 5 for initial CPU testing
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
NUM_CLASSES = 2  # Disease vs Healthy

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
