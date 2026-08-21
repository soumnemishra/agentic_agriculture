"""
Generate Confusion Matrix and Comprehensive Classification Metrics
for CondConViT_V2 (best_model_v7.pth)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import v2
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Add model architecture path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ARCH_DIR = os.path.join(CURRENT_DIR, "model_architectures_files")
if ARCH_DIR not in sys.path:
    sys.path.insert(0, ARCH_DIR)

from model import CondConViT_V2


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Confusion Matrix for best_model_v7.pth")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(CURRENT_DIR, "best_model_v7.pth"),
        help="Path to the trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"D:\agentic_agriculture\datasets\dataset-compiled-training\Tomato_dataset\valid",
        help="Path to dataset directory (validation or test split with class subfolders)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(CURRENT_DIR, "evaluation_results"),
        help="Directory to save confusion matrix plots and reports",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for data loader",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional: Limit total images evaluated (e.g. 500 for rapid test, None for all)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image height and width (default: 224)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Number of CPU threads for PyTorch inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on ('cuda' or 'cpu')",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, num_classes: int, device: torch.device) -> nn.Module:
    """Instantiate and load model weights from checkpoint."""
    print(f"[*] Initializing CondConViT_V2 ({num_classes} classes)...", flush=True)
    model = CondConViT_V2(num_classes=num_classes)
    
    print(f"[*] Loading weights from: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict()
        
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print("[+] Model weights successfully loaded!", flush=True)
    return model


def get_data_loader(data_dir: str, img_size: int, batch_size: int, max_samples: int = None):
    """Setup dataset and dataloader with standardized transforms."""
    eval_transforms = v2.Compose([
        v2.Resize((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    full_dataset = datasets.ImageFolder(data_dir, transform=eval_transforms)
    class_names = full_dataset.classes

    if max_samples and max_samples < len(full_dataset):
        indices = np.random.RandomState(42).choice(len(full_dataset), size=max_samples, replace=False)
        dataset = Subset(full_dataset, indices)
        print(f"[*] Subsampling {max_samples} images from {len(full_dataset)} total images for quick evaluation.", flush=True)
    else:
        dataset = full_dataset

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 0 is safe for Windows
        pin_memory=torch.cuda.is_available(),
    )
    return loader, class_names, len(dataset)


def plot_confusion_matrices(cm, class_names, output_dir):
    """Plot and save both raw count and normalized confusion matrices."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Raw Count Confusion Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Sample Count"},
        linewidths=0.5,
        linecolor="#e0e0e0"
    )
    plt.title("Confusion Matrix (Counts) — CondConViT-V2 (best_model_v7)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("True Label", fontsize=12, fontweight="bold", labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    count_plot_path = os.path.join(output_dir, "confusion_matrix_counts.png")
    plt.savefig(count_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {count_plot_path}", flush=True)

    # 2. Normalized Confusion Matrix (by true class / Recall)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="crest",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Proportion (Recall)"},
        linewidths=0.5,
        linecolor="#e0e0e0"
    )
    plt.title("Normalized Confusion Matrix — CondConViT-V2 (best_model_v7)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("True Label", fontsize=12, fontweight="bold", labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    norm_plot_path = os.path.join(output_dir, "confusion_matrix_normalized.png")
    plt.savefig(norm_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {norm_plot_path}", flush=True)


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cpu" and args.threads:
        torch.set_num_threads(args.threads)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70, flush=True)
    print("  ACA MODEL EVALUATION & CONFUSION MATRIX GENERATOR", flush=True)
    print("=" * 70, flush=True)
    print(f"Device       : {device} (threads={torch.get_num_threads()})", flush=True)
    print(f"Model Path   : {args.model_path}", flush=True)
    print(f"Dataset Path : {args.data_dir}", flush=True)
    print(f"Output Dir   : {args.output_dir}", flush=True)
    print("-" * 70, flush=True)

    # 1. Load Data
    data_loader, class_names, total_samples = get_data_loader(
        args.data_dir, args.img_size, args.batch_size, args.max_samples
    )
    num_classes = len(class_names)
    print(f"[+] Found {total_samples} images across {num_classes} classes.", flush=True)
    for idx, name in enumerate(class_names):
        print(f"    [{idx:2d}] {name}", flush=True)
    print("-" * 70, flush=True)

    # 2. Load Model
    model = load_model(args.model_path, num_classes=num_classes, device=device)

    # 3. Inference Loop
    print(f"\n[*] Running model inference on {total_samples} samples...", flush=True)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating", unit="batch", file=sys.stdout):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 4. Metrics & Reports
    accuracy = (all_preds == all_labels).mean()
    cm = confusion_matrix(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    report_str = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)

    print("\n" + "=" * 70, flush=True)
    print(f"  EVALUATION RESULTS — OVERALL ACCURACY: {accuracy * 100:.2f}%", flush=True)
    print("=" * 70, flush=True)
    print("\n--- CLASSIFICATION REPORT ---", flush=True)
    print(report_str, flush=True)

    # 5. Save Outputs
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_path = os.path.join(args.output_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path)
    print(f"[+] Saved: {cm_csv_path}", flush=True)

    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = os.path.join(args.output_dir, "classification_report.csv")
    report_df.to_csv(report_csv_path)
    print(f"[+] Saved: {report_csv_path}", flush=True)

    # Save Text Report
    text_report_path = os.path.join(args.output_dir, "classification_summary.txt")
    with open(text_report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Dataset: {args.data_dir}\n")
        f.write(f"Total Samples Evaluated: {len(all_labels)}\n")
        f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report_str + "\n\n")
        f.write("Raw Confusion Matrix:\n")
        f.write(cm_df.to_string() + "\n")
    print(f"[+] Saved: {text_report_path}", flush=True)

    # Plot & Save Heatmaps
    plot_confusion_matrices(cm, class_names, args.output_dir)

    print("-" * 70, flush=True)
    print(f"SUCCESS! All results and plots generated in:\n  -> {args.output_dir}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
