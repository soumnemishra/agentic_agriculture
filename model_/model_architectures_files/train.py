import os
import copy
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import config
from dataset import get_dataloaders, get_class_weights
from model import get_model

def train():
    print("=" * 60)
    print("STARTING TRAINING (CPU COMPATIBLE)")
    print("=" * 60)

    # Load Data
    dataloaders, image_datasets = get_dataloaders()
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    print(f"Training images: {dataset_sizes['train']}")
    print(f"Validation images: {dataset_sizes['valid']}")

    # Load Model
    model = get_model()
    
    # Compute Class Weights
    class_weights_tensor = get_class_weights(image_datasets['train'])
    print(f"Class Weights initialized: {class_weights_tensor.tolist()}")

    # Optimizer, Loss, Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        print("-" * 20)

        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (inputs, labels) in enumerate(dataloaders['train']):
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()

            # Safely removed CUDA-specific Mixed Precision (AMP) for CPU compatibility
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(dataloaders['train'])} - Loss: {loss.item():.4f}")

        epoch_train_loss = train_loss / dataset_sizes['train']
        epoch_train_acc = correct_train / total_train

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

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

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), checkpoint_path)
            print(f"--> Saved new best model to {checkpoint_path}")

    print("\nTraining complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    train()
