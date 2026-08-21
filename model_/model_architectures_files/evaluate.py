import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from sklearn.metrics import classification_report, confusion_matrix
import config
from model import get_model

def evaluate():
    print("=" * 60)
    print("EVALUATING ON HIDDEN TEST SET")
    print("=" * 60)

    # 1. Setup Test Data Transformations (same as validation)
    test_transforms = v2.Compose([
        v2.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load Test Dataset
    test_dataset = datasets.ImageFolder(config.TEST_DIR, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    class_names = test_dataset.classes
    print(f"Test images found: {len(test_dataset)}")
    print(f"Classes: {class_names}\n")

    # 3. Load Model and Weights
    model = get_model()
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Could not find model weights at {checkpoint_path}")
        return

    print(f"Loading weights from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    model.eval()

    # 4. Evaluation Loop
    all_preds = []
    all_labels = []
    
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
    
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    print("--- Confusion Matrix ---")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == '__main__':
    evaluate()
