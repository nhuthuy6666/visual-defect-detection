import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.models import get_model

def evaluate_model(checkpoint_path, data_dir, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_names = checkpoint.get('class_names', ['crazing', 'iclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches'])
    num_classes = len(class_names)

    _, val_loader, _ = get_dataloaders(data_dir, batch_size=batch_size)

    file_name = os.path.basename(checkpoint_path).lower()
    if "resnet18" in file_name:
        model_name = "resnet18"
    elif "efficientnet_b0" in file_name:
        model_name = "efficientnet_b0"
    else:
        model_name = "custom_cnn"

    model = get_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    total_inference_time = 0.0
    total_images = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            batch_size_current = images.size(0)

            start_time = time.time()
            outputs = model(images)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            end_time = time.time()
            
            total_inference_time += (end_time - start_time)
            total_images += batch_size_current

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, 
        all_preds, 
        average='weighted', 
        zero_division=0
    )

    print(f"• Accuracy  : {accuracy * 100:.2f}%")
    print(f"• Precision : {precision:.4f}")
    print(f"• Recall    : {recall:.4f}")
    print(f"• F1-score  : {f1:.4f}")

    avg_latency_ms = (total_inference_time / total_images) * 1000.0
    fps = total_images / total_inference_time

    print(f"• Average Latency per Image: {avg_latency_ms:.2f} ms")
    print(f"• Frames Per Second (FPS): {fps:.2f}")

    os.makedirs("./logs", exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name.upper()}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    cm_path = f"./logs/confusion_matrix_{model_name}.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    CHECKPOINT_FILE = "./checkpoints/resnet18_adamw_best.pth"
    DATA_DIRECTORY = "./data"

    if os.path.exists(CHECKPOINT_FILE):
        evaluate_model(checkpoint_path=CHECKPOINT_FILE, data_dir=DATA_DIRECTORY)
    else:
        print(f"Checkpoint file not found: {CHECKPOINT_FILE}. Please ensure the file exists before running the evaluation.")

