import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.dataset import get_dataloaders
from src.models import get_model
from src.train import train_one_epoch, validate  

def main():
    DATA_DIR = "./data"         
    MODEL_NAME = "resnet18"      
    OPTIMIZER_TYPE = "adamw"     
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 6
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    train_loader, val_loader, class_names = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    model = get_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    if OPTIMIZER_TYPE == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    elif OPTIMIZER_TYPE == "adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    elif OPTIMIZER_TYPE == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    else:
        raise ValueError(f"Optimizer not supported: {OPTIMIZER_TYPE}")

    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    writer = SummaryWriter(log_dir=f"./logs/{MODEL_NAME}_{OPTIMIZER_TYPE}")

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        epoch_duration = time.time() - start_time

        writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch + 1)
        writer.add_scalars('Accuracy', {'Train': train_acc, 'Val': val_acc}, epoch + 1)

        print(f"Epoch [{epoch+1}/{EPOCHS}] ({epoch_duration:.1f}s) | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = f"./checkpoints/{MODEL_NAME}_{OPTIMIZER_TYPE}_best.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names
            }, checkpoint_path)

    writer.close()
    print(f"\nHighest validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()