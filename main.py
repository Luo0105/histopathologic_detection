import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os

# Import our custom modules
from src.data_setup import create_dataloaders
from src.train import train_one_epoch, validate
from src.model import get_model

def main():
    # --- Configuration ---
    DATA_PATH = r'E:\data\histopathologic' # ‼️
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10
    MODEL_SAVE_PATH = r'E:\data\histopathologic\models' # Folder to save model weights
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directory for saving weights if it doesn't exist
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # --- Data Preparation ---
    train_loader, val_loader = create_dataloaders(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
        num_workers=4  # You can use more workers in a .py script
    )
    
    # --- Model Definition ---
    print("Loading pre-trained ResNet18 model...")
    model = get_model(model_name='resnet18').to(device) 

    # --- Loss and Optimizer ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- Training Loop ---
    best_val_acc = 0.0
    print(f"🚀 Starting training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(MODEL_SAVE_PATH, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  🚀 New best model saved to {save_path} with Val Acc: {best_val_acc:.4f}")
        print("-" * 40)
            
if __name__ == '__main__':
    main()
