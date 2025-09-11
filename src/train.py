import torch
from tqdm.auto import tqdm

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train() # Set the model to training mode
    
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc="Training", leave=True)
    
    for images, labels in progress_bar:
        # Move data to the selected device (GPU/CPU)
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        # 1. Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 2. Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # --- Calculate metrics ---
        running_loss += loss.item() * images.size(0)
        
        preds = torch.sigmoid(outputs) > 0.5
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
        progress_bar.set_postfix(loss=(running_loss / total_samples))

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Evaluates the model on the validation set."""
    model.eval() # Set the model to evaluation mode
    
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=True)
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            progress_bar.set_postfix(loss=(running_loss / total_samples))

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    
    return epoch_loss, epoch_acc