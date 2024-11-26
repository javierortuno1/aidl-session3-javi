import torch
from torch.utils.data import DataLoader

from model import MyModel
from utils import binary_accuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(model, train_loader, optimizer):
    model.train()
    accs, losses = [], []
    for x, y in train_loader:
        # You will need to do y = y.unsqueeze(1).float() to add an output dimension to the labels and cast to the correct type
        x, y = x.to(device), y.to(device)
        y = y.unsqueeze(1).float()  # Add dimension and convert to float

        optimizer.zero_grad()
        output = model(x)
        loss = F.binary_cross_entropy_with_logits(output, y)
        acc = binary_accuracy(y.cpu(), output.cpu())
        
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accs.append(acc.item())
    return np.mean(losses), np.mean(accs)


def eval_single_epoch(model, val_loader):
    accs, losses = [], []
    with torch.no_grad():
        model.eval()
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y = y.unsqueeze(1).float()
            
            output = model(x)
            loss = F.binary_cross_entropy_with_logits(output, y)
            acc = binary_accuracy(y.cpu(), output.cpu())

            losses.append(loss.item())
            accs.append(acc.item())
    return np.mean(losses), np.mean(accs)


def train_model(config):

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to same size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    # Load datasets
    train_dataset = ImageFolder(
        root='kaggel-dataset/dataset/cars_vs_flowers/training_set',
        transform=data_transforms
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    test_dataset = ImageFolder(
        root='kaggel-dataset/dataset/cars_vs_flowers/test_set',
        transform=data_transforms
    )
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    my_model = MyModel().to(device)

    optimizer = optim.Adam(my_model.parameters(), config["lr"])
    best_acc = 0
    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_single_epoch(my_model, train_loader, optimizer)
        print(f"Train Epoch {epoch} loss={train_loss:.4f} acc={train_acc:.4f}")
        
        val_loss, val_acc = eval_single_epoch(my_model, test_loader)
        print(f"Eval Epoch {epoch} loss={val_loss:.4f} acc={val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(my_model.state_dict(), 'best_model.pth')
    
    return my_model


if __name__ == "__main__":

    config = {
        "lr": 1e-3,
        "batch_size": 32,  # Reduced batch size to handle memory better
        "epochs": 5,      # Increased epochs for better training
    }
    my_model = train_model(config)

    
