import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from library.models import *

# Hardware Constraints defined by the PIM array architecture
NUM_CYCLE = 50
NUM_ROWS = 96
NUM_COLS = 128

# 1. Load MNIST Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=NUM_CYCLE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transform), batch_size=NUM_CYCLE, shuffle=False)

# 2. Initialize Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = PIM_MNIST_MLP().to(device)
model = TILED_PIM_MNIST_MLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Validation to evaluate model on test set after each epoch
def validate(model, device, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    
    # turn off gradients during validation to save memory and speed up
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Sum up batch loss
            test_loss += criterion(output, target).item()
            
            # Get the index of the max log-probability (the prediction)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nValidation set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

# 3. Main Training Loop with Validation
num_epochs = 100
for epoch in range(1, num_epochs + 1):
    # Train for one epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 200 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # Validate after each epoch
    val_acc = validate(model, device, test_loader, criterion)

print("Done!")