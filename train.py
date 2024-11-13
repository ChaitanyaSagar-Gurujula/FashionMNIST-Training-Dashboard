import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
from model import MnistCNN
import random
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Global variables for training history
training_history = {
    'loss': [],
    'accuracy': []
}

def train_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model initialization
    model = MnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 99:
                accuracy = 100. * correct / total
                avg_loss = running_loss / 100
                training_history['loss'].append(avg_loss)
                training_history['accuracy'].append(accuracy)
                
                print(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Loss: {avg_loss:.3f}, Accuracy: {accuracy:.2f}%')
                running_loss = 0.0
                correct = 0
                total = 0

    # Save model
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    return model

def evaluate_random_samples(model, n_samples=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    indices = random.sample(range(len(test_dataset)), n_samples)
    
    model.eval()
    results = []
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for idx, sample_idx in enumerate(indices):
        image, label = test_dataset[sample_idx]
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            pred = output.argmax(dim=1).item()
        
        # Plot image
        axes[idx].imshow(image.cpu().squeeze(), cmap='gray')
        axes[idx].axis('off')
        axes[idx].set_title(f'True: {label}\nPred: {pred}')
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode()
    
    return graphic

if __name__ == '__main__':
    model = train_model()
    evaluate_random_samples(model) 