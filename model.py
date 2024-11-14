import torch
import torch.nn as nn

class MnistCNN(nn.Module):
    def __init__(self, kernel_sizes=[32, 64, 128, 256], dropout_rate=0.5):
        super(MnistCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv layer
            nn.Conv2d(1, kernel_sizes[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv layer
            nn.Conv2d(kernel_sizes[0], kernel_sizes[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv layer
            nn.Conv2d(kernel_sizes[1], kernel_sizes[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fourth conv layer
            nn.Conv2d(kernel_sizes[2], kernel_sizes[3], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(kernel_sizes[3], 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x 