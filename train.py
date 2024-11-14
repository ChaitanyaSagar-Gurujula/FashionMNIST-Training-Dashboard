import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import random
from model import MnistCNN
from tqdm import tqdm
import time
import io

class ModelTrainer:
    def __init__(self, config, model_name="Model"):
        self.config = config
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = {
            'loss': [],
            'accuracy': [],
            'epoch': [],
            'batch': []
        }
        self.model = None
        print(f"Initialized {self.model_name} with config:", config)
        
    def get_optimizer(self, model_parameters):
        if self.config['optimizer'].lower() == 'adam':
            return optim.Adam(model_parameters, lr=float(self.config['learning_rate']))
        elif self.config['optimizer'].lower() == 'sgd':
            return optim.SGD(model_parameters, lr=float(self.config['learning_rate']), momentum=0.9)
        elif self.config['optimizer'].lower() == 'rmsprop':
            return optim.RMSprop(model_parameters, lr=float(self.config['learning_rate']))
        
    def train(self):
        print(f"\nStarting training for {self.model_name}")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=int(self.config['batch_size']), 
            shuffle=True
        )

        total_batches = len(train_loader)
        print(f"{self.model_name} - Total batches per epoch: {total_batches}")

        # Model initialization
        kernel_sizes = [
            int(self.config['kernels1']),
            int(self.config['kernels2']),
            int(self.config['kernels3']),
            int(self.config['kernels4'])
        ]
        
        self.model = MnistCNN(
            kernel_sizes=kernel_sizes,
            dropout_rate=float(self.config['dropout'])
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = self.get_optimizer(self.model.parameters())

        num_epochs = 2
        
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(enumerate(train_loader), 
                       total=total_batches,
                       desc=f'{self.model_name} Epoch {epoch+1}/{num_epochs}')
            
            for batch_idx, (data, target) in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # Update metrics every 20 batches
                if (batch_idx + 1) % 20 == 0:
                    accuracy = 100. * correct / total
                    avg_loss = running_loss / 20
                    
                    self.history['loss'].append(avg_loss)
                    self.history['accuracy'].append(accuracy)
                    self.history['epoch'].append(epoch + 1)
                    self.history['batch'].append(batch_idx + 1)
                    
                    running_loss = 0.0
                    correct = 0
                    total = 0

        print(f"\n{self.model_name} training completed!")
        print(f"Final history lengths - Loss: {len(self.history['loss'])}, "
              f"Accuracy: {len(self.history['accuracy'])}, "
              f"Epochs: {len(self.history['epoch'])}")

        return self.model

    def generate_training_plot(self):
        """Generate training plot and return as base64 string"""
        try:
            if not self.history['loss']:  # Check if there's data to plot
                return None
            
            plt.figure(figsize=(12, 5))
            
            # Create figure with secondary y-axis
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Plot loss on primary y-axis
            color = 'tab:blue'
            ax1.set_xlabel('Steps (100 batches)')
            ax1.set_ylabel('Loss', color=color)
            ax1.plot(self.history['loss'], color=color, label='Loss')
            ax1.tick_params(axis='y', labelcolor=color)
            
            # Create secondary y-axis and plot accuracy
            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.set_ylabel('Accuracy (%)', color=color)
            ax2.plot(self.history['accuracy'], color=color, label='Accuracy')
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Add title
            plt.title(f'{self.model_name} Training Progress')
            
            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            # Adjust layout and convert to base64
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            plt.close('all')  # Close all figures to free memory
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()
            
            return image_base64
        except Exception as e:
            print(f"Error generating plot: {str(e)}")
            return None

    def evaluate_samples(self, model, n_samples=10):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)
        indices = random.sample(range(len(test_dataset)), n_samples)
        
        images = []
        predictions = []
        labels = []
        
        model.eval()
        for idx in indices:
            image, label = test_dataset[idx]
            image = image.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = model(image)
                pred = output.argmax(dim=1).item()
            
            images.append(image.cpu().squeeze().numpy())
            predictions.append(self.class_names[pred])  # Use class names instead of numbers
            labels.append(self.class_names[label])      # Use class names instead of numbers
            
        return images, predictions, labels