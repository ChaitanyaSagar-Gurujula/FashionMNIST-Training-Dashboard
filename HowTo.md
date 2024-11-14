# FashionMNIST CNN Training Monitor

This project implements a real-time training visualization system for comparing two CNN models on the FashionMNIST dataset. It features an interactive web interface for configuring and monitoring the training progress of both models simultaneously.

## Requirements

- Python 3.7+
- PyTorch
- Flask
- Plotly.js
- torchvision
- tqdm
- matplotlib

## Installation

1. Create a virtual environment (optional but recommended): 

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

2. Install required packages:

```bash
pip install torch torchvision flask matplotlib tqdm
```


## Running the Application

1. Start the Flask server:

```bash
python server.py
```

2. Open your web browser and navigate to:

```
http://localhost:8080
```

3. Configure both models using the web interface:
   - Set kernel sizes for each convolutional layer
   - Choose optimizer and learning rate
   - Set dropout rate and batch size
   - Click "Start Training" to begin


## Features

- Real-time training visualization using Plotly.js
- Simultaneous training of two models with different configurations
- Interactive configuration of model parameters:
  - Number of kernels in each layer
  - Choice of optimizer (Adam, SGD, RMSprop)
  - Learning rate
  - Dropout rate
  - Batch size
- Live metrics display:
  - Training loss
  - Training accuracy
- Combined plot showing:
  - Loss curves for both models
  - Accuracy curves for both models
  - Fractional epoch progress
  - Detailed hover information

## Project Structure

```
fashion_mnist_monitor/
├── HowTo.md
├── train.py        # Training logic and ModelTrainer class
├── model.py        # CNN architecture definition
├── server.py       # Flask server and API endpoints
├── templates/      # HTML templates
│   └── monitor.html
└── static/         # Static files
    └── style.css
```


## Notes

- Training progress is saved in real-time
- The interface updates every 2 seconds
- Both models train simultaneously in separate threads
- The plot shows fractional epochs for precise progress tracking

## Troubleshooting

1. Port Issues:
   - If port 8080 is in use, modify the port number in `server.py`
   - Default port can be changed in the last line of `server.py`

2. CUDA/GPU Issues:
   - The code automatically detects and uses GPU if available
   - Falls back to CPU if CUDA is not available

3. Memory Issues:
   - Reduce batch size if encountering memory problems
   - Adjust kernel sizes in the model configuration


## Model Comparison

The application is designed to compare two CNN models with different architectures and hyperparameters in real-time:

### Comparison Features
- Side-by-side training visualization
- Combined plot showing both models' performance
- Different metrics for comparison:
  - Training Loss
  - Training Accuracy
  - Training Speed (batches/second)

### What to Compare
1. Architecture Differences:
   - Effect of different kernel sizes
   - Impact of layer configurations
   - Influence of dropout rates

2. Optimization Strategies:
   - Optimizer performance (Adam vs SGD vs RMSprop)
   - Learning rate effects
   - Batch size impact

3. Performance Metrics:
   - Convergence speed
   - Final accuracy
   - Loss reduction rate
   - Training stability

### Example Comparisons

1. Deep vs Shallow:
   ```
   Model 1: [32, 64, 128, 256] kernels
   Model 2: [16, 32, 64, 128] kernels
   ```

2. Optimizer Impact:
   ```
   Model 1: Adam (lr=0.001)
   Model 2: SGD (lr=0.01)
   ```

3. Regularization Effect:
   ```
   Model 1: Dropout=0.5
   Model 2: Dropout=0.3
   ```

### Interpreting Results

1. Loss Curves:
   - Lower curves indicate better model fit
   - Smoother curves suggest stable training
   - Sharp drops may indicate learning rate issues

2. Accuracy Curves:
   - Higher curves show better performance
   - Plateaus suggest learning saturation
   - Large fluctuations might indicate instability

3. Training Speed:
   - Larger models typically train slower
   - Batch size affects memory usage and speed
   - GPU utilization varies with model size

### Best Practices for Comparison

1. Controlled Variables:
   - Change only one parameter at a time
   - Keep dataset and preprocessing consistent
   - Use same number of epochs

2. Fair Evaluation:
   - Consider both final performance and training time
   - Look for stability in training
   - Compare resource usage (memory, GPU)

3. Common Patterns:
   - Larger models may overfit on simple tasks
   - Higher learning rates need more stability
   - Dropout affects training speed

For any additional questions or issues, please refer to the source code comments or create an issue in the project repository.