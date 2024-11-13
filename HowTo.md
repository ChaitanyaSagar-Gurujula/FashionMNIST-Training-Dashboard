# MNIST CNN Training Monitor

This project implements a 4-layer CNN for MNIST digit classification with real-time training visualization.

## Requirements

- Python 3.7+
- PyTorch
- Flask
- Matplotlib
- torchvision

## Installation

1. Create a virtual environment (optional but recommended): 

bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate


2. Install required packages:

bash
pip install -r requirements.txt


## Running the Application

1. Make sure all files are in their correct directories as shown in the project structure.

2. Start the application:

bash
python server.py

3. Open your web browser and navigate to:

http://localhost:5000


## Features

- Real-time training loss and accuracy visualization
- Automatic GPU/CPU detection and utilization
- Results visualization on 10 random test images after training
- Interactive web interface

## Project Structure

mnist_cnn/
├── HowTo.md
├── train.py # Training logic and evaluation
├── model.py # CNN model definition
├── server.py # Flask server
├── templates/ # HTML templates
│ └── monitor.html
└── static/ # Static files
└── style.css


## Notes

- Training progress is automatically visualized in real-time
- The model will use CUDA if available, otherwise CPU
- Training takes approximately 10-15 minutes depending on your hardware
- Results will be displayed automatically once training is complete

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are correctly installed
2. Check if CUDA is properly set up if using GPU
3. Ensure all files are in the correct directory structure
4. Check the console for any error messages

## Additional Information

- The CNN architecture consists of 4 convolutional layers followed by fully connected layers
- Each convolutional layer is followed by ReLU activation and max pooling
- The model uses dropout for regularization
- Training parameters:
  - Batch size: 64
  - Learning rate: 0.001
  - Optimizer: Adam
  - Number of epochs: 10

For any additional questions or issues, please refer to the source code comments or create an issue in the project repository.