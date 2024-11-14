from flask import Flask, render_template, jsonify
import threading
from train import train_model, training_history, evaluate_random_samples
from model import MnistCNN
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
from io import BytesIO  # Add this import

app = Flask(__name__)
training_complete = False
model = None

@app.route('/')
def home():
    return render_template('monitor.html')

@app.route('/get_training_data')
def get_training_data():
    return jsonify(training_history)

@app.route('/training_status')
def training_status():
    return jsonify({'complete': training_complete})

@app.route('/results')
def get_results():
    if training_complete and model is not None:
        # Get the data in the background thread
        images, predictions, labels = evaluate_random_samples(model)
        # Create the plot in the main thread
        results_image = create_results_plot(images, predictions, labels)
        return jsonify({'image': results_image})
    return jsonify({'image': None})

def start_training():
    global training_complete, model
    model = train_model()
    training_complete = True

def create_results_plot(images, predictions, labels):
    """Create the plot in the main thread"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for idx in range(len(images)):
        axes[idx].imshow(images[idx], cmap='gray')
        axes[idx].axis('off')
        axes[idx].set_title(f'True: {labels[idx]}\nPred: {predictions[idx]}')
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()  # Close the figure to free memory
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode()
    
    return graphic

if __name__ == '__main__':
    training_thread = threading.Thread(target=start_training)
    training_thread.start()
    app.run(debug=False, port=5000) 