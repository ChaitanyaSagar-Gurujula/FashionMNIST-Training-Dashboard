from flask import Flask, render_template, jsonify
import threading
from train import train_model, training_history, evaluate_random_samples
from model import MnistCNN

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
        results_image = evaluate_random_samples(model)
        return jsonify({'image': results_image})
    return jsonify({'image': None})

def start_training():
    global training_complete, model
    model = train_model()
    training_complete = True

if __name__ == '__main__':
    training_thread = threading.Thread(target=start_training)
    training_thread.start()
    app.run(debug=False, port=5000) 