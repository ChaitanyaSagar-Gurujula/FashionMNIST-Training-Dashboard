from flask import Flask, render_template, jsonify, request
import threading
from train import ModelTrainer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Global variables
model1_trainer = None
model2_trainer = None
model1_complete = False
model2_complete = False

@app.route('/')
def home():
    return render_template('monitor.html')

@app.route('/start_training', methods=['POST'])
def start_training():
    global model1_trainer, model2_trainer
    try:
        config = request.json
        print("Received configurations:", config)  # Debug print
        
        model1_trainer = ModelTrainer(config['model1'], "Model 1")
        model2_trainer = ModelTrainer(config['model2'], "Model 2")
        
        # Start training threads
        threading.Thread(target=train_model1).start()
        threading.Thread(target=train_model2).start()
        
        return jsonify({'status': 'success', 'message': 'Training started'})
    except Exception as e:
        print(f"Error starting training: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def train_model1():
    global model1_complete, model1_trainer
    try:
        model1_trainer.train()
        model1_complete = True
    except Exception as e:
        print(f"Error in model1 training: {str(e)}")

def train_model2():
    global model2_complete, model2_trainer
    try:
        model2_trainer.train()
        model2_complete = True
    except Exception as e:
        print(f"Error in model2 training: {str(e)}")

@app.route('/get_training_data')
def get_training_data():
    try:
        if model1_trainer and model2_trainer:
            data = {
                'model1': {
                    'loss': model1_trainer.history['loss'],
                    'accuracy': model1_trainer.history['accuracy'],
                    'epoch': model1_trainer.history['epoch'],
                    'batch': model1_trainer.history['batch']
                },
                'model2': {
                    'loss': model2_trainer.history['loss'],
                    'accuracy': model2_trainer.history['accuracy'],
                    'epoch': model2_trainer.history['epoch'],
                    'batch': model2_trainer.history['batch']
                }
            }
            print("\nSending training data:")  # Debug print
            print(f"Model 1 data points: {len(data['model1']['loss'])}")
            print(f"Model 2 data points: {len(data['model2']['loss'])}")
            return jsonify(data)
    except Exception as e:
        print(f"Error in get_training_data: {str(e)}")
        return jsonify({'error': str(e)})

    return jsonify({'model1': None, 'model2': None})

@app.route('/training_status')
def training_status():
    return jsonify({
        'model1_complete': model1_complete,
        'model2_complete': model2_complete
    })

@app.route('/results')
def get_results():
    if model1_complete and model2_complete and model1_trainer and model2_trainer:
        try:
            # Get results for both models
            images1, preds1, labels1 = model1_trainer.evaluate_samples(model1_trainer.model)
            images2, preds2, labels2 = model2_trainer.evaluate_samples(model2_trainer.model)
            
            # Create plots
            plot1 = create_results_plot(images1, preds1, labels1, "Model 1")
            plot2 = create_results_plot(images2, preds2, labels2, "Model 2")
            
            return jsonify({
                'image1': plot1,
                'image2': plot2
            })
        except Exception as e:
            print(f"Error generating results: {str(e)}")
            return jsonify({'error': str(e)})
    return jsonify({'image1': None, 'image2': None})

def create_results_plot(images, predictions, labels, title):
    plt.figure(figsize=(15, 6))
    for idx in range(len(images)):
        plt.subplot(2, 5, idx + 1)
        plt.imshow(images[idx], cmap='gray')
        plt.axis('off')
        plt.title(f'True: {labels[idx]}\nPred: {predictions[idx]}')
    
    plt.suptitle(f'{title} Test Results')
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    return base64.b64encode(image_png).decode()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False) 