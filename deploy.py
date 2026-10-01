import os

from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS module

app = Flask(__name__)
CORS(app)  # Apply CORS to your Flask app

import tensorflow as tf
import numpy as np
from PIL import Image

from keras.layers import TFSMLayer

MODEL_PATH = os.environ.get(
    "WOUND_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"),
)

model = TFSMLayer(MODEL_PATH, call_endpoint='serving_default')

# Load the TensorFlow model
#model = tf.keras.models.load_model('~/Downloads/temp')

@app.route('/predict', methods=['POST'])
def predict():
    # Receive the image from the frontend
    print(request)
    image = request.files['image']

    # Open the image and resize it to 224x224
    img = Image.open(image)
    img = img.resize((224, 224))

    # Convert the image to numpy array
    image_array = np.array(img)

    # Normalize the image
    #image_array = image_array / 255.0
    print("before Prediction")
    #return jsonify({'prediction': "asda"})

    # Perform prediction
    prediction = model(np.expand_dims(image_array, axis=0))['dense_5']

    # You can post-process the prediction if necessary
    
    print(prediction)
    # Send the prediction back to the frontend
    print(np.argmax(prediction, axis = 1)) #, prediction.tolist())
    class_names = {0: 'Background', 1: 'Diabetic', 2: 'Nerves', 3: 'Pressure', 4: 'Surgical', 5: 'Venous' }
    return jsonify({'prediction': float(prediction[0][int(np.argmax(prediction, axis = 1))]), 'class': class_names[int(np.argmax(prediction, axis = 1))]})

if __name__ == '__main__':
    app.run(port = 8888, debug=True)

