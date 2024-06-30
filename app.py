import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os
import io
app = Flask(__name__)

# Load your trained model
model_path = 'densemodel.h5'  # Adjusted to .h5 format
model = tf.keras.models.load_model(model_path)


@app.route('/')
def home():
    return render_template('home.html')

def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image to match model's expected sizing
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

@app.route('/predict_api', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read image file data
        img = image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
        img_array = preprocess_image(img)

        # Predict using your model
        prediction = model.predict(img_array)
        classification = 'Stroke' if prediction < 0.5 else 'Normal'

        # Format prediction response
        result = {
            'prediction': float(prediction[0][0]),
            'classification': classification
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True)
