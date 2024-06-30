import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained model
densemodel = pickle.load(open('densemodel.pkl', 'rb'))

# Function to preprocess images
def preprocess_image(img):
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read image file string data
        img = image.load_img(file, target_size=(224, 224))
        img_array = preprocess_image(img)

        # Predict using your model
        prediction = densemodel.predict(img_array)
        classification = 'Stroke' if prediction > 0.5 else 'Normal'

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
