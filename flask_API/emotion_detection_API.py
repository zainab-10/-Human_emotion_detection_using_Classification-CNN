from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import io

app = Flask(__name__)
model = load_model('my_model.h5')

# Define a list of class names
class_names = ['disgust', 'surprise']

def preprocess_input(image):
    # Convert the image to a 3-channel RGB image if it is grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Resize the image to the input shape of the model
    image = cv2.resize(image, (224, 224))
    # Convert the image to a numpy array and normalize the pixel values
    image_array = np.array(image) / 255.0
    # Add an extra dimension to the array to match the input shape of the model
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input image from the request
    image = request.files['image']
    # Convert the image to a NumPy array
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    # Preprocess the input image using the appropriate function
    input_data = preprocess_input(img)
    # Use the model to make predictions on the input data
    predictions = model.predict(input_data)
    # Get the predicted class label
    predicted_class = class_names[np.argmax(predictions)]
    # Return the predicted class label as a JSON object
    return jsonify({'class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
