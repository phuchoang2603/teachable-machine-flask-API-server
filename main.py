import os
from flask import Flask
from flask import request

import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import requests
import tensorflow as tf

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load Flask app
app = Flask(__name__)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')
class_names = open("labels.txt", "r").readlines()

@app.route('/', methods=['POST'])
def prediction():
    apikey = request.args.get('apikey')
    url = request.args.get('url')

    if apikey == "1234":
        image_url = tf.keras.utils.get_file('image.jpg', origin =url, cache_dir="~/images")
        # os.remove(image_url)
        # image_url = tf.keras.utils.get_file('image.jpg', url)
        
        image = tf.keras.preprocessing.image.load_img(image_url, target_size=(224, 224))
        os.remove(image_url)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index][2:].replace("\n", "")
        confidence_score = float(prediction[0][index])

        return {
            "class_name": class_name,
            "confidence_score": confidence_score
        }
    
    else:
        return {
            "error": "Invalid API Key"
        }
    

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))