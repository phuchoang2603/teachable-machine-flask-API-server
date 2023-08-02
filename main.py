from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import requests 
import tensorflow as tf 

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def read_tensor_from_image_url(url,
                               input_height=224,
                               input_width=224,
                               input_mean=0,
                               input_std=255):
    image_reader = tf.image.decode_jpeg(
        requests.get(url).content, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize(dims_expander,[input_height,input_width], method='bilinear',antialias=True, name = None)
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

    return normalized

#decode and resize the image from the URL
image = read_tensor_from_image_url(r'Cloudinary URL goes here!')

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)
