""" Load the Saved Model"""

import os

# Disabled debugging logs using os.environ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np


new_model = load_model('E:/FYP 2021/Final Project/saved_models/model_1.h5')

# new_model.summary()

# new_model.get_weights()

# provider any image path to load_image to make prediction

""" Making Prediction on Test data """
print("\nMaking Prediction")
test_image = image.load_img("E:/Machine Learning Series/Datasets/archive/real_vs_fake/real-vs-fake/test/fake"
                            "/0EDS0OZ1XY.jpg", target_size=(150, 150, 3))

class_name = ['fake', 'real']

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = new_model.predict(test_image)

print("This image {} with {:.2f} percentage. ".format(class_name[np.argmax(result)], 100 * np.max(result)))
