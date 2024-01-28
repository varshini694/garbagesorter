import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

import os

model_path = "/Users/varshini/desktop/gd/prediction_model/keras_model.h5"

if os.path.exists(model_path):
    print(f"Model file found at: {os.path.abspath(model_path)}")
    model = tf.keras.models.load_model(model_path)
else:
    print(f"Model file not found at: {os.path.abspath(model_path)}")

### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

map_dict = {0: 'PLASTIC',
            1: 'BIO-WASTE',
            2: 'E-WASTE',
            3: 'GLASS',
            4: 'METAL',
            5: 'TRASH',
           }


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))