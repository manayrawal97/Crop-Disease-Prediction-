import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

model = load_model('trained_model_mobilenet.h5')

CLASS_NAMES = ['Apple Scab', 'Apple black rot', 'Cedar apple rust', 'Healthy Apple', 'Healthy Blueberry', 'Cherry healthy', 'Cherry Powdery Mildew', 'Cercospora Leaf Spot'
             , 'Corn common rust', 'Healthy corn', 'Corn Northern leaf blight', 'Grape black rot', 'Grape Black measles', 'Healthy Grape', 'Grape leaf blight', 'Orange Huanglong bing'
             , 'Peach bacterial spot', 'Healthy Peach', 'Bell Pepper bacterial spot', 'Bell pepper healthy', 'Potato Early blight', 'Potato healthy blight', 'Potato late blight'
             , 'Healthy Raspberry', 'Healthy Soyabean', 'Squash powdery mildew', 'Healthy Strawberry', 'Strawberry leaf scotch', 'Tomato bacterial spot', 'Tomato early blight', 'Healthy Tomato'
             , 'Tomato late blight', 'Tomato leaf mold', 'Tomato septoria leaf spot', 'Tomato two spotted spider mite', 'Tomato target spot', 'Tomato mosiac virus', 'Tomato yellow leaf curl virus']

st.title("Crop Disease Prediction")
st.markdown("Upload an image of the Plant leaf")
dog_image = st.file_uploader("Choose an image...", type="png")
submit = st.button('Predict')
if submit:
    if dog_image is not None:
        
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        st.image(opencv_image, channels="BGR")
      
        opencv_image = cv2.resize(opencv_image, (224,224))
      
        opencv_image.shape = (1,224,224,3)
        
        Y_pred = model.predict(opencv_image)

        st.title(str("Result :-  "+CLASS_NAMES[np.argmax(Y_pred)]))

st.markdown("<div class='footer'><hr>This is a Group Project of IBM Industrial Training Created by<br>Rahul, Divyansh, Manay, Yashraj</div>", unsafe_allow_html=True)
