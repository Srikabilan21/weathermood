import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("weather_model.h5")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.title("Weather Mood Prediction from Sky Image")

uploaded_file = st.file_uploader("Upload Sky Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, width=500)

    img = np.array(image)
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    if len(faces) > 0:
        st.error("Please upload a sky image")

    else:
        img = cv2.resize(img,(64,64))
        img = img/255.0
        img = np.reshape(img,(1,64,64,3))

        prediction = model.predict(img)

        if prediction > 0.5:
            st.success("Prediction: HOT Weather ☀️")
        else:
            st.success("Prediction: COLD Weather ☁️")