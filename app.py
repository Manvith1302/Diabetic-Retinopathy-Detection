import zipfile
import os

if not os.path.exists("dr_model.h5"):
    with zipfile.ZipFile("dr_model.zip", "r") as zip_ref:
        zip_ref.extractall(".")

model = tf.keras.models.load_model("dr_model.h5")
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load Model
model = tf.keras.models.load_model("dr_model.h5")

# Class Names
classes = {
    0:"No DR",
    1:"Mild DR",
    2:"Moderate DR",
    3:"Severe DR",
    4:"Proliferative DR"
}

IMG_SIZE = 224

def preprocess_image(image):

    image = np.array(image)

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = image.astype(np.float32)

    image /= 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image

st.title(
    "AI-Based Diabetic Retinopathy Detection"
)

st.write(
    "Upload a retinal image and predict DR severity."
)

uploaded_file = st.file_uploader(
    "Upload Retinal Image",
    type=["png","jpg","jpeg"]
)

if uploaded_file:

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        caption="Uploaded Retinal Image",
        use_container_width=True
    )

    processed = preprocess_image(
        image
    )

    prediction = model.predict(
        processed,
        verbose=0
    )

    pred_class = np.argmax(
        prediction
    )

    st.success(
        f"Prediction: {classes[pred_class]}"
    )

    st.subheader(
        "Confidence Scores"
    )

    for i,score in enumerate(
        prediction[0]
    ):

        st.write(
            f"{classes[i]} : {score*100:.2f}%"
        )