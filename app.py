import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt


IMG_SIZE = 224
from sklearn.preprocessing import LabelBinarizer
import numpy as np

lb = LabelBinarizer()
lb.classes_ = np.array([
    "Crown and Root Rot",
    "Healthy Wheat",
    "Leaf Rust",
    "Wheat Loose Smut"
])



@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("wheat_model.h5")
    return model

model = load_model()
from tensorflow.keras.applications.vgg19 import preprocess_input

def preprocess_image(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)  
    image = np.expand_dims(image, axis=0)
    return image

st.title("🌾 Wheat Disease Detection (VGG19)")
st.write("Upload a wheat leaf image to predict disease")



uploaded_file = st.file_uploader("Upload wheat image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)


    # PREDICT BUTTON BLOCK GOES HERE
    if st.button("Predict"):
        processed_image = preprocess_image(image)  # your preprocessing function
        predictions = model.predict(processed_image)[0]

        predicted_class = lb.classes_[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        st.success(f"Prediction: **{predicted_class}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        # Plot probabilities
        fig, ax = plt.subplots()
        ax.barh(lb.classes_, predictions)
        ax.set_xlabel("Probability")
        ax.set_title("Class Probabilities")
        st.pyplot(fig)

