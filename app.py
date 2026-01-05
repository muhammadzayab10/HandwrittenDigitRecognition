import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_model.h5")

model = load_model()

st.title("✍️ Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0–9) and let AI predict it.")

# Image preprocessing
def preprocess_image(image):
    image = image.convert("L")        # Grayscale
    image = image.resize((28, 28))    # Resize to MNIST size
    image = np.array(image) / 255.0   # Normalize
    image = image.reshape(1, 784)     # FLATTEN
    return image


# Upload
uploaded_file = st.file_uploader(
    "Upload Handwritten Digit Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    if st.button("Predict Digit"):
        with st.spinner("Predicting..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_digit = np.argmax(prediction)

            st.success(f"✅ **Predicted Digit: {predicted_digit}**")
