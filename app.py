import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

st.title("Fish Freshness Classifier 🐟")
st.write("Upload or capture an image of a fish to check if it's **fresh** or **not fresh**.")

# --- User input option ---
option = st.radio("Choose input method:", ["Upload Image", "Use Camera"])

# Get image based on input type
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

elif option == "Use Camera":
    camera_image = st.camera_input("Take a picture of the fish")
    if camera_image is not None:
        image = Image.open(camera_image)

# If image is available, show and predict
if 'image' in locals():
    st.image(image, caption="Input Image", use_column_width=True)
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]

    label = "Fresh 🟢" if prediction < 0.5 else "Not Fresh 🔴"
    confidence = (1 - prediction) if prediction < 0.5 else prediction
    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"Confidence: **{confidence:.2%}**")
