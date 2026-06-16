import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set up page configurations
st.set_page_config(page_title="CIFAR-10 Image Classifier", layout="centered")

st.title("📸 CIFAR-10 Image Classification System")
st.write("Upload an image, and our trained Convolutional Neural Network (CNN) will classify it.")

# Load the saved model with caching so it only loads once
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('cifar10_model.h5')

try:
    model = load_my_model()
    st.success("AI Model loaded successfully into production!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define CIFAR-10 class categories
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# File uploader widget
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("🤖 Analyzing structural components...")
    
    # Preprocess the uploaded image to match model requirements (32x32 pixels)
    img_resized = image.resize((32, 32))
    img_array = np.array(img_resized)
    
    # Handle cases where images might have an alpha transparency channel (RGBA)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    # Normalize pixel data to match training scale [0, 1]
    img_array = img_array.astype('float32') / 255.0
    img_tensor = np.expand_dims(img_array, axis=0) # Add batch dimension
    
    # Execute Model Inference
    predictions = model.predict(img_tensor)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display Results
    st.subheader(f"Prediction: **{predicted_class}**")
    st.write(f"Model Confidence Score: **{confidence:.2f}%**")
