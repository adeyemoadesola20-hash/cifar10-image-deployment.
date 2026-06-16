import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image

st.set_page_config(page_title="CIFAR-10 Image Classifier", layout="centered")

st.title("📸 Real-Time CIFAR-10 Image Classification")
st.write("Upload an image to see live predictions directly from your trained CNN model.")

@st.cache_resource
def load_onnx_model():
    return ort.InferenceSession("cifar10_model.onnx")

try:
    session = load_onnx_model()
    input_name = session.get_inputs()[0].name
    st.success("Real AI Model weights loaded successfully into production!")
except Exception as e:
    st.error(f"Error loading real model weights: {e}")

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("🤖 Processing image array matrices...")
    
    img_resized = image.resize((32, 32))
    img_array = np.array(img_resized)
    
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    img_array = img_array.astype('float32') / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)
    
    # Run REAL mathematical inference on your weights
    raw_outputs = session.run(None, {input_name: img_tensor})
    probabilities = raw_outputs[0][0]
    
    predicted_index = np.argmax(probabilities)
    predicted_class = class_names[predicted_index]
    confidence = (np.exp(probabilities) / np.sum(np.exp(probabilities)))[predicted_index] * 100

    st.subheader(f"Prediction Output: **{predicted_class}**")
    st.write(f"Model Inference Confidence: **{confidence:.2f}%**")
