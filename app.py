import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="ASL Predictor", page_icon="")
st.title("ASL Hand Sign Predictor")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("asl_model.h5")

@st.cache_data
def get_classes():
    return sorted(os.listdir("asl_dataset"))
    
model = load_model()
classes = get_classes()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, width=300)
    
    if st.button("Predict"):
        img = image.resize((64, 64))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_normalized = img_array / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        predictions = model.predict(img_batch, verbose=0)
        predicted_class_index = np.argmax(predictions)
        predicted_class = classes[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index] * 100)
        
        st.success(f"Predicted Class: **{predicted_class.upper()}**")
        st.info(f"Confidence: {confidence:.2f}%")
