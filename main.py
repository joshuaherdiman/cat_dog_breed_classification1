import streamlit as st
from fastai.vision.all import *
from requirements.txt import *

st.title("Breed Classifier")
st.text("Built By Joshua")

breed_classification_model = load_learner("breed-classification(1).pkl")

def predict(image):
    img = PILImage.create(image)
    outputs = breed_classification_model.predict(img)
    breed_name = outputs[1].item()
    return(breed_name)


uploaded_file = st.file_uploader("Choose an image to upload...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    prediction = predict(uploaded_file)
    st.write(prediction)