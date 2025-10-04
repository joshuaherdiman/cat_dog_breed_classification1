import streamlit as st
from fastai.vision.all import *
from requirements.txt import *

st.title("Breed Classifier")
st.text("Built By Joshua")

breed_classification_model = load_learner("breed-classification(1).pkl")

def extract_breed(file_name):
    p = Path(file_name)
    breed_name_parts = p.stems.split("_")

    final_breed_name = " "
    length_parts = len(breed_name_parts)-1
    for i in range(length_parts):
        final_breed_name += breed_name_parts[i]
        if i != length_parts:
            final_breed_name += "_"

    return final_breed_name

def predict(image):
    img = PILImage.creat(image)
    pred_class, pred_idx, outputs = breed_classification_model.predict(img)
    return pred_class

uploaded_file = st.file_uploader("Choose an image to upload...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    prediction = predict(uploaded_file)
    st.subheader(f"Predicted Breed: {prediction}")

st.text("Built with Streamlit and FastAI.")