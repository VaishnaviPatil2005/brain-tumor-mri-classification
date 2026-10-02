
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("brain_tumor_mobilenetv2.h5")

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

st.title("Brain Tumor MRI Image Classification")

st.write("Upload a brain MRI image for classification.")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    img = img.resize((224, 224))

    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader("Prediction")
    st.write("Predicted Class:", predicted_class)
    st.write("Confidence:", f"{confidence:.2f}%")

st.info(
    "This application is for educational and research purposes only "
    "and should not be used for medical diagnosis."
)
