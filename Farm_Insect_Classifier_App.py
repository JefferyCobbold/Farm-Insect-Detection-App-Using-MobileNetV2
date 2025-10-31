import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(page_title="Insect Detection Model", layout="centered")
st.title("Farm Insect Detection Using MobileNetV2")
st.write("Upload an image to identify the type of insect detected by the model.")

# -----------------------------
# Load Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_MV2.h5")
    return model

model = load_model()

# -----------------------------
# Class Names
# -----------------------------
class_names = [
    'Africanized Honey Bees (Killer Bees)',
    'Aphids',
    'Armyworms',
    'Brown Marorated Stink Bugs',
    'Cabbage Loopers',
    'Citrus Canker',
    'Colorado Potato Beetles',
    'Corn Bores',
    'Corn Earworms',
    'Fruit Files',
    'Spider Mites',
    'Thrips',
    'Tomato Hornworms',
    'Western Corn Rootworms'
]

# -----------------------------
# Create Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Upload Image", "Prediction Result", "Class Probabilities"])

# -----------------------------
# Upload Tab
# -----------------------------
with tab1:
    uploaded_file = st.file_uploader("Upload an insect image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess Image
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = np.max(predictions)

        # -----------------------------
        # Prediction Tab
        # -----------------------------
        with tab2:
            st.subheader("Model Prediction")
            st.success(f"**Detected Insect:** {predicted_class}")
            st.info(f"**Confidence:** {confidence:.2%}")

        # -----------------------------
        # Probability Tab
        # -----------------------------
        with tab3:
            st.subheader("Class Probabilities")
            sorted_indices = np.argsort(predictions[0])[::-1]
            for i in sorted_indices:
                st.write(f"**{class_names[i]}:** {predictions[0][i]:.2%}")
