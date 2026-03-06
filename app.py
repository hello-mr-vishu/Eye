import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Eye Diagnostic MVP", page_icon="👁️", layout="centered")

st.title("👁️ Non-Invasive Eye Disease Detector")
st.markdown("""
This smart system examines eye images to find early signs of:
* **Anemia:** Pale conjunctiva (inside of the lower eyelid).
* **Jaundice:** Yellowing of the sclera (the white part of the eye).
""")
st.divider()

# --- LOAD MODELS ---
# @st.cache_resource prevents the app from reloading the heavy models every time you click a button
@st.cache_resource
def load_models():
    anemia_model = tf.keras.models.load_model('anemia_model.h5')
    jaundice_model = tf.keras.models.load_model('jaundice_model.h5')
    return anemia_model, jaundice_model

try:
    anemia_model, jaundice_model = load_models()
    models_loaded = True
except Exception as e:
    st.error("⚠️ Models not found. Please ensure 'anemia_model.h5' and 'jaundice_model.h5' are in the same folder as this script.")
    models_loaded = False

if models_loaded:
    # --- USER INTERFACE ---
    disease_choice = st.radio(
        "Which condition would you like to screen for?",
        ("Anemia (Conjunctiva Image)", "Jaundice (Sclera Image)")
    )

    st.subheader("Choose Image Source")
    image_source = st.radio("How would you like to provide the image?", ("Upload an image", "Use a sample image"), horizontal=True)
    
    image = None

    if image_source == "Upload an image":
        uploaded_file = st.file_uploader("Upload an image of the eye (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
    else:
        if "Anemia" in disease_choice:
            sample_options = {
                "Sample 1": "Dataset_Anemia/dataset anemia/India/1/20200118_164733.jpg",
                "Sample 2": "Dataset_Anemia/dataset anemia/India/10/20200201_114324.jpg",
                "Sample 3": "Dataset_Anemia/dataset anemia/India/11/20200203_094523.jpg"
            }
        else:
            sample_options = {
                "Sample 1": "Dataset_Jaundice/train/images/-jaundice-medicalshort_mp4-0000_jpg.rf.22f14a3a8913a83cfd9de327f949beb4.jpg",
                "Sample 2": "Dataset_Jaundice/train/images/-jaundice-medicalshort_mp4-0004_jpg.rf.2cf303cd27dec507e5d5612bff0fed72.jpg",
                "Sample 3": "Dataset_Jaundice/train/images/-jaundice-medicalshort_mp4-0005_jpg.rf.92c2f6329f19b44c8787b59aa852fe65.jpg"
            }
            
        selected_sample = st.selectbox("Select a sample image", list(sample_options.keys()))
        try:
            image = Image.open(sample_options[selected_sample]).convert('RGB')
        except Exception as e:
            st.error(f"Could not load sample image: {e}")

    if image is not None:
        # Display the uploaded/selected image
        st.image(image, caption="Eye Image for Analysis", width=350)
        
        if st.button("Run AI Diagnostic", type="primary"):
            with st.spinner('Analyzing image architecture...'):
                
                # --- PREPROCESSING ---
                # Resize and normalize to match our ResNet50 training pipeline
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized)
                # ResNet50 requires a specific preprocessing function, not just / 255.0
                img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0) # Add batch dimension: shape becomes (1, 224, 224, 3)
                
                # --- PREDICTION ---
                if "Anemia" in disease_choice:
                    prediction = anemia_model.predict(img_array)[0][0]
                    # Our training mapped Anemic to 1 (close to 1.0) and Normal to 0 (close to 0.0)
                    is_positive = prediction > 0.5
                    disease_name = "Anemia"
                else:
                    prediction = jaundice_model.predict(img_array)[0][0]
                    # Our training mapped Jaundice to 1 and Normal to 0
                    is_positive = prediction > 0.5
                    disease_name = "Jaundice"
                
                # Calculate confidence score percentage
                confidence = max(prediction, 1 - prediction) * 100
                
                # --- RESULTS ---
                st.divider()
                if is_positive:
                    st.error(f"🚨 **Diagnostic Alert:** The model detected signs of {disease_name}.")
                else:
                    st.success(f"✅ **Clear:** No visible signs of {disease_name} detected. The eye appears normal.")
                
                st.caption(f"AI Confidence Score: {confidence:.2f}%")
                st.caption("*Disclaimer: This is an MVP diagnostic tool for research purposes. Always consult a medical professional for actual health concerns.*")