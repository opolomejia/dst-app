import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
import joblib
import warnings
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
directory = "/mount/src/dst-app/src/streamlit/"
#directory = "./"

def load_cv_model():
    model_path = directory+"models/transfert_learn_inception_v3_final.h5"
    model = tf.keras.models.load_model(model_path, 
           custom_objects={'preprocess_input': preprocess_input})
    # Recompile the model to avoid the metrics warning
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy')
    return model

def load_text_model():
    model_path = directory+"models/logistic_regression_model.joblib"
    return joblib.load(model_path)

def demo_interface():
    """
    Streamlit interface for image classification demo.
    """
    st.title("Classification de Documents")
    uploaded_file = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])
    global cv_model
    global text_model
    global document_type
    if uploaded_file is not None:
        img = tf.keras.preprocessing.image.load_img(
            uploaded_file,
            target_size=(300, 226),
            color_mode="grayscale",
            interpolation='bilinear'
        )
        st.image(img, caption="Uploaded Image", use_container_width=False)
        cv_predict, text_predict = st.columns(2)
        
        if st.button("Classifier"):
            with cv_predict:
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)
                st.subheader("Prédictions avec Computer Vision")
                predictions = cv_model.predict(img_array)
                top_3_indices = np.argsort(predictions[0])[::-1][:3]
                top_3_probs = predictions[0][top_3_indices]
                for idx, prob in zip(top_3_indices, top_3_probs):
                    st.markdown(
                        f"""
                        <div style="line-height: 1.2;">
                            <strong>Type de Document</strong>: {document_type[str(int(idx))]}
                            <br>
                            <strong>Probabilité</strong>: {prob:.4f}
                            <br><br>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

            with text_predict:
                st.subheader("Prédictions avec Text Mining")
                #ICI ON DOIT LANCER LA RECUPERATION DU TEXTE (OCR) 
                #TRAITEMENT DU TEXTE
                #PUIS LANCER LA PRÉDICTION AVEC LE MODELE 'text_model' DEJA CHARGÉ
                #UNE FOIS LA PREDICTION FAITE, ON AFFICHE LES RÉSULTAT
                #UTILISER "img"comme INPUT POUR L'EXTRACTION DU TEXTE


cv_model = load_cv_model()
text_model = load_text_model()

document_type = {
        "0": "letter",
        "1": "form",
        "2": "email",
        "3": "handwritten",
        "4": "advertisement",
        "5": "scientific report",
        "6": "scientific publication",
        "7": "specification",
        "8": "file folder",
        "9": "news article",
        "10": "budget",
        "11": "invoice",
        "12": "presentation",
        "13": "questionnaire",
        "14": "resume",
        "15": "memo"
    }