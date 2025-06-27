import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
import joblib
import easyocr
import warnings
import os
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer


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
    global easyocr_reader
    global tfidf_transformer

    if uploaded_file is not None:
        # Open the uploaded image
        pil_image = Image.open(uploaded_file)
        
        # Reduce the size of the image to half
        width, height = pil_image.size
        resized_image = pil_image.resize((width // 3, height // 3))
        
        # Display the resized image
        st.image(resized_image, caption="Uploaded Image", use_container_width=False)

        cv_predict, text_predict = st.columns(2)
        if st.button("Classer"):
            with cv_predict:
                img = tf.keras.preprocessing.image.load_img(
                uploaded_file,
                target_size=(300, 226),
                color_mode="grayscale",
                interpolation='bilinear')
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
                
                # Convert uploaded file to format EasyOCR can handle
                pil_image = Image.open(uploaded_file)
                img_array = np.array(pil_image)
                
                # Extract text using EasyOCR
                extracted_text = easyocr_reader.readtext(img_array, detail=0)
                # Convert extracted text to lowercase
                cleaned_text = ' '.join(extracted_text).lower() if isinstance(extracted_text, list) else str(extracted_text).lower()
                
                # Remove punctuation from the cleaned text
                cleaned_text = cleaned_text.translate(str.maketrans('', '', string.punctuation))

                # Remove extra spaces
                cleaned_text = ' '.join(cleaned_text.split())

                #Supprimer les caractères spéciaux
                cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
                
                # Tokenize the cleaned text
                tokenized_text = word_tokenize(cleaned_text)

                # Initialiser la variable des mots vides
                stop_words = set(stopwords.words('english'))
                #Ajouter des mots vides
                add_stopwords = ['*']  # List for multiple words
                stop_words.update(add_stopwords)  # Add the whole list at once

                # Remove stop words from the tokenized text
                tokenized_text = [word for word in tokenized_text if word not in stop_words]

                #LEMMATISATION
                wordnet_lemmatizer = WordNetLemmatizer()
                lemmatized_text = [wordnet_lemmatizer.lemmatize(word) for word in tokenized_text]
                # Join the lemmatized words back into a single string
                lemmatized_text = ' '.join(lemmatized_text)

                # Transform the lemmatized text using the TF-IDF vectorizer
                tfidf_features = tfidf_transformer.transform([lemmatized_text])
  
                # Make predictions using the text model
                text_prediction = text_model.predict(tfidf_features)
                text_probabilities = text_model.predict_proba(tfidf_features)
                # Get the indices of the top 3 probabilities

                top_3_indices = np.argsort(text_probabilities[0])[::-1][:3]
                top_3_probs = text_probabilities[0][top_3_indices]

                for idx, prob in zip(top_3_indices, top_3_probs):
                    st.markdown(
                        f"""
                        <div style="line-height: 1.2;">
                            <strong>Type de Document</strong>: {document_type[str(idx)]}
                            <br>
                            <strong>Probabilité</strong>: {prob:.4f}
                            <br><br>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )


cv_model = load_cv_model()
text_model = load_text_model()

#we instantiate the easyocr reader
easyocr_reader = easyocr.Reader(['en'], gpu=True)

# Load the trained TF-IDF vectorizer
tfidf_path = directory + "models/trained_tfidf.joblib"
tfidf_transformer = joblib.load(tfidf_path)

#The numerical labels where sorted in alphabetical order (as strings)
#so we need to map them to the original labels
document_type = {
        "0": "Letter",
        "1": "Form",
        "2": "Budget",
        "3": "Invoice",
        "4": "Presentation",
        "5": "Questionnaire",
        "6": "Resume",
        "7": "Memo",
        "8": "Email",
        "9": "Handwritten",
        "10": "Advertisement",
        "11": "Scientific report",
        "12": "Scientific publication",
        "13": "Specification",
        "14": "File folder",
        "15": "News article"
    }