import streamlit as st

def home():
    st.title("Projet: Classification de Documents")
    st.subheader("Contexte")
    st.markdown(
        """
        <div style="text-align: justify;">
        Le projet s'inscrit dans un contexte de numérisation croissante des documents et de développement 
        de l'intelligence artificielle.Les entreprises, en particulier celles du secteur de l'assurance
        cherchent à automatiser le classement de leurs documents (acte de naissance, acte de vente …) 
        pour gagner en efficacité et en précision.Les techniques comme l’OCR (Optical Character Recognition), 
        le NLP (Natural Language Processing), le CV (Computer Vision) ou une approche hybride combinant le NLP et 
        le CV offrent des solutions prometteuses pour relever ce défi.
        </div>
        """,
        unsafe_allow_html=True
    )    
    st.subheader("Feature 2")
    st.write("Description of feature 2.")
    
    st.subheader("Feature 3")
    st.write("Description of feature 3.")


    
if __name__ == "__main__":
    home()