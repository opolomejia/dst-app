import streamlit as st
import pandas as pd
def text_mining():
    st.subheader("Les modèles de classifications")
    st.markdown(
        """
        <div style="text-align: justify;">
        Pour notre apprentissage supervisé, nous avons un problème de classification multi-classe, 
        car la variable cible (la nature du document) peut prendre 16 labels. Nous prévoyons de tester 
        trois algorithmes couramment utilisés et robustes pour ce type de problème afin de sélectionner
        celui qui performe le mieux : <b>la régression logistique, le SVC (Support Vector Classifier) et 
        le Random Forest.</b>
        <br><br>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Sélection de modèles")
    st.markdown(
        """
        <div style="text-align: justify;">
        La sélection de modèles est un élément central dans le processus de construction de bons modèles 
        de Machine Learning (supervisé) et peut se résumer au choix des meilleurs hyperparamètres.
        Ces hyperparamètres sont des paramètres définis avant l'entraînement du modèle.Pour ce faire, trois 
        techniques principales peuvent être utilisées pour explorer l'espace des hyperparamètres : GridSearchCV,
        RandomSearchCV et BayesSearchCV.
        Les résultats sont présentés ci-dessous et sont très faibles, quel que soit le modèle utilisé : 
        </div>
        """,
        unsafe_allow_html=True
    )
    #st.image("text_model_comp_1.png", caption="Résultats de la sélection de modèles", use_container_width=True)
    st.image("/mount/src/dst-app/src/streamlit/text_model_comp_1.png", caption="Résultats de la sélection de modèles", use_container_width=True)


    st.markdown(
        """
        <div style="text-align: justify;">
        Cela indique que le premier pré-traitement des données n'est pas satisfaisant et doit être revu et/ou il 
        faut retraiter les images. Ainsi, un retraitement de l’image (librairie CV2)  a été initié pour essayer 
        d’améliorer l’extraction des textes des images.
        <br><br>
        Les traitements se font sur le chargement d'une image en N&B, le calibrage sur le contraste et la luminosité
        d’une image ainsi que sur la réduction du bruit gaussien d'une image en N&B. Le bruit gaussien apparaît dans
        une image comme des fluctuations aléatoires de l’intensité des pixels. Cela donne à l’image un aspect granuleux, 
        irrégulier ou  neigeux.
        <br><br>
        Face à la lourdeur des traitements, nous avons réduit considérablement l’échantillon et évalué les modèles 
        avec les mêmes hyperparamètres pour un <b>résultat très faible</b>.
        <br><br>
        Face à ce résultat, nous avons écarté le retraitement des images et travaillé sur l’étape de preprocessing
        (la présente note décrit le dernier prétraitement sans les traitements sur les images). Les entraînements 
        et les évaluations se font sur les deux meilleurs modèles à savoir LogisticRegression et RandomForestClassifier 
        avec le dernier prétraitement. Les résultats sont dans le tableau ci-dessous :
        </div>
        """,
        unsafe_allow_html=True
    )

    #st.image("text_model_final_comp.png", caption="Résultats de la sélection de modèles après prétraitement", use_container_width=True)
    st.image("/mount/src/dst-app/src/streamlit/text_model_final_comp.png", caption="Résultats de la sélection de modèles après prétraitement", use_container_width=True)
    
    st.subheader("Rapport de classification et matrice de confusion")
    st.markdown(
        """
        <div style="text-align: justify;">
        Dans la suite, nous présentons uniquement le rapport de classification et la matrice de confusion du modèle 
        <b>LogisticRegression</b> avec le dernier prétraitement, car il s'agit du modèle ayant obtenu les meilleurs 
        résultats lors de nos expérimentations.
        </div>
        """,
        unsafe_allow_html=True
    )
    #st.image("text_model_class_rep.png", caption="Rapport de classification du modèle LogisticRegression", use_container_width=False)
    st.image("/mount/src/dst-app/src/streamlit/text_model_class_rep.png", caption="Rapport de classification du modèle LogisticRegression", use_container_width=False)

    st.markdown("<br>", unsafe_allow_html=True)

    #st.image("text_model_conf_mat.png", caption="Matrice de confusion du modèle LogisticRegression", use_container_width=True)
    st.image("/mount/src/dst-app/src/streamlit/text_model_conf_mat.png", caption="Matrice de confusion du modèle LogisticRegression", use_container_width=True)


def objectives():
    st.subheader("Objectifs")
    st.markdown(
        """
        <div style="text-align: justify;">
        Le projet vise à développer un système de classification automatique des documents en utilisant des techniques avancées 
        d'intelligence artificielle. L'objectif principal est de créer un modèle capable de reconnaître et de classer différents 
        types de documents (acte de naissance, acte de vente, etc.) avec une précision élevée. 
        <br><br>
        Pour atteindre cet objectif, nous allons explorer plusieurs approches, notamment l'utilisation de l'OCR pour extraire le texte 
        des documents, le NLP pour analyser le contenu textuel et le Computer Vision pour traiter les éléments visuels (à l'aide des réseaux de neurones).
        Une approche hybride combinant ces techniques sera également envisagée (CLIP).
        </div>
        """,
        unsafe_allow_html=True
    )

def data():
    st.subheader("Données d'entraînement")
    st.markdown(
        """
        <div style="text-align: justify;">
        Les données d'entrainement proviennent du projet public de Adam W. Harley
        <a href="https://adamharley.com/rvl-cdip/" target="_blank">RVL-CDIP 
        (Ryerson Vision Lab Complex Document Information Processing)</a>.
        <br><br>
        Le jeu de données comprend 400 000 images en niveaux de gris réparties en 16 classes, 
        avec 25 000 images par classe. Il contient 320 000 images d'entraînement, 40 000 images 
        de validation et 40 000 images de test. Les images sont dimensionnées de manière à ce que 
        leur plus grande dimension ne dépasse pas 1000 pixels.
        <br><br>
        Les classes de documents présentes dans ce jeu de données sont les suivantes :
        <ul>
            <li><b>0</b> : letter</li>
            <li><b>1</b> : form</li>
            <li><b>2</b> : email</li>
            <li><b>3</b> : handwritten</li>
            <li><b>4</b> : advertisement</li>
            <li><b>5</b> : scientific report</li>
            <li><b>6</b> : scientific publication</li>
            <li><b>7</b> : specification</li>
            <li><b>8</b> : file folder</li>
            <li><b>9</b> : news article</li>
            <li><b>10</b> : budget</li>
            <li><b>11</b> : invoice</li>
            <li><b>12</b> : présentation</li>
            <li><b>13</b> : questionnaire</li>
            <li><b>14</b> : resume</li>
            <li><b>15</b> : memo</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    st.title("Modelisation")
    # Create tabs
    text_mining_tab, cv_tab, clip_tab = st.tabs(["Text Mining", "Computer Vision", "CLIP"])
    with text_mining_tab:
        text_mining()

    with cv_tab:
        objectives()
    
    with clip_tab:
        data()

    
if __name__ == "__main__":
    main()