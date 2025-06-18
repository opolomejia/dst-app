import streamlit as st

def context():
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
        <br><br>
        L'utilisation d'algorithmes de classification dans l'industrie a déjà prouvé son efficacité, 
        notamment dans le traitement et la priorisation de documents volumineux. A Air France, par exemple,
        des algorithmes de machine learning (basées principalement sur le NLP) sont utilisés pour analyser et 
        classer des rapports de vol rédigés par le personnel navigant, permettant ainsi de prioriser les rapports 
        nécessitant une prise en charge rapide. Ce type de solution montre l'importance de l'automatisation et de la 
        précision dans la gestion des documents en grand volume. <br><br>
        Dans le monde de l’assurance, lors de souscription en ligne les sociétaires sont souvent amenés à télécharger des documents. 
        L’implémentation d’un système de classification automatique des documents permettrait de reconnaître et d’organiser ces 
        fichiers de manière optimale. Grâce aux technologies de reconnaissance optique des caractères (OCR) et 
        d’apprentissage automatique, ce système pourrait analyser et classer divers types de documents rapidement et avec précision.
        <br><br>
        L’expérience utilisateur serait facilitée si ces documents étaient reconnus automatiquement, ce qui permettait de gagner 
        du temps sur un parcours client. En réduisant le besoin de vérifications manuelles, les compagnies d’assurance pourraient 
        réduire les coûts opérationnels. De plus, une meilleure efficacité dans le traitement des dossiers pourrait augmenter la 
        satisfaction des clients et renforcer la fidélité, ayant ainsi un impact positif sur la rentabilité à long terme.
        </div>
        """,
        unsafe_allow_html=True
    ) 


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
        """,
        unsafe_allow_html=True
    )

def home():
    st.title("Projet: Classification de Documents")
    # Create tabs
    context_tab, objectives_tab, data_tab = st.tabs(["Contexte", "Objectifs", "Données d'entraînement"])
    with context_tab:
        context()

    with objectives_tab:
        objectives()
    
    with data_tab:
        data()

    
if __name__ == "__main__":
    home()