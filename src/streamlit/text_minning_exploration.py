import streamlit as st
import pandas as pd 
import seaborn as sns
import string
import matplotlib.pyplot as plt
import re
import numpy as np
import random
from wordcloud import WordCloud
#Fonction pour supprimer les séquences indésirables
def clean_special_chars_and_numbers(text):
    #re.sub(modèle, remplacement, chaîne)
    #Supprimer les caractères spéciaux
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Fonction pour supprimer les espaces superflus
def remove_extra_spaces(text):
    return ' '.join(text.split())

# Fonction pour enlever la ponctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))
    
def text_mining_tab():
    st.header("Exploration des Données - Text Mining")
    #df = pd.read_parquet("./df.parquetv.gzip")
    df = pd.read_parquet("/mount/src/dst-app/src/streamlit/df.parquetv.gzip")

    labels = {
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
        "12": "présentation",
        "13": "questionnaire",
        "14": "resume",
        "15": "memo"
    }

    st.markdown(
        """
        <div style="text-align: justify;">
        Étant donné le volume très important du jeu de données d'entraînement, nous avons extrait des textes 
        sur 50 000 images soit  15% des images d'entraînement.  
        Sur cet échantillon, les différentes classes  sont représentées de façon uniforme. 
        Le jeu de données pour l'entraînement est équilibré. Comme nous pouvons le voir sur le graphique ci-dessus: 
        </div>
        """,
        unsafe_allow_html=True
    )


    df["easyocr_text_nbr"] = df["easyocr_text"].apply(len)
    #Supprimer les lignes vides
    df = df[df['easyocr_text'].apply(len) > 0].copy()
    # Convertir la colonne 'label' en type entier
    df['label'] = df['label'].astype(int)
    label_counts = df['label'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [plt.cm.tab20(random.randint(0, 19)) for _ in range(len(label_counts))]
    bars = ax.bar(label_counts.index, label_counts.values, color=colors)

    ax.set_title("Distribution des labels")
    ax.set_xlabel("Label")
    ax.set_ylabel("Nombre d'occurrences")

    for i, v in enumerate(label_counts.values):
        ax.text(label_counts.index[i], v + 0.5, str(v), ha='center', va='bottom', fontsize=10)

    ax.set_xticks(np.arange(0, 16, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown(
        """
        <div style="text-align: justify;">
        Nous ajoutons ci-dessous un graphique de type "moustache" (boxplot) afin d'analyser la distribution 
        du nombre de mots extraits pour chaque type de document à classifier. Cela permet de visualiser la dispersion, 
        les éventuelles différences selon les classes, ainsi que d'identifier la présence de valeurs aberrantes.
        <br><br>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Sélection interactive du label
    label_options = ['Tous'] + sorted(df['label'].unique().tolist())
    selected_label = st.selectbox("Sélectionnez un label à afficher :", label_options)

    if selected_label == 'Tous':
        filtered_df = df
        box_title = 'Boxplot Nombre de mots par label (données brutes)'
    else:
        filtered_df = df[df['label'] == selected_label]
        box_title = f'Boxplot Nombre de mots pour le label {selected_label} (données brutes)'

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        x='label' if selected_label == 'Tous' else None,
        y='easyocr_text_nbr',
        data=filtered_df,
        ax=ax2
    )
    ax2.set_title(box_title)
    ax2.set_xlabel('Label' if selected_label == 'Tous' else '')
    ax2.set_ylabel('Nombre de mots')
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown(
        """
        <div style="text-align: justify;">
        Nous constatons qu’il y a beaucoup de valeurs extrêmes dans les données brutes. Il convient de détecter 
        et supprimer les valeurs aberrantes (outliers) en utilisant la méthode de l'écart interquartile (IQR). Ci-dessous, 
        nous appliquons cette méthode pour supprimer les valeurs extrêmes et afficher à nouveau le boxplot du nombre de mots 
        extraits par type de document:

        </div>
        """,
        unsafe_allow_html=True
    )

    # Supprimer les données aberrantes
    # Calcul des quartiles et de l'IQR
    Q1 = df['easyocr_text_nbr'].quantile(0.25)
    Q3 = df['easyocr_text_nbr'].quantile(0.75)
    IQR = Q3 - Q1

    # Calcul des seuils haut et bas pour identifier les valeurs aberrantes
    seuil_haut = Q3 + 1.5 * IQR
    seuil_bas = Q1 - 1.5 * IQR

    non_aberrantes_mask = (df['easyocr_text_nbr'] >= seuil_bas) & (df['easyocr_text_nbr'] <= seuil_haut)

    # Créer un nouveau DataFrame contenant uniquement les valeurs non aberrantes
    df_clean = df[non_aberrantes_mask].copy()

    # Sélection interactive du label
    label_options = ['Tous'] + sorted(df_clean['label'].unique().tolist())
    selected_label = st.selectbox("Sélectionnez un label à afficher:", label_options)

    if selected_label == 'Tous':
        filtered_df = df_clean
        box_title = 'Boxplot Nombre de mots par label (données nettoyées)'
    else:
        filtered_df = df_clean[df_clean['label'] == selected_label]
        box_title = f'Boxplot Nombre de mots pour le label {selected_label} (données nettoyées)'

    fig3, ax3= plt.subplots(figsize=(12, 6))
    sns.boxplot(
        x='label' if selected_label == 'Tous' else None,
        y='easyocr_text_nbr',
        data=filtered_df,
        ax=ax3
    )
    ax3.set_title(box_title)
    ax3.set_xlabel('Label' if selected_label == 'Tous' else '')
    ax3.set_ylabel('Nombre de mots')
    plt.tight_layout()
    st.pyplot(fig3)

    st.markdown(
        """
        <div style="text-align: justify;">
        Nous ajoutons également des graphiques de type "word cloud" (nuages de mots) pour chaque classe de document. 
        Ces visualisations permettent d’identifier rapidement les mots les plus fréquents dans les textes extraits, 
        facilitant ainsi la compréhension des thématiques dominantes et des particularités lexicales propres à chaque catégorie.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Remplacer la valeur entière de 'label' par la chaîne correspondante
    df['label'] = df['label'].astype(str).map(labels)
    df_clean['label'] = df_clean['label'].astype(str).map(labels)

    #Convertir les mots en minuscules
    df['easyocr_text_cleaned'] = df['easyocr_text'].apply(lambda x: ' '.join(x).lower() if isinstance(x, list) else str(x).lower())
    df_clean['easyocr_text_cleaned'] = df_clean['easyocr_text'].apply(lambda x: ' '.join(x).lower() if isinstance(x, list) else str(x).lower())
    
    #Appliquer la suppression de la ponctuation
    df['easyocr_text_cleaned'] = df['easyocr_text_cleaned'].apply(remove_punctuation)
    df_clean['easyocr_text_cleaned'] = df_clean['easyocr_text_cleaned'].apply(remove_punctuation)

    #espaces multiples entre les mots et des espaces au début et à la fin.
    df['easyocr_text_cleaned'] = df['easyocr_text_cleaned'].apply(remove_extra_spaces)
    df_clean['easyocr_text_cleaned'] = df_clean['easyocr_text_cleaned'].apply(remove_extra_spaces)

    #Nettoyer les données
    df['easyocr_text_cleaned'] = df['easyocr_text_cleaned'].apply(clean_special_chars_and_numbers)
    df_clean['easyocr_text_cleaned'] = df_clean['easyocr_text_cleaned'].apply(clean_special_chars_and_numbers)

    # Create tabs
    tab1, tab2 = st.tabs(["Word Cloud: Données brutes", "Word Cloud: Données nettoyées"])

    with tab1:
        st.subheader("Nuage de mots - Données brutes")
        # Générer le nuage de mots pour les données brutes
        label_options_wc = sorted(df['label'].unique().tolist())
        selected_label_wc = st.selectbox("Sélectionnez un label pour le nuage de mots :", label_options_wc, key="wordcloud_label")

        # Filtrer les données selon le label sélectionné
        df_wc = df[df['label'] == selected_label_wc]

        # Concaténer tous les textes en une seule chaîne
        all_text = ' '.join(df_wc['easyocr_text_cleaned'].astype(str))

        # Générer le nuage de mots
        wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(all_text)

        # Afficher le nuage de mots
        fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)

    with tab2:
        st.subheader("Nuage de mots - Données nettoyées")
        # Sélection du label pour le nuage de mots
        label_options_wc = sorted(df_clean['label'].unique().tolist())
        selected_label_wc = st.selectbox("Sélectionnez un label pour le nuage de mots :", label_options_wc, key="wordcloud_label_clean")

        # Filtrer les données selon le label sélectionné
        df_wc = df_clean[df_clean['label'] == selected_label_wc]

        # Concaténer tous les textes en une seule chaîne
        all_text = ' '.join(df_wc['easyocr_text_cleaned'].astype(str))

        # Générer le nuage de mots
        wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(all_text)

        # Afficher le nuage de mots
        fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)