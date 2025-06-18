import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

def computer_vision_tab(df):
    st.header("Exploration des Données - Computer Vision")

    st.markdown(
        """
        <div style="text-align: justify;">
        La première étape de l’approche de modélisation pour une classification basée directement sur les images, 
        est la conversion des images à classer en ‘array” des pixels. Dans notre cas, tous les images sont en blanc 
        et noir, nous pouvons donc utiliser la  scale de gris (IMREAD_GRAYSCALE) de la librairie OpenCV (CV2); 
        et representer nos images comme des ‘arrays’ 2D.
        <br><br>
        Ci-dessous une table présentant des statistiques des largeurs (width) et des hauteurs (height) des images en 
        fonction du type de document:
        <br>
        </div>
        """,
        unsafe_allow_html=True
    )

    reduced = df.drop('file_name', axis=1)
    table = reduced.groupby('label').agg(['mean', 'min', 'max']).style.set_caption("Width and Height Statistics")
    st.write(table)

    st.markdown(
        """
        <div style="text-align: justify;">
        De l’information affichée de la table nous observons que la moyenne, le min et le max ont les mêmes 
        valeurs entre eux et pour tous les types des documents. Nous pouvons donc conclure que toutes les 
        images ont une hauteur de 1000 pixels.
        <br><br>
        Par contre, quand nous analysons les statistiques pour des largeurs (width), nous observons des 
        différences sur ces valeurs et nous pouvons même suspecter la présence de valeurs extrêmes pour certains
        type de documents comme “advertisement” and  “scientific publication” où la valeur max est assez éloigné 
        de la valeur moyenne. Nous devons donc tracer des boîtes de moustaches pour le confirmer.
        La distribution des largeurs des images est représentée par le graphique ci-dessous.
        <br><br>
        </div>
        """,
        unsafe_allow_html=True
    )

    # First graph: Boxplot of widths by label
    widths = reduced.drop('height', axis=1)

    # Interactive boxplot: allow user to select one label or all
    labels = widths['label'].unique().tolist()
    selected_label = st.selectbox(
        "Sélectionnez un type de document à afficher (ou 'Tous')",
        options=['Tous'] + labels,
        index=0
    )

    if selected_label == 'Tous':
        plot_data = widths
        title = 'Boxplot of Document Widths by Type'
    else:
        plot_data = widths[widths['label'] == selected_label]
        title = f'Boxplot of Document Widths for "{selected_label}"'

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='label', y='width', data=plot_data, hue='label', ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=75)
    ax1.set_xlabel('Document type')
    ax1.set_ylabel('Width in Pixels')
    ax1.set_title(title)
    st.pyplot(fig1)

    st.markdown(
        """
        <div style="text-align: justify;">
        Les boîtes à moustache nous confirment bien la présence des valeurs extrêmes pour la largeur 
        (width) des images dans certaines catégories comme <b>“advertisement”, “scientific publication”, 
        “news articles” et “questionnaire”</b>.
        <br><br>
        La plupart des modèles de classification demandent d’avoir des images en entrée ayant la même 
        taille, nous forçant à effectuer un changement de taille de nos images. L'application de ces 
        techniques de “resizing” sur des données extrêmes pourrait nuire à la performance du modèle. 
        Nous devrions donc exclure les données extrêmes de notre jeu d'entraînement.
        <br><br>
        On propose de <b>ne conserver que les données entre les quantiles 0.5% et 99.5%</b>, ce qui nous fera 
        perdre au maximum 1% des données. Nous présentons ci-dessous la distribution des images restantes 
        par type de document, le tableau de statistiques et des boîtes à moustaches pour analyser la 
        distribution des données restantes.
        <br><br>
        Le graphique ci-dessous présente les boîtes à moustaches des largeurs des images après suppression des valeurs extrêmes. 
        On observe que la distribution des largeurs est désormais plus homogène pour chaque type de document, 
        sans la présence de valeurs aberrantes qui pouvaient fausser l'analyse. 
        Cette étape permet de mieux préparer les données pour l'entraînement des modèles de classification, 
        en limitant l'impact des cas atypiques sur la performance globale.
        <br><br>
        </div>
        """,
        unsafe_allow_html=True
    )


    # Second graph: Excluding extreme values
    q_low = widths['width'].astype(int).quantile(0.005)
    q_high = widths['width'].astype(int).quantile(0.995)
    filtered_widths = widths[(widths['width'].astype(int) >= q_low) & (widths['width'].astype(int) <= q_high)]

    # Interactive boxplot after excluding extreme values
    filtered_labels = filtered_widths['label'].unique().tolist()
    selected_filtered_label = st.selectbox(
        "Sélectionnez un type de document à afficher après filtrage (ou 'Tous')",
        options=['Tous'] + filtered_labels,
        index=0,
        key="filtered_label_select"
    )

    if selected_filtered_label == 'Tous':
        plot_filtered_data = filtered_widths
        filtered_title = 'Boxplot of Document Widths by Type (Excluding Extreme Values)'
    else:
        plot_filtered_data = filtered_widths[filtered_widths['label'] == selected_filtered_label]
        filtered_title = f'Boxplot of Document Widths for "{selected_filtered_label}" (Excluding Extreme Values)'

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='label', y='width', data=plot_filtered_data, hue='label', ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=75)
    ax2.set_xlabel('Document type')
    ax2.set_ylabel('Width in Pixels')
    ax2.set_title(filtered_title)
    st.pyplot(fig2)

    st.markdown(
        """
        <div style="text-align: justify;">
        Même après avoir filtré les données nous avons une distribution presque uniforme des images 
        sur les différents types de documents. 
        Les ranges des valeurs pour les largeurs est aussi plus petite. <b>Nous sommes passés de 320 000 
        images à 316 860 (soit 99,02% des images initiales)</b>, ce qui est un bon compromis et nous permet
        de continuer a avoir une bonne représentation des différents types de documents.
        <br><br>
        </div>
        """,
        unsafe_allow_html=True
    )

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.histplot(filtered_widths, x='label', hue='label', discrete=True, binwidth=0.5, shrink=0.8, legend=False, ax=ax3)
    ax3.set_xlabel('Document type')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=75)
    ax3.set_title('Distribution Excluding Extreme Values')

    for container in ax3.containers:
        bar_labels = [int(v.get_height()) if v.get_height() > 0 else '' for v in container]
        ax3.bar_label(container, labels=bar_labels, padding=3)

    st.pyplot(fig3)