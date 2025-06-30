import streamlit as st
import pandas as pd 
import seaborn as sns
import os
import matplotlib.pyplot as plt
from cv_exploration import *
from text_minning_exploration import *

#directory = "/mount/src/dst-app/src/streamlit/"
directory = "./"

def main():

    st.title("Exploration des données")

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

    # Load the DataFrame from the parquet file
    #df = pd.read_parquet("./df.parquet.gzip")
    df = pd.read_parquet(directory+"df.parquet.gzip")
    
    st.markdown(
        """
        <div style="text-align: justify;">
        La distribution des types de documents dans le jeu de données d'entraînement montre que les différentes catégories sont représentées de façon uniforme. 
        Un tel équilibre entre les classes est essentiel pour garantir la performance et la robustesse des modèles de machine learning. 
        Le graphique ci-dessous présente la répartition des documents par type, confirmant l'absence de déséquilibre significatif dans les données.
        <br><br>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Allow user to select which labels to display using a multiselect (multi choice)
    label_options = [labels[str(i)] for i in sorted(map(int, labels.keys()))]
    selected_labels = st.multiselect(
        "Select document types to display",
        options=label_options,
        default=label_options  # Show all by default
    )

    # Map selected label names back to their numeric codes (as strings)
    selected_label_codes = [k for k, v in labels.items() if v in selected_labels]
    filtered_df = df[df['label'].astype(str).isin(selected_label_codes)].copy()
    # Ensure 'label' is categorical with the selected order
    filtered_df['label'] = pd.Categorical(filtered_df['label'].astype(str), categories=selected_label_codes, ordered=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    # Only plot bars for the selected labels
    sns.histplot(
        filtered_df,
        x='label',
        hue='label',
        discrete=True,
        binwidth=0.5,
        shrink=0.8,
        legend=False,
        ax=ax
    )
    ax.set_xlabel('Document type')
    # Set x-tick labels to the selected label names, only for selected labels
    ax.set_xticks(range(len(selected_label_codes)))
    ax.set_xticklabels([labels[code] for code in selected_label_codes], rotation=75)

    for container in ax.containers:
        bar_labels = [int(v.get_height()) if v.get_height() > 0 else '' for v in container]
        ax.bar_label(container, labels=bar_labels, padding=3)

    st.pyplot(fig)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Text Mining", "Computer Vision"])
    
    with tab1:
        text_mining_tab()
    
    with tab2:
        # Replace numerical label with string value using the labels dictionary
        df['label'] = df['label'].astype(str).map(labels)    
        computer_vision_tab(df)

if __name__ == "__main__":
    main()