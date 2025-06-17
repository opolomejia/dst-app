import streamlit as st
import sys
sys.path.append('../../tools')
from tools import *
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

def text_mining_tab():
    st.header("Text Mining Analysis")
    # Add text mining visualizations and analysis here

    st.write("Text mining analysis content goes here")

def computer_vision_tab(df):
    st.header("Computer Vision Analysis")
    # Add computer vision visualizations and analysis here

    q_low = df['width'].astype(int).quantile(0.005)
    q_high = df['width'].astype(int).quantile(0.995)
    filtered_widths = df[(df['width'].astype(int) >= q_low) & (df['width'].astype(int) <= q_high)]
    st.write(filtered_widths.describe())


    reduced = df.drop('file_name', axis=1)
    table = reduced.groupby('label').agg(['mean', 'min', 'max']).style.set_caption("Width and Height Statistics")
    st.write(table)

    # First graph: Boxplot of widths by label
    widths = reduced.drop('height', axis=1)

    st.write("Mean width:", widths['width'].mean())
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='label', y='width', data=widths, hue='label', ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=75)
    ax1.set_xlabel('Document type')
    ax1.set_ylabel('Width in Pixels')
    ax1.set_title('Boxplot of Document Widths by Type')
    st.pyplot(fig1)

    # Second graph: Excluding extreme values
    q_low = widths['width'].astype(int).quantile(0.005)
    q_high = widths['width'].astype(int).quantile(0.995)
    filtered_widths = widths[(widths['width'].astype(int) >= q_low) & (widths['width'].astype(int) <= q_high)]

    st.write("Mean width (excluding extremes):", filtered_widths['width'].mean())
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='label', y='width', data=filtered_widths, hue='label', ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=75)
    ax2.set_xlabel('Document type')
    ax2.set_ylabel('Width in Pixels')
    ax2.set_title('Boxplot of Document Widths by Type (Excluding Extreme Values)')
    st.pyplot(fig2)

    st.write(filtered_widths.groupby('label').agg(['mean', 'min', 'max', 'count']).style.set_caption("Width Statistics (Excluding Extreme Values)"))
    st.write(filtered_widths.describe())

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.histplot(filtered_widths, x='label', hue='label', discrete=True, binwidth=0.5, shrink=0.8, legend=False, ax=ax3)
    ax3.set_xlabel('Document type')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=75)
    ax3.set_title('Distribution Excluding Extreme Values')

    for container in ax3.containers:
        bar_labels = [int(v.get_height()) if v.get_height() > 0 else '' for v in container]
        ax3.bar_label(container, labels=bar_labels, padding=3)

    st.pyplot(fig3)

    st.write("Computer vision analysis content goes here")

def main():

    st.title("Data Exploration")

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
    df = pd.read_parquet('df.parquet.gzip')
    
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