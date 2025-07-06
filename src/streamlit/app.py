import streamlit as st
import home
import data_exploration 
import demo
import modelisation

# Main entry point of the Streamlit application
def main():
    
    # Show the sidebar with navigation and contact info
    st.sidebar.title("Classification de Documents")
    page = st.sidebar.radio("Menu", ("Introduction", "Exploration des données", 
                                    "Modelisation", "Démonstration", "Conclusion & perspectives"))
    

    st.sidebar.header("Equipe")
    st.sidebar.markdown("""
    <ul>
        <li>
            <a href="https://www.linkedin.com/in/hiep-leconte-a7355bb4/" target="_blank">
                <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg" width="16" style="vertical-align:middle;"/> 
                Hiep LECONTE
            </a>
        </li>
        <li>
            <a href="https://www.linkedin.com/in/oliver-polo/" target="_blank">
                <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg" width="16" style="vertical-align:middle;"/> 
                Oliver POLO MEJIA
            </a>
        </li>
        <li>
            <a href="https://www.linkedin.com/in/somphone-sengsavang-4530823/" target="_blank">
                <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg" width="16" style="vertical-align:middle;"/>
                Somphone SENGSAVANG
            </a>
        </li>
    </ul>
    """, unsafe_allow_html=True)

    st.sidebar.header("Projet GitHub")
    st.sidebar.markdown("""
    <a href="https://github.com/DataScientest-Studio/sept24_cds_extractions_docs" target="_blank">
        <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg" width="20" style="vertical-align:middle; margin-right:8px;"/>
        Voir le projet sur GitHub
    </a>
    """, unsafe_allow_html=True)
    if page == "Introduction":
        home.home()
    elif page == "Exploration des données": 
        data_exploration.main()
    elif page == "Modelisation":
        modelisation.main()
    elif page == "Démonstration":
        demo.demo_interface()
    elif page == "Conclusion & perspectives":

        st.title("Conclusion et Perspectives")
        conclusion, perspectives = st.tabs(["Conclusion", "Perspectives"])

        with conclusion:
            st.subheader("Conclusion")
            st.markdown(
                """
                <div style="text-align: justify;">
                L'extraction et la classification de documents numérisés est un sujet techniquement complexe , 
                il nous a permis de travailler sur un projet complet suivant une méthodologie rigoureuse en 
                Data Science et de créer une application web Streamlit.
                <br><br>
                Nous avons testé trois types d'approches différentes pour résoudre notre problème de classification 
                multi classes nominal en utilisant les modèles de classifications, les techniques avancées en Machine 
                Learning, le Computer Vision, le Text Mining, les Réseaux de Neurones Convolutifs et CLIP.
                <br><br>
                Pour traiter les images, nous sommes confrontés à un problème de ressources matérielles, nous obligeant 
                à réduire considérablement nos jeux de données (15%) et la résolution des images.
                <br><br>
                Malgré ces traitements, nous avons obtenu des bons résultats sur les modèles de classifications multiclasses, 
                les réseaux de neurones et des mauvais résultats sur le modèle CLIP.
                <br><br>
                Les résultats décevants de CLIP peuvent s’expliquer par une inadéquation entre les images utilisées et celles 
                sur lesquelles le modèle a été initialement entraîné.
                </div>
                """,
                unsafe_allow_html=True
            )

        with perspectives:
            st.subheader("Perspectives")
            st.markdown(
                """
                <div style="text-align: justify;">
                Cette expérience souligne : <br>
                - l'importance des pré-traitements, <br>
                - d'avoir des machines puissantes afin d'entraîner nos modèles avec plus de données et faire des traitements plus poussés sur les images, <br>
                - la nécessité d’explorer des modèles les mieux adaptés, plus particulièrement pour CLIP.
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()