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
                                              "Modelisation", "Démonstrateur"))
    

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
    elif page == "Démonstrateur":
        demo.demo_interface()

if __name__ == "__main__":
    main()