import streamlit as st
import home
import data_exploration 
import demo

# Main entry point of the Streamlit application
def main():
    
    # Show the sidebar with navigation and contact info
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page:", ("Introduction", "Exploration des données", "Modelisation", "Demo"))
    
    st.sidebar.header("Contact Info")
    st.sidebar.write("For inquiries, please contact:")
    st.sidebar.write("Email: contact@example.com")
    st.sidebar.write("Phone: +123456789")

    if page == "Introduction":
        home.home()
    elif page == "Exploration des données": 
        data_exploration.main()
    elif page == "Modelisation":
        st.title("Modélisation")
        st.markdown(
            """
            <div style="text-align: justify;">
            Cette section est dédiée à la modélisation des données. Vous pouvez explorer les différentes approches de modélisation 
            et les résultats obtenus.
            </div>
            """,
            unsafe_allow_html=True
        )
        # Placeholder for modelisation content
        st.write("Contenu de la modélisation à venir...")
    elif page == "Demo":
        demo.demo_interface()

if __name__ == "__main__":
    main()