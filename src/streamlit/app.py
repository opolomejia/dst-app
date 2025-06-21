import streamlit as st
import home
import data_exploration 
import demo
import modelisation

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
        modelisation.main()
    elif page == "Demo":
        demo.demo_interface()

if __name__ == "__main__":
    main()