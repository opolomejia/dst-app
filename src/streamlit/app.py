import streamlit as st
import home
import data_exploration 
import demo

# Main entry point of the Streamlit application
def main():
    
    # Show the sidebar with navigation and contact info
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page:", ("Home", "Data Exploration", "Demo"))
    
    st.sidebar.header("Contact Info")
    st.sidebar.write("For inquiries, please contact:")
    st.sidebar.write("Email: contact@example.com")
    st.sidebar.write("Phone: +123456789")

    if page == "Home":
        home.home()
    elif page == "Data Exploration": 
        data_exploration.main()
    elif page == "Demo":
        demo.demo_interface()

if __name__ == "__main__":
    main()