import streamlit as st

def home():
    st.title("Welcome to the Streamlit App!")
    st.write("This is the home page where you can find introductory information about the application.")
    st.write("Use the sidebar to navigate to different sections of the app.")
    
if __name__ == "__main__":
    home()