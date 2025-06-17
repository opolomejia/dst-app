import streamlit as st

def create_contact_sidebar():
    st.sidebar.markdown("## Contact Information")
    st.sidebar.markdown("---")
        
    # Contact details with custom CSS for better visibility
    st.sidebar.markdown("""
    <style>
    .contact-info {
        position: fixed;
        padding: 10px;
        background-color: #f0f2f6;
        border-radius: 5px;
        margin-top: 20px;
    }
    .contact-link {
        display: flex;
        align-items: center;
        padding: 5px 0;
        text-decoration: none;
        color: inherit;
    }
    </style>
    
    <div class="contact-info">
        <div class="contact-link">
            📧 <a href="mailto:your.email@example.com">your.email@example.com</a>
        </div>
        <div class="contact-link">
            🐱 <a href="https://github.com/yourusername">GitHub Profile</a>
        </div>
        <div class="contact-link">
            💼 <a href="https://linkedin.com/in/yourusername">LinkedIn</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Your App Name",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"  # Keep sidebar always expanded
    )
    
    # Initialize session state for persistent sidebar
    if 'sidebar_state' not in st.session_state:
        st.session_state.sidebar_state = 'expanded'
    
    # Add contact info to sidebar
    create_contact_sidebar()
    
    # Main content area
    st.title("Welcome to Your App")
    # ...existing code...

if __name__ == "__main__":
    main()