import streamlit as st


def load_css():
    with open("gui/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def initialize_session_state():
    if "favorites" not in st.session_state:
        st.session_state.favorites = []
