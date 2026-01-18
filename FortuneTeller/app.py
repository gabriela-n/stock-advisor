import streamlit as st
from ui_config import setup_ui, stock_search, analyze_page, favorites_page


setup_ui()
selected_page = st.sidebar.radio("🔮", ["Stock Analysis", "Favorites"])

if selected_page == "Stock Analysis":
    symbol, timeframe, interval = stock_search()
    analyze_page(symbol, timeframe, interval)
    
elif selected_page == "Favorites":
    favorites_page()
