import streamlit as st
import pandas as pd
import numpy as np

st.markdown("""
    <style>
            
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }        
    .stButton button {
        margin: auto;
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center;'>Forecasting Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<h6 style='text-align: center;'>Select a model to forecast</h6>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.button("SARIMA", key="sarima_button")
with col2:
    st.button("Prophet", key="prophet_button")
with col3:
    st.button("LGBM", key="lgbm_button")
with col4:
    st.button("XGBoost", key="xgboost_button")

    


