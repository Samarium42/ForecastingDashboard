import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from warnings import filterwarnings
filterwarnings("ignore")

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

col1, col2, col3, col4 = st.columns(4)

# Function to handle SARIMAX forecasting
def sarima_forecast():
    df = pd.read_csv('./ForecastingDashboard/salesorderheader.csv')  # Ensure this file is in your directory
    df = df[['OrderDate', 'SubTotal']]
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df.set_index('OrderDate', inplace=True)
    df = df.groupby('OrderDate').sum()
    train = df[:int(0.8 * len(df))]
    test = df[int(0.8 * len(df)):]
    
    model = SARIMAX(train, order=(3, 0, 1), seasonal_order=(3, 0, 1, 30))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=len(test))
    
    # Plotting the results
    fig, ax = plt.subplots()
    ax.plot(test.index, test['SubTotal'], label='Actual')
    ax.plot(test.index, forecast, label='Forecast', linestyle='--')
    ax.legend()
    st.pyplot(fig)

with col1:
    if st.button("SARIMA"):
        sarima_forecast()
with col2:
    st.button("Prophet", key="prophet_button")
with col3:
    st.button("LGBM", key="lgbm_button")
with col4:
    st.button("XGBoost", key="xgboost_button")
