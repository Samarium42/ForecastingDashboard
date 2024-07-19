import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from warnings import filterwarnings
from prophet import Prophet
import altair as alt
from streamlit_navigation_bar import st_navbar
filterwarnings("ignore")

    
# Function to handle SARIMAX forecasting
def sarima_forecast():

    df = pd.read_csv('salesorderheader.csv')  
    df = df[['OrderDate', 'SubTotal']]
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df.set_index('OrderDate', inplace=True)
    df = df.groupby('OrderDate').sum()
    
    train = df[:int(0.8 * len(df))]
    test = df[int(0.8 * len(df)):]
    
    model = SARIMAX(train, order=(3, 0, 1), seasonal_order=(3, 0, 1, 30))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=len(test))
    
    forecast_df = pd.DataFrame(forecast, index=test.index, columns=['Forecast'])
    train['Type'] = 'Train'
    test['Type'] = 'Test'
    forecast_df['Type'] = 'Forecast'
    
    combined_df = pd.concat([train, test, forecast_df])
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={'OrderDate': 'Date'}, inplace=True)
    
    chart = alt.Chart(combined_df).mark_line().encode(
        x='Date:T',
        y='SubTotal:Q',
        color='Type:N'
    ).properties(
        title='Sales and Forecast'
    )
    
    st.altair_chart(chart, use_container_width=True)
    

def prophet_forecast():
    df = pd.read_csv('salesorderheader.csv')  # Ensure this file is in your directory
    df = df[['OrderDate', 'SubTotal']]
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df.set_index('OrderDate', inplace=True)
    df = df.groupby('OrderDate').sum()
    train = df[:int(0.8 * len(df))]
    test = df[int(0.8 * len(df)):]

    model = Prophet()
    model.fit(train.reset_index().rename(columns={'OrderDate': 'ds', 'SubTotal': 'y'}))
    forecast = model.predict(test.reset_index().rename(columns={'OrderDate': 'ds'}))

    st.line_chart(forecast[['ds', 'yhat']])
    st.line_chart(df)
    



page = st_navbar(["SARIMAX", "Prophet", "XGBoost", "LGBM"])
if page == "SARIMAX":
    sarima_forecast()