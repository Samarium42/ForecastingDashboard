import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from warnings import filterwarnings
from prophet import Prophet
import altair as alt
from streamlit_navigation_bar import st_navbar
import lightgbm as lgb
import xgboost as xgb
filterwarnings("ignore")

    
# Function to handle SARIMAX forecasting
def sarima_forecast():

    df = pd.read_csv('./ForecastingDashboard/salesorderheader.csv')  
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

    future_df = pd.DataFrame(index=pd.date_range(start=df.index[-1], periods=30))
    
    combined_df = pd.concat([train, test, forecast_df])
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={'OrderDate': 'Date'}, inplace=True)
    
    chart1 = alt.Chart(combined_df).mark_line().encode(
        x='Date:T',
        y='SubTotal:Q',
        color='Type:N'
    ).properties(
        title='Sales and Test Forecast'
    )

    chart2 = alt.Chart(future_df).mark_line(color='red').encode(
        x='index:T',
        y='Forecast:Q'
    ).properties(
        title='Future Forecast'
    )    
    
    
    st.altair_chart(chart1, use_container_width=True)
    st.altair_chart(chart2, use_container_width=True)
    

def prophet_forecast():
    df = pd.read_csv('./ForecastingDashboard/salesorderheader.csv')  # Ensure this file is in your directory
    df = df[['OrderDate', 'SubTotal']]
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df.set_index('OrderDate', inplace=True)
    df = df.groupby('OrderDate').sum()
    train = df[:int(0.8 * len(df))]
    test = df[int(0.8 * len(df)):]

    model = Prophet()
    model.fit(train.reset_index().rename(columns={'OrderDate': 'ds', 'SubTotal': 'y'}))
    forecast = model.predict(test.reset_index().rename(columns={'OrderDate': 'ds'}))

    future = model.make_future_dataframe(periods=30)

    chart1 = alt.Chart(forecast).mark_line().encode(
        x='ds:T',
        y='yhat:Q'
    ).properties(
        title='Sales and Test Forecast'
    )
    chart2 = alt.Chart(future).mark_line(color='red').encode(
        x='ds:T',
        y='yhat:Q'
    ).properties(
        title='Future Forecast'
    )

    st.altair_chart(chart1, use_container_width=True)
    st.altair_chart(chart2, use_container_width=True)

def xgboost_forecast():

    df = pd.read_csv('./ForecastingDashboard/salesorderheader.csv')  # Ensure this file is in your directory
    df = df[['OrderDate', 'SubTotal']]
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df.set_index('OrderDate', inplace=True)
    df = df.groupby('OrderDate').sum()
    train = df[:int(0.8 * len(df))]
    test = df[int(0.8 * len(df)):]

    def create_features(df,label= None):

        df['date'] = df.index
        df['dayofweek'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        df['dayofmonth'] = df['date'].dt.day
        X = df[['dayofweek','month','year','dayofyear','dayofmonth']]
        if label:
            y = df[label]
            return X,y
        return X
    
    X_train, y_train = create_features(train, label='SubTotal')
    X_test, y_test = create_features(test, label='SubTotal')


    model = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=1000,verbose= True)
    model.fit(X_train, y_train)   
    test['Prediction'] = model.predict(X_test)
    total = pd.concat([test, train], sort= False)
    future = model.predict(create_features(total, label=None).tail(30))

    chart1 = alt.Chart(test.reset_index()).mark_line().encode(
        x='OrderDate:T',
        y='Prediction:Q'
    ).properties(
        title='Sales and Test Forecast'
    )

    chart2 = alt.Chart(pd.DataFrame(future)).mark_line(color='red').encode(
        x=alt.X('index:T', title='Date'),
        y=alt.Y(0, title='Forecast')
    ).properties(
        title='Future Forecast'
    )

    st.altair_chart(chart1, use_container_width=True)
    st.altair_chart(chart2, use_container_width=True)
    
def lgbm():

    df = pd.read_csv('./ForecastingDashboard/salesorderheader.csv')
    df = df[['OrderDate', 'SubTotal']]
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df.set_index('OrderDate', inplace=True)
    df = df.groupby('OrderDate').sum()

    train = df[:int(0.8 * len(df))]
    test = df[int(0.8 * len(df)):]

    x_train = train.index.values.reshape(-1, 1)
    y_train = train['SubTotal'].values
    x_test = test.index.values.reshape(-1, 1)
    y_test = test['SubTotal'].values

    model = lgb.LGBMRegressor()
    model.fit(x_train, y_train)
    forecast = model.predict(x_test)
    future = model.predict(np.arange(len(df), len(df) + 30).reshape(-1, 1))

    forecast_df = pd.DataFrame(forecast, index=test.index, columns=['Forecast'])
    future_df = pd.DataFrame(future, index=pd.date_range(start=df.index[-1], periods=30), columns=['Forecast'])

    chart1 = alt.Chart(forecast_df.reset_index()).mark_line().encode(
        x='index:T',
        y='Forecast:Q'
    ).properties(
        title='Sales and Test Forecast'
    )

    chart2 = alt.Chart(future_df.reset_index()).mark_line(color='red').encode(
        x='index:T',
        y='Forecast:Q'
    ).properties(
        title='Future Forecast'
    )

    st.altair_chart(chart1, use_container_width=True)
    st.altair_chart(chart2, use_container_width=True)


page = st_navbar(["SARIMAX", "Prophet", "XGBoost", "LGBM"])
if page == "SARIMAX":
    sarima_forecast()
if page == "Prophet":
    prophet_forecast()
