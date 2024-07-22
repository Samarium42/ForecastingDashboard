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

st.set_page_config(layout="wide")
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

    # Generate future forecast
    
    future_forecast = model_fit.forecast(steps=60)
    future_forecast_df = np.array(future_forecast)
    future_forecast_df = pd.DataFrame(future_forecast_df, columns=['Forecast'])
    future_forecast_df['Date'] = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=60, freq='D')
    future_forecast_df.set_index('Date', inplace=True)

    combined_df = pd.concat([train, test, forecast_df])
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={'OrderDate': 'Date'}, inplace=True)

    future_forecast_df.reset_index(inplace=True)

    # Plot the data using Altair
    chart1 = alt.Chart(combined_df).mark_line().encode(
        x='Date:T',
        y='SubTotal:Q',
        color='Type:N'
    ).properties(
        title='Sales and Test Forecast'
    )

    chart2 = alt.Chart(future_forecast_df).mark_line(color='red').encode(
        x='Date:T',
        y='Forecast:Q'
    ).properties(
        title='Future Forecast'
    )

    st.altair_chart(chart1, use_container_width=True)
    st.altair_chart(chart2, use_container_width=True)

    # Ensure alignment
    if len(test) != len(forecast_df):
        st.error('Length mismatch between test set and forecast.')
        return

    test_aligned = test.loc[forecast_df.index]
    if len(test_aligned) != len(forecast_df):
        st.error('Alignment issue between test and forecast data.')
        return

    # Calculate the Mean Squared Percentage Error
    mean_squared_percentage_error = np.mean(np.abs(test_aligned['SubTotal'] - forecast_df['Forecast']) / test_aligned['SubTotal']) * 100
    st.error(f'Mean Absolute Percentage Error: {mean_squared_percentage_error:.2f}%')

def prophet_forecast():
    df = pd.read_csv('./ForecastingDashboard/salesorderheader.csv')
    df = df[['OrderDate', 'SubTotal']]
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df.set_index('OrderDate', inplace=True)
    df = df.groupby('OrderDate').sum()
    df.reset_index(inplace=True)
    df.rename(columns={'OrderDate': 'ds', 'SubTotal': 'y'}, inplace=True)

    train = df[:int(0.8 * len(df))]
    test = df[int(0.8 * len(df)):]

    model = Prophet()
    model.fit(train)

    # Make predictions for the test set
    test_forecast = model.predict(test[['ds']])

    # Make future predictions
    future = model.make_future_dataframe(periods=60)
    future_forecast = model.predict(future)

    # Prepare data for plotting
    test_forecast['Type'] = 'Test Forecast'
    future_forecast['Type'] = 'Future Forecast'
    train['Type'] = 'Train'

    combined_df = pd.concat([test_forecast[['ds', 'yhat', 'Type']], train[['ds', 'y', 'Type']].rename(columns={'y': 'yhat'})])

    chart1 = alt.Chart(combined_df).mark_line().encode(
        x='ds:T',
        y='yhat:Q',
        color='Type:N'
    ).properties(
        title='Prophet Model Sales Data Test'
    )

    chart2 = alt.Chart(future_forecast).mark_line(color='red').encode(
        x='ds:T',
        y='yhat:Q'
    ).properties(
        title='Prophet Model Future Forecast'
    )
    st.altair_chart(chart1, use_container_width=True)
    st.altair_chart(chart2, use_container_width=True)


    if len(test) != len(test_forecast):
        st.error('Length mismatch between test set and forecast.')
        return

    test_aligned = test.loc[test_forecast['ds'].values]
    if len(test_aligned) != len(test_forecast):
        st.error('Alignment issue between test and forecast data.')
        return

    mean_squared_percentage_error = np.mean(np.abs(test_aligned['y'] - test_forecast['yhat']) / test_aligned['y']) * 100
    st.error(f'Mean Absolute Percentage Error: {mean_squared_percentage_error:.2f}%')
    

def xgboost_forecast():
    df = pd.read_csv('./ForecastingDashboard/salesorderheader.csv')  # Ensure this file is in your directory
    df = df[['OrderDate', 'SubTotal']]
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df.set_index('OrderDate', inplace=True)
    df = df.groupby('OrderDate').sum()
    train = df[:int(0.8 * len(df))]
    test = df[int(0.8 * len(df)):]

    def create_features(df, label=None):
        df['date'] = df.index
        df['dayofweek'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        df['dayofmonth'] = df['date'].dt.day
        X = df[['dayofweek', 'month', 'year', 'dayofyear', 'dayofmonth']]
        if label:
            y = df[label]
            return X, y
        return X

    X_train, y_train = create_features(train, label='SubTotal')
    X_test, y_test = create_features(test, label='SubTotal')

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, verbose=True)
    model.fit(X_train, y_train)   
    test['Prediction'] = model.predict(X_test)
    total = pd.concat([test, train], sort=False)
    
    future_dates = pd.date_range(start=total.index.max() + pd.Timedelta(days=1), periods=60)
    future_features = create_features(pd.DataFrame(index=future_dates), label=None)
    future_forecast = model.predict(future_features)
    future = pd.DataFrame({
        'Date': future_dates,
        'Forecast': future_forecast
    })

    chart1 = alt.Chart(test.reset_index()).mark_line().encode(
        x='OrderDate:T',
        y='Prediction:Q'
    ).properties(
        title='Sales and Test Forecast'
    )

    chart2 = alt.Chart(future).mark_line(color='red').encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Forecast:Q', title='Forecast')
    ).properties(
        title='Future Forecast'
    )

    st.altair_chart(chart1, use_container_width=True)
    st.altair_chart(chart2, use_container_width=True)

    mean_squared_percentage_error = np.mean(np.abs(test['SubTotal'] - test['Prediction'])/test['SubTotal'])
    st.error(f'Mean Squared Percentage Error: {mean_squared_percentage_error}')

def lgbm():
    df = pd.read_csv('./ForecastingDashboard/salesorderheader.csv')
    df = df[['OrderDate', 'SubTotal']]
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df.set_index('OrderDate', inplace=True)
    df = df.groupby('OrderDate').sum()

    train = df[:int(0.8 * len(df))]
    test = df[int(0.8 * len(df)):]

    # Ensure the index is datetime for both train and test
    x_train = np.array([x.toordinal() for x in train.index]).reshape(-1, 1)
    y_train = train['SubTotal'].values
    x_test = np.array([x.toordinal() for x in test.index]).reshape(-1, 1)
    y_test = test['SubTotal'].values

    model = lgb.LGBMRegressor(objective='regression', n_estimators=1000)
    model.fit(x_train, y_train)
    forecast = model.predict(x_test)

    # Prepare future dates for prediction
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=60).to_pydatetime()
    future_ordinals = np.array([x.toordinal() for x in future_dates]).reshape(-1, 1)
    future_forecast = model.predict(future_ordinals)

    # Create DataFrames for the forecasts
    forecast_df = pd.DataFrame({'OrderDate': test.index, 'Forecast': forecast})
    future_df = pd.DataFrame({'OrderDate': future_dates, 'Forecast': future_forecast})

    # Combine the train, test, and forecast data for plotting the first chart
    train['Type'] = 'Train'
    test['Type'] = 'Test'
    forecast_df['Type'] = 'Forecast'
    combined_df = pd.concat([train.reset_index(), test.reset_index(), forecast_df])

    # Plot the train, test, and forecast data
    chart1 = alt.Chart(combined_df).mark_line().encode(
        x='OrderDate:T',
        y='SubTotal:Q',
        color='Type:N'
    ).properties(
        title='Sales and Test Forecast'
    )

    # Plot the future predictions
    chart2 = alt.Chart(future_df).mark_line(color='red').encode(
        x='OrderDate:T',
        y='Forecast:Q'
    ).properties(
        title='Future Forecast'
    )

    st.altair_chart(chart1, use_container_width=True)
    st.altair_chart(chart2, use_container_width=True)
    
    mean_squared_percentage_error = np.mean(np.abs(test['SubTotal'] - forecast)/test['SubTotal'])
    st.error(f'Mean Squared Percentage Error: {mean_squared_percentage_error}')


page = st_navbar(["SARIMAX", "Prophet", "XGBoost", "LGBM"])
if page == "SARIMAX":
    sarima_forecast()
if page == "Prophet":
    prophet_forecast()
if page == "XGBoost":
    xgboost_forecast()
if page == "LGBM":
    lgbm()
