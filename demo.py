import streamlit as st
import pandas as pd
import numpy as np
import time

dic = {"Name": "Divyansh", "Age": 21, "Profession": "Student"}
df = pd.read_csv("salesterritory.csv")
st.write(dic)
st.dataframe(dic)
st.table(dic)
st.write(df)
st.dataframe(df)
st.table(df)

@st.cache_data
def ret_time(num):
    time.sleep(5)
    return time.time()

if st.checkbox("1"):
    st.write(ret_time(1))

if st.checkbox("2"):
    st.write(ret_time(2))

if st.checkbox("3"):
    st.write(ret_time(1))