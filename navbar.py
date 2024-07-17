import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
import streamlit as st
import time
st.set_option('deprecation.showPyplotGlobalUse', False)
data = {
    "num" : [x for x in range(1,11)],
    "square": [x**2 for x in range(1,11)],
    "twice": [x*2 for x in range(1,11)],
    "thrice": [x*3 for x in range(1,11)]
}
option = st.sidebar.radio("Navigation", ["Home", "About Us"])
if option == "Home":
    df = pd.DataFrame(data)
    st.dataframe(df)

    col= st.sidebar.selectbox("Select a column", df.columns)
    plt.plot(df["num"], df[col])
    st.pyplot()

if option == "About Us":
    st.write("Hello World")
    st.error("This is an error")
    st.success("Success")
    st.info("Info")
    st.exception(RuntimeError("Exception of run time error"))
    st.warning("Warning")
    
    progress = st.progress(0, "Progress Bar")
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i+1)