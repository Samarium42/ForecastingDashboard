import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
import altair as alt
import graphviz

a = np.random.randn(100,3)
df = pd.DataFrame(a, columns = ["a","b","c"])

st.line_chart(df)
st.area_chart(df)
st.bar_chart(df)

plt.scatter(df["a"],df["b"])
plt.title("Scatter PLot")
st.pyplot()


st.graphviz_chart('''
    digraph{
        "a" -> "b"
        "b" -> "c"
        "c" -> "a"
    }
'''
)


st.map()

# st.image(url)
# st.audio(url)
# st.video(url)