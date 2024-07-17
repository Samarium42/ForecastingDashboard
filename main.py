import streamlit as st
import pandas as pd
st.title("New App")
df = pd.DataFrame([[1,2],[4,5]], columns = ["ID", "ID2"], index=[0,1])
st.write("Hello World", df)
st.header("Header with a :rainbow[divider]", divider= 'rainbow')
st.subheader(":red[Subheader] :sunglasses: _italics_", divider = "blue")
st.markdown(""" # h1 tag
## h2 tag
### h3 tag 
:moon: <br> 
:sunflower:""", True)
st.latex(r''' a + ar + a r^2 \cdots + a r^{n-1} ''')
