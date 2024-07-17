import streamlit as st

st.title("Widgets")
if st.button("Hello") == True:
    st.write("Hello there!")

name = st.text_input("Name:")
st.write(name)

address = st.text_area("Address:")
st.write(address)

date = st.date_input("Date:")
st.write(date)

time = st.time_input("Time:")
st.write(time)

st.checkbox("1")
st.checkbox("2")

st.radio("Colours", ["red","green", "blue"], index = 0)

st.selectbox("Colours", ["red","green", "blue"], index = 0)

st.multiselect("Colours", ["red","green", "blue"])

st.slider("Age", min_value = 18, max_value = 60, value = 25, step = 2)

st.number_input("Insert a number: ", step = 2)

st.file_uploader("Upload")