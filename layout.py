import streamlit as st

first, last = st.columns(2)
first.text_input("First Name")
last.text_input("Last Name")

email, age = st.columns([3,1])
email.text_input("Email")
age.text_input("Age")

pas, cpas = st.columns(2)
pas.text_input("Password", type = "password")
cpas.text_input("Confirm Password", type = "password")

cbox, button = st.columns(2)
cbox.checkbox("I agree")
button.button("Submit")