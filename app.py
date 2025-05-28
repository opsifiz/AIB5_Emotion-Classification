import streamlit as st

st.title("Mental Health Analysis Sentiment!")
# st.text("Deploy Ahh moment")
st.text("NLP Machine Learning from AIB2025")
txt = st.text_input("Input your Text: ")

if txt != "":
    st.write(f"Analysis \"{txt}\"...")
else:
    st.write("No message to analyze ~ _ ~")