import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

st.title("My AI app")

with st.sidebar:
    st.header("Settings")
    name = st.text_input("Enter your name")
    creativity = st.slider("Creativity", 0.0, 1.0, 0.3)
    model = st.selectbox("Model", ["openai/gpt-oss-120b", "openai/gpt-oss-20b"])
    if st.button("Save"):
        st.write(f"Saved, your name is {name}, and your creativity is {creativity}")

prompt = st.chat_input("Ask something in here.. ")

if prompt:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",  # <-Check ZOOM chat for change
        api_key=os.getenv("GITHUB_TOKEN"),
    )
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
             r = client.chat.completions.create(
                model=model,
                temperature=creativity,
                messages=[{"role": "user", "content": prompt}]
             )
        st.write(f" {r}")