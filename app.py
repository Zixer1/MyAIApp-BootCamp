import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

st.title("My AI app")

with st.sidebar:
    st.header("Settings")
    name = st.text_input("Enter your name")
    mood = st.selectbox("What will your AI's mood be?", ["Happy", "Sad", "Angry"])
    creativity = st.slider("Creativity", 0.0, 1.0, 0.3)
    if st.button("Save"):
        st.write(f"Saved, your name is {name}, your mood is {mood}, and your creativity is {creativity}")

prompt = st.chat_input("Ask me something.. ")

if prompt:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GITHUB_TOKEN"),
    )
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        st.write(f"{r.choices[0].message.content}")