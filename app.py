import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

import chromadb
from doc_helper import read_file

load_dotenv()

db = chromadb.PersistentClient(path="./chroma_db")
brain = db.get_or_create_collection("documents")
memory = db.get_or_create_collection("conversations")

def chunk_it(text, size=800):
    bits = text.split(". ")
    chunks, current = [], ""
    for bit in bits:
        if len(current) + len(bit) < size:
            current += bit + ". "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = bit + ". "
    if current.strip():
        chunks.append(current.strip())
    return chunks

def store_document(file):
    chunks = chunk_it(read_file(file))
    prefix = file.name.replace(" ", "_")
    brain.upsert(
        documents=chunks,
        ids=[f"{prefix}_{i}" for i in range(len(chunks))],
    )
    return len(chunks)

def store_conversation(question, answer):
    text = f"Q: {question}\nA: {answer}"
    chunks = chunk_it(text)
    turn = memory.count()
    memory.upsert(
        documents=[f"[past chat] {c}" for c in chunks],
        metadatas=[{"kind": "chat", "turn": turn} for c in chunks],
        ids=[f"turn{turn}_{i}" for i in range(len(chunks))],
    )
    return len(chunks)

st.title("ZeusAI")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    name = st.text_input("Enter your name")
    creativity = st.slider("Creativity", 0.0, 1.0, 0.3)
    message_history = st.slider("Message History", 1, 15, 5)
    recall = st.slider("Number of chunks for recall", 1, 10, 5)
    n_chunks = st.slider("Number of Chunks", 0, 15, 5)
    model = st.selectbox("Model", ["openai/gpt-oss-120b", "openai/gpt-oss-20b"])
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    if st.button("Clears all document history"):
        db.delete_collection("documents")
        st.rerun()
    if st.button("Clear all past chat history"):
        db.delete_collection("conversations")
        st.rerun()
    st.caption(f"{len(st.session_state.messages)} messages have been sent in this chat")
    st.caption(f"{brain.count()} chunks stored inside the chat")
    st.caption(f"{memory.count()} past conversation chunks stored")

SYSTEM_PROMPT = ("You are a greek god, you are all mighty and powerful. "
                 "You are wise, and know many thinks, Please answer using greek mythology and many puns. "
                 "Try to not discuss anything outside or greek mythology, but remain helpful if asked custom questions"
                 "Answer clearly, using relatively simple language so it is easy to read"
                 "ALl of the above are critical")

for old in st.session_state.messages:
    with st.chat_message(old["role"]):
        st.markdown(old["content"])

user_input = st.chat_input("Ask something here..", accept_file=True, file_type=["pdf", "txt"])

if user_input:
    prompt = user_input.text
    if user_input.files:
        with st.spinner(f"Processing {user_input.files[0].name}.."):
            n = store_document(user_input.files[0])
        st.success(f"Stored {n} new chunks inside of the chat, from {user_input.files[0].name}")

if user_input and prompt:
    st.session_state.messages.append({"role":"user", "content":prompt})
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GITHUB_TOKEN"),
    )
    with st.chat_message("user"):
        st.write(prompt)
    notes = ""
    if brain.count()>0:
        hits = brain.query(query_texts=[prompt], n_results=n_chunks)
        notes = "\n\n".join(hits["documents"][0])

        with st.expander("What I looked up"):
            for doc, dist, in zip(hits["documents"][0], hits["distances"][0]):
                st.text(f"{dist:.3f}, {doc[:70]}")
    recalled = ""
    if recall>0 and memory.count()>message_history:
        old = memory.query(query_texts=[prompt], n_results=recall)
        recalled = "\n\n".join(old["documents"][0])

        with st.expander("What I remembered from past conversations"):
            for doc, dist in zip(old["documents"][0], old["distances"][0]):
                st.text(f"{dist:.3f}, {doc[:70]}")

    if notes or recalled:
        full_prompt = (f"These are POTENTIALLY, relevant notes to the user's prompt, "
                       f"they might be irrelevant:\n {notes}\n\n"
                       f"These are POTENTIALLY, relevant past conversations, "
                       f"they might be irrelevant:\n {recalled}\n\n"
                       f"Now answer based on the above: {prompt}")
    else:
        full_prompt = prompt

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=model,
            temperature=creativity,
            messages=[ {"role":"system", "content":SYSTEM_PROMPT}]
                     + st.session_state.messages[-message_history:-1]
                     + [{"role":"user", "content":full_prompt}],
            stream=True,
        )
        thinking = st.expander("Thinking", expanded=True).empty()
        answer = st.empty()
        t = a = ""
        for chunk in stream:
            d = chunk.choices[0].delta
            if getattr(d, "reasoning", None):
                t += d.reasoning
                thinking.markdown(f"*{t}*")
            if d.content:
                a += d.content
                answer.markdown(a)
    st.session_state.messages.append({"role":"assistant", "content":a})
    store_conversation(prompt, a)