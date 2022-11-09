import random

import streamlit as st
from streamlit_chat import message

from src.model import Model
from src.search_engine import SearchEngine


@st.experimental_singleton
def get_search():
    preprocessed_data_dir = "data/indexed/questions_answers.json"
    embeddings_dir = "data/indexed/embeddings.json"

    model = Model("distiluse-base-multilingual-cased-v1")
    search = SearchEngine(embeddings_dir, preprocessed_data_dir, model)
    return search

if "history" not in st.session_state:
    st.session_state.history = []

if "nextID" not in st.session_state:
    st.session_state.nextID = "1"

if "search" not in st.session_state:
    st.session_state.search = get_search()

def update_id():
    st.session_state.nextID += "1"

def gen_answer(user_input: str) -> str:
    result = st.session_state.search.run(user_input)
    question = result["question"]
    answer = result["answer"]
    return f"{question} => {answer}"


user_input = st.text_input("Talk with bot: ")


if user_input:
    st.session_state.history.append({
        "message": user_input, 
        "is_user": True, 
        "key" : st.session_state.nextID
        })
    update_id()
    st.session_state["history"].append({
        "message": gen_answer(user_input),
        "key": st.session_state.nextID
    })
    update_id()
    user_input = None

for kwargs in reversed(st.session_state["history"]):
    message(**kwargs)
