import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from graph import graph

st.set_page_config(page_title="MedQuery", page_icon="🏥")
st.title("MedQuery")
thread={"configurable":{"thread_id":"123"}}

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi I am your Medical Query Assistant! How can I help you?"}
    ]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.chat_input("Ask me your query")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    
    response=graph.invoke({"question":question},config=thread)                              # ← no config
    final_response = response["generation"]
    st.chat_message("assistant").write(final_response)

    st.session_state.messages.append({"role": "assistant", "content": final_response})