import streamlit as st
import retriever

st.set_page_config(
    page_title="HSBC Helper Chatbot",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.title("HSBC Helper Chatbot")
input_text = st.text_input("Welcome to HSBC Chatbot. Please enter your query:")


if input_text:
    response = retriever.retrieval_chain.invoke({"input": input_text})
    st.write(response["answer"])
