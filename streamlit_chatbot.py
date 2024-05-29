import os
import yaml
import bs4
import streamlit as st
from langchain import hub
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

load_dotenv()

# Load configuration file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Load documents
loader = WebBaseLoader(
    web_paths=config["web_paths"],
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=config["bs_kwargs"]["parse_only"])),
)
text_documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config["text_splitter"]["chunk_size"], 
    chunk_overlap=config["text_splitter"]["chunk_overlap"]
)
documents = text_splitter.split_documents(text_documents)

# Create vector store
db = Chroma.from_documents(documents, OpenAIEmbeddings())

# Initialize model
model = ChatOpenAI()

# Create retriever
retriever = db.as_retriever()

# Load prompt from hub
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model=config["llm"]["model"], temperature=config["llm"]["temperature"])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know with an apology. \
Use three sentences maximum and keep the answer concise.\

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

class ChatBot:
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain
        self.chat_history = []

    def ask_question(self, question):
        ai_msg = self.rag_chain.invoke({"input": question, "chat_history": self.chat_history})
        self.chat_history.extend([HumanMessage(content=question), ai_msg["answer"]])
        return ai_msg["answer"]

# Initialize the chatbot with the RAG chain
chatbot = ChatBot(rag_chain)

# Streamlit app
# Set page configuration for a Google-like UI
st.set_page_config(
    page_title="HSBC Helper Chatbot",
    # layout="wide",  # Maximize screen width for a spacious layout
    initial_sidebar_state="collapsed",  # Hide sidebar for a cleaner look
    page_icon="https://www.hsbc.co.in/content/dam/hsbc/in/images/01_HSBC_MASTERBRAND_LOGO_RGB.svg",
)

# Display the logo without background
st.markdown(
    """
    <style>
    img {
        # background-color: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display the logo
st.image("https://www.hsbc.co.in/content/dam/hsbc/in/images/01_HSBC_MASTERBRAND_LOGO_RGB.svg", width=100)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_answer(question):
    answer = chatbot.ask_question(question)
    st.session_state.chat_history.append({"question": question, "answer": answer})
    return answer

def write_chat_history():
    for entry in st.session_state.chat_history:
        st.markdown(f"""
        <div style="display: flex; flex-direction: column; margin-bottom: 20px;">
            <div style="display: flex; justify-content: flex-start; align-items: flex-start;">
                <div style="background-color: #f8f8f8; border-radius: 5px; padding: 10px; max-width: 70%; margin-right: 10px;">
                    <strong>You:</strong> {entry['question']}
                </div>
            </div>
            <div style="display: flex; justify-content: flex-end; align-items: flex-start;">
                <div style="background-color: #fcfcfc ; border-radius: 5px; padding: 10px; max-width: 70%; margin-left: 10px;">
                    <strong>HSBC Assistant:</strong> {entry['answer']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Input and button
question = st.chat_input(placeholder="Enter your query here")

if question:
    answer = get_answer(question)
    # st.write(f"**Answer:** {answer}")
    write_chat_history()
