from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import yaml
import bs4
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

# Load configuration and environment variables
load_dotenv()
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Load documents
loader = WebBaseLoader(
    web_paths=config["web_paths"],
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=config["bs_kwargs"]["parse_only"]))
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
        ("human", "{input}")
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
        ("human", "{input}")
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

# Define FastAPI app
app = FastAPI()

# Define request and response models
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

# Define API endpoint
@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    try:
        answer = chatbot.ask_question(request.question)
        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
