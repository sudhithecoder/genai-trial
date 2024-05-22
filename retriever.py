import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


loader = WebBaseLoader(
    web_paths=("https://www.hsbc.co.in/help/faqs/online-banking/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_="cc-wrapper O-COLCTRL-RW-DEV")),
)
text_documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(text_documents)


db = Chroma.from_documents(documents, OpenAIEmbeddings())

model = ChatOpenAI()


prompt = ChatPromptTemplate.from_template(
    """
        Answer the following question based only on the provided context. 
        For any out of context questions, please provide a generic answer.
        Think step by step before providing a detailed answer.
        Also please provide step by step instructions.
        <context>
        {context}
        </context>
        Question: {input}
    """
)


document_chain = create_stuff_documents_chain(model, prompt)

retriever = db.as_retriever()


retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "Can use mobile"})
# print(response["answer"])
