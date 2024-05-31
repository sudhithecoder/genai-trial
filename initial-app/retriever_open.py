import osfrom dotenv import load_dotenvfrom langchain_community.document_loaders import WebBaseLoaderfrom langchain_text_splitters import RecursiveCharacterTextSplitterimport bs4
# from langchain_openai import OpenAIEmbeddings, ChatOpenAIfrom langchain_community.vectorstores import Chromafrom langchain_core.prompts import ChatPromptTemplatefrom langchain.chains.combine_documents import create_stuff_documents_chainfrom langchain.chains import create_retrieval_chainfrom langchain_community.llms import Ollamafrom sentence_transformers import SentenceTransformerfrom langchain.embeddings import HuggingFaceEmbeddingsfrom transformers import pipelinefrom transformers import AutoTokenizer, AutoModelForCausalLM
# import lama
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

loader = WebBaseLoader( web_paths=("https://www.hsbc.co.in/help/faqs/online-banking/",), bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_="cc-wrapper O-COLCTRL-RW-DEV")),)text_documents = loader.load()
# Load an open-source embedding model# model = SentenceTransformer("all-MiniLM-L6-v2")model_name = "sentence-transformers/all-mpnet-base-v2"

# Function to get embeddings from the SentenceTransformer model# embeddings = model.encode(sentences)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)documents = text_splitter.split_documents(text_documents)

# Create embeddings# embeddings = model.encode(documents)embeddings = HuggingFaceEmbeddings(model_name=model_name)
# Use Chroma to create a database from documents and their embeddingsdb = Chroma.from_documents(documents, embeddings)

# db = Chroma.from_documents(documents, OpenAIEmbeddings())

prompt = ChatPromptTemplate.from_template( """ Answer the following question based only on the provided context. For any out of context questions, please provide an answer "Please ask a relevant question". Do not mention about context in the answer Think step by step before providing a detailed answer. Also please provide step by step instructions. <context> {context} </context> Question: {input} """)# llm = Ollama(model="llama3")

pipe = pipeline("text-generation", model="openai-community/gpt2")

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
document_chain = create_stuff_documents_chain(model, prompt)
retriever = db.as_retriever()

retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": "Forgot password?"})print(response["answer"])
