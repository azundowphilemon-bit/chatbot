import streamlit as st
from dotenv import load_dotenv
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")  # or your API key variable

st.title("Azundow Python Tutor Chatbot")

# ------------------------------
# 1. Load and split documents
# ------------------------------
# Example: Load a single text file
loader = TextLoader("docs/python_tutorial.txt", encoding="utf-8")
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)
splits = text_splitter.split_documents(documents)

# ------------------------------
# 2. Setup embeddings
# ------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # prevents meta tensor errors
)

# ------------------------------
# 3. Setup Chroma vector store
# ------------------------------
vector_store = Chroma(
    collection_name="azundow_collection",
    embedding_function=embeddings,
    persist_directory=None  # in-memory; no tenant/db needed
)
vector_store.add_documents(splits)

# ------------------------------
# 4. Setup LLM with prompt
# ------------------------------
prompt_template = """You are a helpful Python tutor.
Answer the user's questions clearly and provide example code if needed."""

prompt = ChatPromptTemplate.from_template(prompt_template)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0
)

# Create conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever()
)

# ------------------------------
# 5. Streamlit UI
# ------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a Python question:")

if user_input:
    result = qa_chain({"question": user_input, "chat_history": st.sessi_









