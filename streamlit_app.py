import streamlit as st
import os
import shutil
from dotenv import load_dotenv

# --- Fix old sqlite3 on Streamlit Cloud (safe to keep) ---
try:
    import pysqlite3 as sqlite3
    import sys
    sys.modules["sqlite3"] = sqlite3
except ImportError:
    pass

import chromadb
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (local only)
load_dotenv()

# Get Groq API key
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("Groq API key not found.")
    st.info("Local: add to .env file\nOnline: add in Streamlit Cloud → Settings → Secrets")
    st.stop()

# Page config — must be first
st.set_page_config(page_title="Azundow Intelligent Document Chatbot", page_icon="🤖", layout="centered")

# Header
col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo.png", width=100)
with col2:
    st.markdown("<h1 style='margin-top: 30px;'>Azundow Intelligent Document Chatbot</h1>", unsafe_allow_html=True)
st.caption("Built by Azundow — Ask questions on Python")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False

# --- Critical fix: Clear any stale Chroma cache ---
chromadb.api.client.SharedSystemClient.clear_system_cache()

# --- Clean persistent directory on every fresh deploy ---
persist_dir = "./chroma_db"
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
os.makedirs(persist_dir, exist_ok=True)

@st.cache_resource(show_spinner="Loading documents and building vector index...")
def build_rag_chain(_api_key):
    documents_folder = "documents"
    docs = []

    # Load documents from folder
    if os.path.exists(documents_folder):
        files = [f for f in os.listdir(documents_folder) if f.lower().endswith(('.pdf', '.csv'))]
        if not files:
            return None, "No documents found in 'documents' folder — general chat mode"
        for filename in files:
            file_path = os.path.join(documents_folder, filename)
            ext = filename.lower().split(".")[-1]
            loader = PyPDFLoader(file_path) if ext == "pdf" else CSVLoader(file_path)
            docs.extend(loader.load())
    else:
        return None, "No 'documents' folder found — general chat mode"

    if not docs:
        return None, "No documents loaded — general Python help available"

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # --- Use PersistentClient directly (this is the key to stability) ---
    chroma_client = chromadb.PersistentClient(
        path=persist_dir,
        settings=chromadb.config.Settings(
            chroma_db_impl="duckdb+parquet",  # Reliable storage backend
            anonymized_telemetry=False,
        ),
    )

    # Get or create collection
    collection = chroma_client.get_or_create_collection(name="azundow_collection")

    # Wrap with LangChain Chroma
    vector_store = Chroma(
        client=chroma_client,
        collection_name="azundow_collection",
        embedding_function=embeddings,
    )

    # Add documents only if collection is empty (fresh start)
    if collection.count() == 0:
        vector_store.add_documents(splits)

    # Build the chain
    llm = ChatGroq(groq_api_key=_api_key, model_name="llama-3.1-8b-instant", temperature=0.3)

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful Python tutor.
        Use only the provided context to answer.
        Be clear, friendly, and accurate.

        Context: {context}
        Question: {question}
        Answer:"""
    )

   









