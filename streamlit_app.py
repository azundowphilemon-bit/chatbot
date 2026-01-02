import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# =========================
# Load .env file (local only)
# =========================
load_dotenv()

# Get Groq API key (local .env or Streamlit Secrets)
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("Groq API key not found.")
    st.info("Local: add to .env file\nOnline: add in Streamlit Secrets (Settings → Secrets)")
    st.stop()

# =========================
# Page configuration
# =========================
st.set_page_config(page_title="Azundow Intelligent Document Chatbot",
                   page_icon="🤖",
                   layout="centered")

# Title with logo
col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo.png", width=100)
with col2:
    st.markdown("<h1 style='margin-top: 30px;'>Azundow Intelligent Document Chatbot</h1>",
                unsafe_allow_html=True)
st.caption("Built by Azundow — Ask questions on Python")

# =========================
# Session state initialization
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = N_






