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

# Load .env file
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Groq API key not found. Please check your .env file in the main project folder.")
    st.info("The .env file should contain: GROQ_API_KEY=your_key_here")
    st.stop()

# Page config — MUST BE FIRST
st.set_page_config(page_title="Azundow Intelligent Document Chatbot", page_icon="🤖", layout="centered")

# Title with logo
col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo.png", width=100)
with col2:
    st.markdown("<h1 style='margin-top: 30px;'>Azundow Intelligent Document Chatbot</h1>", unsafe_allow_html=True)

st.caption("Built by Azundow — Ask questions on Python")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None

# Load documents from "documents" folder
if st.session_state.chain is None:
    documents_folder = "documents"
    docs = []

    if os.path.exists(documents_folder):
        files = [f for f in os.listdir(documents_folder) if f.lower().endswith(('.pdf', '.csv'))]
        if files:
            for filename in files:
                file_path = os.path.join(documents_folder, filename)
                ext = filename.lower().split(".")[-1]
                if ext == "pdf":
                    loader = PyPDFLoader(file_path)
                elif ext == "csv":
                    loader = CSVLoader(file_path)
                docs.extend(loader.load())
        else:
            st.info("No documents found in 'documents' folder — general chat mode")
    else:
        st.info("No 'documents' folder found — general chat mode")

    if docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        # FIX: Force CPU to avoid meta tensor error on Streamlit Cloud
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", temperature=0.3)
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful Python tutor.
            Use only the context below.
            Answer in your own words.
            Be clear and friendly.
            Context: {context}
            Question: {question}
            Answer:"""
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        st.session_state.chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        st.success("Documents loaded — ready!")
    else:
        st.info("No documents loaded — general Python help available")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if st.session_state.chain:
                response = st.session_state.chain.invoke(prompt)
            else:
                response = "I can help with general Python questions!"
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.caption("Azundow Intelligent Document Chatbot — Fast • Professional")
