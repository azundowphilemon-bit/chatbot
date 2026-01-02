import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# MUST BE FIRST — Streamlit rule!
st.set_page_config(page_title="Azundow Intelligent Document Chatbot", page_icon="🤖", layout="centered")

# Get API key from Streamlit Secrets (no .env!)
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("Groq API key not found. Add it in Streamlit Cloud → Settings → Secrets")
    st.stop()

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

# Load documents with cache
@st.cache_resource
def load_chain():
    documents_folder = "documents"
    docs = []

    if os.path.exists(documents_folder):
        files = [f for f in os.listdir(documents_folder) if f.lower().endswith(('.pdf', '.csv'))]
        if files:
            for filename in files:
                file_path = os.path.join(documents_folder, filename)
                if filename.lower().endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif filename.lower().endswith(".csv"):
                    loader = CSVLoader(file_path)
                docs.extend(loader.load())

    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
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
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# Load chain
chain = load_chain()
if chain:
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
            if chain:
                response = chain.invoke(prompt)
            else:
                response = "I can help with general Python questions!"
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.caption("Azundow Intelligent Document Chatbot — Fast • Professional")
