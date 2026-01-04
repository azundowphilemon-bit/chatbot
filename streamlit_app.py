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
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load .env
load_dotenv()

# API key
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("Groq API key not found.")
    st.stop()

# Page config
st.set_page_config(page_title="Azundow Intelligent Document Chatbot", page_icon="🤖", layout="centered")

# Title
col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo.png", width=100)
with col2:
    st.markdown("<h1 style='margin-top: 30px;'>Azundow Intelligent Document Chatbot</h1>", unsafe_allow_html=True)

st.caption("Built by Azundow — Ask questions on Python")

# Session state for messages and chain
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain_with_history" not in st.session_state:
    st.session_state.chain_with_history = None
if "store" not in st.session_state:
    st.session_state.store = {}  # for session history

# Build the RAG chain with memory
if st.session_state.chain_with_history is None:
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

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

    if docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vector_store = Chroma(collection_name="azundow_collection", embedding_function=embeddings, persist_directory=None)
        vector_store.add_documents(splits)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        st.success("Documents loaded — ready!")
    else:
        retriever = None
        st.info("No documents loaded — general Python help available")

    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", temperature=0.3)

    system_prompt = """
You are Azundow, a kind and patient Python teacher for complete beginners.

STRICT RULES:
- Use only simple words.
- Never use advanced terms unless asked.
- Be encouraging.
- Remember the full conversation and refer to previous questions when helpful.
- Use context only if simple and relevant.

Context: {context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])

    # Base chain
    if retriever:
        base_chain = (
            {"context": retriever, "question": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    else:
        base_chain = (
            {"question": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
            | prompt.format(context="")
            | llm
            | StrOutputParser()
        )

    # Add memory using RunnableWithMessageHistory
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ConversationBufferMemory(return_messages=True)
        return st.session_state.store[session_id]

    st.session_state.chain_with_history = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

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
            response = st.session_state.chain_with_history.invoke(
                {"question": prompt},
                config={"configurable": {"session_id": "azundow_session"}}
            )
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.caption("Azundow Intelligent Document Chatbot — Fast • Professional")









