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

# Load .env file (local only)
load_dotenv()

# Get API key — works locally (.env) and online (Streamlit Secrets)
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not api_key:
    st.error("Groq API key not found.")
    st.info("Local: add to .env file\nOnline: add in Streamlit Cloud → Settings → Secrets")
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

# Load documents and build RAG chain
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
        
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        vector_store = Chroma(
            collection_name="azundow_collection",
            embedding_function=embeddings,
            persist_directory=None  # in-memory only
        )
        vector_store.add_documents(splits)
        
        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", temperature=0.3)

        # === NEW IMPROVED SYSTEM PROMPT ===
        system_prompt = """
You are Azundow, a friendly and patient Python teacher for complete beginners.

Rules you must always follow:
- Always give the simplest, shortest answer first.
- Use easy words. Never use advanced terms like "iterator", "exception", "StopIteration", "under the hood" unless the student asks.
- Always show working code in a code block.
- Explain like you are teaching a child who has never coded before.
- If the student asks for an example, give one small clear example.
- Do not repeat the same thing many times.
- Do not add extra details the student did not ask for.
- Always be encouraging and kind.

Use the context below only when it helps answer the question.
Context: {context}

Now answer the student's question clearly and kindly.
Question: {question}
Answer:
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])

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
        # Even without documents, we can still use the LLM for general help
        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", temperature=0.3)
        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt.format(context="")),  # same rules, no context
            ("human", "{question}")
        ])
        st.session_state.chain = simple_prompt | llm | StrOutputParser()

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
            try:
                response = st.session_state.chain.invoke(prompt)
            except Exception as e:
                response = f"Sorry, temporary error: {e}"
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.caption("Azundow Intelligent Document Chatbot — Fast • Professional")
   









