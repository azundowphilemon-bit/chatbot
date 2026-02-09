import streamlit as st
import os
from dotenv import load_dotenv
import auth  # Import the auth module
import topics # Import topics helper

from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ────────────────────────────────────────────────
# Load environment variables
# ────────────────────────────────────────────────
load_dotenv()

api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

# ────────────────────────────────────────────────
# Page config — MUST BE FIRST
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Azundow Intelligent Document Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ────────────────────────────────────────────────
# Auth & Session State Initialization
# ────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "current_topic_index" not in st.session_state:
    st.session_state.current_topic_index = 0

# Initialize chat session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None

# Get Topics
ALL_TOPICS = topics.get_topics()

# ────────────────────────────────────────────────
# Application Flow
# ────────────────────────────────────────────────

# If NOT logged in, show Login/Register
if not st.session_state.logged_in:
    # Header for Login Screen
    col1, col2 = st.columns([1, 5])
    with col1:
        if os.path.exists("logo.png"):
            st.image("logo.png", width=80)
        else:
            st.write("🤖")
    with col2:
        st.markdown("<h2 style='margin-top: 20px;'>Welcome to Azundo Intelligent Chatbot</h2>", unsafe_allow_html=True)
    
    st.info("Please Login or Register to continue your Python learning journey.")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Login")

            if submit_login:
                # Login returns (username, password, name, progress_index)
                user = auth.login_user(username, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.username = user[0][0]
                    st.session_state.user_name = user[0][2]
                    st.session_state.current_topic_index = user[0][3]
                    st.success(f"Welcome back, {st.session_state.user_name}!")
                    st.session_state.messages = [] # Clear messages on new login to restart context if needed
                    st.session_state.chain = None # Rebuild chain with new context
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    with tab2:
        with st.form("register_form"):
            new_user = st.text_input("Username")
            new_name = st.text_input("Full Name")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit_register = st.form_submit_button("Register")

            if submit_register:
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif len(new_password) < 4:
                    st.error("Password must be at least 4 characters.")
                else:
                    if auth.register_user(new_user, new_password, new_name):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Username already exists.")

    st.stop()  # Stop execution here if not logged in

# ────────────────────────────────────────────────
# MAIN APP (Only runs if Logged In)
# ────────────────────────────────────────────────

# Current Topic Logic
current_topic_name = "Introduction"
if 0 <= st.session_state.current_topic_index < len(ALL_TOPICS):
    current_topic_name = ALL_TOPICS[st.session_state.current_topic_index]
    
next_topic_name = "None"
if st.session_state.current_topic_index + 1 < len(ALL_TOPICS):
    next_topic_name = ALL_TOPICS[st.session_state.current_topic_index + 1]

# Sidebar for User Info & Logout
with st.sidebar:
    st.write(f"👤 **{st.session_state.user_name}**")
    st.write(f"📚 **Current Topic:** {current_topic_name}")
    st.progress(min(1.0, (st.session_state.current_topic_index + 1) / len(ALL_TOPICS)))
    
    if st.button("Mark Topic as Complete & Next"):
        new_index = st.session_state.current_topic_index + 1
        if new_index < len(ALL_TOPICS):
            if auth.update_progress(st.session_state.username, new_index):
                st.session_state.current_topic_index = new_index
                st.session_state.chain = None # Rebuild chain for new topic context
                # Add a system notification in chat
                st.session_state.messages.append({"role": "assistant", "content": f"Great job! Moving on to **{ALL_TOPICS[new_index]}**."})
                st.rerun()
            else:
                st.error("Failed to update progress (DB Error).")
        else:
            st.success("You have completed the tutorial!")

    st.markdown("---")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.user_name = None
        st.session_state.messages = [] # Clear history on logout
        st.session_state.chain = None
        st.rerun()

# API Key Check
if not api_key:
    st.error("Groq API key not found.")
    st.info("Local → create .env file\nOnline → add in Streamlit Cloud → Settings → Secrets")
    st.stop()

# Header
col1, col2 = st.columns([1, 5])
with col1:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=100)
    else:
        st.write("🤖")
with col2:
    st.markdown("<h1 style='margin-top: 30px;'>Azundow Intelligent Chatbot</h1>", unsafe_allow_html=True)

st.caption(f"Welcome, {st.session_state.user_name}! Learning: **{current_topic_name}**")

# ────────────────────────────────────────────────
# Build RAG chain (runs once per session or if chain is missing)
# ────────────────────────────────────────────────
if st.session_state.chain is None:
    # Initialize messages for new session
    if not st.session_state.messages:
        initial_greeting = (
            f"Hello {st.session_state.user_name}! We are currently on **{current_topic_name}**. "
            "Shall I explain this topic to you?"
        )
        st.session_state.messages.append({"role": "assistant", "content": initial_greeting})

    docs = []
    folder = "documents"

    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.pdf', '.csv', '.md'))]
        if files:
            for f in files:
                path = os.path.join(folder, f)
                try:
                    if f.lower().endswith(".pdf"):
                        loader = PyPDFLoader(path)
                        docs.extend(loader.load())
                    elif f.lower().endswith(".csv"):
                        loader = CSVLoader(path)
                        docs.extend(loader.load())
                    elif f.lower().endswith(".md"):
                         # Simple text loader for MD is enough or we use langchain's UnstructuredMarkdownLoader
                         # For simplicity/speed without extra deps, let's just read it as text and make a Document
                         from langchain_core.documents import Document
                         with open(path, "r", encoding="utf-8") as f_md:
                             docs.append(Document(page_content=f_md.read(), metadata={"source": f}))
                except Exception as e:
                    st.warning(f"Could not load {f}: {e}")

    # Fallback if no docs
    context_part = {"context": lambda x: "No documents.", "question": RunnablePassthrough()}
    
    if docs:
        with st.spinner("Building knowledge base…"):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )

            vector_store = Chroma(
                collection_name="azundow_collection",
                embedding_function=embeddings,
                persist_directory=None
            )
            vector_store.add_documents(splits)
            retriever = vector_store.as_retriever(search_kwargs={"k": 4})
            context_part = {"context": retriever, "question": RunnablePassthrough()}
            
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.3
    )

    # ────────────────────────────────────────────────
    # UPDATED SYSTEM PROMPT (Personalized + Proactive Progress)
    # ────────────────────────────────────────────────
    prompt = ChatPromptTemplate.from_template(
        f"""You are Azundo, a helpful and patient Python tutor.
The user's name is {st.session_state.user_name}.
The user is currently learning the topic: "{current_topic_name}".
The NEXT topic is: "{next_topic_name}".

YOUR INSTRUCTIONS:
1. Explain the main concept of "{current_topic_name}" clearly.
2. Provide a short, practical Python code example for "{current_topic_name}".
3. CRITICAL: Do NOT wait for the user to ask for the next topic.
4. Once you have provided the explanation and example, IMMEDIATEY ask: "That covers {current_topic_name}. The next topic is {next_topic_name}. Shall we proceed?"
5. Use only the context below if available.
6. Answer in your own words.

Context: {{context}}
Question: {{question}}
Answer:"""
    )

    st.session_state.chain = (
        context_part
        | prompt
        | llm
        | StrOutputParser()
    )


# ────────────────────────────────────────────────
# Chat interface
# ────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Python..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            if st.session_state.chain:
                try:
                    response = st.session_state.chain.invoke(prompt)
                except Exception as e:
                    response = f"Sorry — temporary error: {e}"
            else:
                response = "I can help with general Python questions!"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.caption("Azundow Intelligent Chatbot — Fast • Professional")








