import streamlit as st
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from typing import TypedDict
import tempfile
import os

# API Key Configuration
genai.configure(api_key="AIzaSyAUosghuWPO6hSDsPRWZkHvfnncPcGCcQk")

# Embeddings for RAG
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

embedding_model = load_embeddings()

# Streamlit Page Config
st.set_page_config(page_title="Gemini Chat", layout="wide")
st.title("Chat Assistant + PDF RAG Agent")

# Session State Initialization
if "gemini_model" not in st.session_state:
    st.session_state["gemini_model"] = "models/gemini-2.5-flash"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat" not in st.session_state:
    model = genai.GenerativeModel(st.session_state["gemini_model"])
    st.session_state.chat = model.start_chat(history=[])

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# PDF Upload & Indexing
uploaded_pdf = st.sidebar.file_uploader("Upload PDF for knowledge base", type=["pdf"])

if uploaded_pdf is not None:
    with st.sidebar:
        with st.spinner("Processing PDF..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_pdf.read())
                    tmp_path = tmp_file.name
                
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=100
                )
                chunks = splitter.split_documents(docs)
                
                st.session_state.vectorstore = FAISS.from_documents(
                    chunks, 
                    embedding_model
                )
                
                os.unlink(tmp_path)
                
                st.sidebar.success(f"PDF indexed with {len(chunks)} chunks")
                
            except Exception as e:
                st.sidebar.error(f"Error processing PDF: {str(e)}")

# Define State Schema
class AgentState(TypedDict):
    input: str
    context: str
    output: str

# Define LangGraph Nodes
def retrieve(state: AgentState) -> AgentState:
    query = state["input"]
    if st.session_state.vectorstore:
        try:
            results = st.session_state.vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([r.page_content for r in results])
            state["context"] = context
        except Exception as e:
            state["context"] = f"Error retrieving context: {str(e)}"
    else:
        state["context"] = ""
    return state

def generate(state: AgentState) -> AgentState:
    try:
        model = genai.GenerativeModel(st.session_state["gemini_model"])
        
        if state["context"]:
            prompt = f"""You are a helpful AI assistant. Use the context below to answer the question.

Context from PDF:
{state['context']}

Question: {state['input']}

Answer based on the context above. If the context doesn't contain relevant information, say so and provide a general answer."""
        else:
            prompt = state["input"]
        
        response = model.generate_content(prompt)
        state["output"] = response.text
        
    except Exception as e:
        state["output"] = f"Error generating response: {str(e)}"
    
    return state

# Build LangGraph Pipeline
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

agent = workflow.compile()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = agent.invoke({
                    "input": prompt,
                    "context": "",
                    "output": ""
                })
                full_response = result["output"]
                st.markdown(full_response)
                
            except Exception as e:
                full_response = f"Error: {str(e)}"
                st.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar Settings
with st.sidebar:
    st.header("Settings")
    
    model_option = st.selectbox(
        "Choose Model",
        ["models/gemini-2.5-flash","models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-2.0-flash-exp"],
        index=0,
        help="Flash is faster, Pro is more capable"
    )

    if model_option != st.session_state["gemini_model"]:
        st.session_state["gemini_model"] = model_option
        model = genai.GenerativeModel(st.session_state["gemini_model"])
        st.session_state.chat = model.start_chat(history=[])
        st.success(f"Switched to {model_option}")

    st.divider()

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat = genai.GenerativeModel(
            st.session_state["gemini_model"]
        ).start_chat(history=[])
        st.rerun()

    st.divider()
    
    st.caption("Powered by:")
    st.caption("Google Gemini AI")
    st.caption("LangGraph for workflow")
    st.caption("FAISS for vector search")
    st.caption(f"Model: {st.session_state['gemini_model']}")
    st.caption(f"Messages: {len(st.session_state.messages)}")
    
    if st.session_state.vectorstore:
        st.caption("PDF knowledge base: Active")
    else:
        st.caption("PDF knowledge base: No PDF uploaded")

    st.divider()

    with st.expander("Tips & Examples"):
        st.markdown("""
        Without PDF:
        - Explain quantum computing
        - Write a Python function
        - Help me with coding
        
        With PDF uploaded:
        - Summarize the document
        - What does it say about [topic]?
        - Extract key points from the PDF

        Features:
        - PDF Retrieval-Augmented Generation
        - Real-time streaming responses
        - Context-aware from documents
        - LangGraph workflow orchestration
        """)

    with st.expander("Get API Key"):
        st.markdown("""
        Get Google AI API Key:
        1. Visit: https://aistudio.google.com/apikey  
        2. Click Create API Key
        3. Copy and paste in code
        """)

    with st.expander("How RAG Works"):
        st.markdown("""
        Retrieval-Augmented Generation:
        
        1. Upload PDF → Document is split into chunks
        2. Create Embeddings → Text converted to vectors
        3. Store in FAISS → Vector database for fast search
        4. User asks question → Find relevant chunks
        5. Send to Gemini → Answer using context
        
        Helps AI answer questions about your documents
        """)

