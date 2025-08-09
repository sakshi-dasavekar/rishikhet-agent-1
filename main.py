# # streamlit app
# #!/usr/bin/env python3
# """
# Agricultural RAG System - Streamlit Web Interface

# A user-friendly web interface for the agricultural RAG system.
# """

# import streamlit as st
# import os
# import glob
# import json
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.retrievers import BaseRetriever

# # Page configuration
# st.set_page_config(
#     page_title="Agricultural Expert System",
#     page_icon="🌾",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         font-weight: bold;
#         color: #2E8B57;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .sub-header {
#         font-size: 1.5rem;
#         color: #556B2F;
#         margin-bottom: 1rem;
#     }
#     .info-box {
#         background-color: #F0F8FF;
#         padding: 1rem;
#         border-radius: 10px;
#         border-left: 5px solid #2E8B57;
#         margin: 1rem 0;
#     }
#     .dataset-info {
#         background-color: #F5F5F5;
#         padding: 0.5rem;
#         border-radius: 5px;
#         margin: 0.25rem 0;
#     }
#     .stButton > button {
#         background-color: #2E8B57;
#         color: white;
#         border-radius: 10px;
#         padding: 0.5rem 2rem;
#         font-weight: bold;
#     }
#     .stButton > button:hover {
#         background-color: #3CB371;
#     }
# </style>
# """, unsafe_allow_html=True)

# @st.cache_resource
# def load_rag_system():
#     """Load the RAG system with caching"""
    
#     # Load environment variables
#     load_dotenv(override=True)
#     print("GROQ_API_KEY loaded:", os.getenv("GROQ_API_KEY"))
#     groq_api_key = os.getenv("GROQ_API_KEY")
    
#     if not groq_api_key:
#         st.error("❌ Groq API key not found. Please create a .env file with your GROQ_API_KEY.")
#         return None
    
#     try:
#         # Load all available vector stores
#         all_vectorstores = []
#         dataset_folders = glob.glob("rag_storage_filtered/*")
#         embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
#         loaded_datasets = []
#         total_vectors = 0
        
#         for folder in dataset_folders:
#             dataset_name = folder.split('/')[-1]
#             embeddings_path = f"{folder}/embeddings"
            
#             if os.path.exists(embeddings_path):
#                 try:
#                     vectorstore = FAISS.load_local(embeddings_path, embeddings, allow_dangerous_deserialization=True)
#                     all_vectorstores.append((dataset_name, vectorstore))
#                     loaded_datasets.append(dataset_name)
#                     total_vectors += vectorstore.index.ntotal
#                 except Exception as e:
#                     st.warning(f"⚠️ Error loading {dataset_name}: {e}")
        
#         if not all_vectorstores:
#             st.error("❌ No vector stores could be loaded")
#             return None
        
#         # Create combined retriever
#         class CombinedRetriever(BaseRetriever):
#             vectorstores: list
            
#             def _get_relevant_documents(self, query, *, runnable_manager=None):
#                 all_docs = []
#                 for dataset_name, vectorstore in self.vectorstores:
#                     docs = vectorstore.similarity_search(query, k=5)
#                     for doc in docs:
#                         doc.metadata['dataset'] = dataset_name
#                     all_docs.extend(docs)
                
#                 # Sort by relevance and return top 5
#                 return all_docs[:5]
        
#         combined_retriever = CombinedRetriever(vectorstores=all_vectorstores)
        
#         # Initialize LLM
#         llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=groq_api_key)
        
#         # Create retrieval chain
#         agricultural_prompt_template = """
#         You are an expert agricultural advisor with deep knowledge of crop production, farming practices, disease management, and agricultural technologies. 
#         Answer the following question based only on the provided agricultural knowledge base:

#         Context:
#         {context}

#         Question: {input}

#         Provide a comprehensive, practical answer that includes:
#         - Specific recommendations based on the context
#         - Any relevant location-specific information (state/district)
#         - Seasonal considerations if mentioned
#         - Practical steps or solutions
#         - Mention which dataset the information comes from

#         Answer:
#         """
        
#         from langchain_core.prompts import ChatPromptTemplate
#         prompt = ChatPromptTemplate.from_template(agricultural_prompt_template)
#         document_chain = create_stuff_documents_chain(llm, prompt)
#         retrieval_chain = create_retrieval_chain(combined_retriever, document_chain)
        
#         return {
#             'chain': retrieval_chain,
#             'datasets': loaded_datasets,
#             'total_vectors': total_vectors
#         }
        
#     except Exception as e:
#         st.error(f"❌ Error loading RAG system: {e}")
#         return None

# def main():
#     """Main Streamlit application"""
    
#     # Header
#     st.markdown('<h1 class="main-header">🌾 Agricultural Expert System</h1>', unsafe_allow_html=True)
#     st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Your AI-powered agricultural advisor</p>', unsafe_allow_html=True)
    
#     # Load RAG system
#     with st.spinner("🔄 Loading Agricultural Expert System..."):
#         rag_system = load_rag_system()
    
#     if rag_system is None:
#         st.stop()
    
#     # Sidebar with system info
#     with st.sidebar:
#         st.markdown("## 📊 System Information")
        
#         # Dataset info
#         st.markdown("### 📁 Available Datasets")
#         for dataset in rag_system['datasets']:
#             st.markdown(f"• {dataset}")
        
#         st.markdown(f"### 🔢 Total Vectors: {rag_system['total_vectors']:,}")
        
#         # Sample questions
#         st.markdown("### 💡 Sample Questions")
#         sample_questions = [
#             "How to control Ranikhet disease in poultry?",
#             "How does the Pradhan Mantri Fasal Bima Yojana work for cotton farmers in Maharashtra?",
#             "What are the best practices for field preparation?",
#             "How to manage crop diseases effectively?",
#             "What are the recommended fertilizers for different crops?",
#             "How to prepare poultry feed at home?",
#             "How to grow tomatoes in Maharashtra?",
#             "How to rejuvenate old mango orchards with poor yields?",
#             "How to practice crop rotation effectively?",
#             "What are natural ways to control pests?",
#             "How to improve soil fertility organically?"
#         ]
        
#         for question in sample_questions:
#             if st.button(question, key=f"sample_{hash(question)}"):
#                 st.session_state.user_question = question
#                 st.rerun()
    
#     # Main content area
#     col1, col2, col3 = st.columns([1, 2, 1])
    
#     with col2:
#         # Question input
#         st.markdown('<h2 class="sub-header">🤔 Ask Your Agricultural Question</h2>', unsafe_allow_html=True)
        
#         # Initialize session state
#         if 'user_question' not in st.session_state:
#             st.session_state.user_question = ""
#         if 'chat_history' not in st.session_state:
#             st.session_state.chat_history = []
        
#         # Question input
#         user_question = st.text_area(
#             "Enter your agricultural question:",
#             value=st.session_state.user_question,
#             height=100,
#             placeholder="e.g., How to control Ranikhet disease in poultry?"
#         )
        
#         # Submit button
#         col_a, col_b, col_c = st.columns([1, 1, 1])
#         with col_b:
#             submit_button = st.button("🚀 Get Expert Answer", use_container_width=True)
        
#         # Process question
#         if submit_button and user_question.strip():
#             st.session_state.user_question = user_question
            
#             with st.spinner("🔍 Searching agricultural knowledge base..."):
#                 try:
#                     response = rag_system['chain'].invoke({"input": user_question})
#                     answer = response["answer"]
                    
#                     # Add to chat history
#                     st.session_state.chat_history.append({
#                         'question': user_question,
#                         'answer': answer
#                     })
                    
#                 except Exception as e:
#                     st.error(f"❌ Error getting answer: {e}")
#                     st.stop()
        
#         # Display chat history
#         if st.session_state.chat_history:
#             st.markdown('<h3 class="sub-header">💬 Conversation History</h3>', unsafe_allow_html=True)
            
#             for i, chat in enumerate(reversed(st.session_state.chat_history)):
#                 with st.expander(f"Q{i+1}: {chat['question'][:50]}...", expanded=True):
#                     st.markdown("**Question:**")
#                     st.write(chat['question'])
#                     st.markdown("**Answer:**")
#                     st.markdown(chat['answer'])
                    
#                     # Add copy button
#                     if st.button(f"📋 Copy Answer {i+1}", key=f"copy_{i}"):
#                         st.write("✅ Answer copied to clipboard!")
        
#         # Clear chat history
#         if st.session_state.chat_history and st.button("🗑️ Clear History"):
#             st.session_state.chat_history = []
#             st.rerun()
    
#     # Footer
#     st.markdown("---")
#     st.markdown(
#         """
#         <div style="text-align: center; color: #666; padding: 1rem;">
#             <p>🌾 Agricultural Expert System | Powered by LangChain & Groq</p>
#             <p>Built with ❤️ for farmers and agricultural professionals</p>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# if __name__ == "__main__":
#     main() 

#!/usr/bin/env python3
"""
FastAPI application for the Agricultural RAG System using Pinecone and HF Inference API
"""

import os
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# --- NEW IMPORTS ---
from fastapi.responses import HTMLResponse
from langchain_pinecone import PineconeVectorStore
# --- THIS IMPORT HAS CHANGED ---
from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings

# LangChain components
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# --- Pydantic Models (No change) ---
class Query(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str
    source_documents: List[dict]

# --- Initialize FastAPI App (No change) ---
app = FastAPI(
    title="Agricultural RAG API",
    description="An API for asking agricultural questions.",
    version="1.0.0"
)

# --- Add CORS Middleware (No change) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global variable to hold the RAG chain ---
retrieval_chain = None

# --- UPDATED Server Startup Logic ---
@app.on_event("startup")
async def startup_event():
    """
    Loads models and connects to the Pinecone vector store when the server starts.
    """
    global retrieval_chain
    print("🚀 SERVER STARTUP: Initializing services...")
    
    load_dotenv(override=True)
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    hf_token = os.getenv("HUGGINGFACE_API_TOKEN") # New required secret
    
    if not all([groq_api_key, pinecone_api_key, hf_token]):
        raise ValueError("❌ One or more API keys are missing from environment variables.")

    # --- 1. Initialize Embedding Model via API ---
    # This change solves the memory issue by not loading the model locally.
    print("   ➡️ Initializing embedding model via Hugging Face API...")
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token, model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # --- 2. Connect to Pinecone Vector Store (No change) ---
    print("   ➡️ Connecting to Pinecone vector store...")
    index_name = "rishikhet-agent"
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    
    retriever = vectorstore.as_retriever()

    # --- 3. Initialize LLM (No change) ---
    print("   ➡️ Initializing LLM...")
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=groq_api_key)
    
    # --- 4. Create Prompt and Chain (No change) ---
    prompt = ChatPromptTemplate.from_template("""
    You are an expert agricultural advisor with deep knowledge of crop production, farming practices, disease management, and agricultural technologies. 
    Answer the following question based only on the provided agricultural knowledge base:

    Context:
    {context}

    Question: {input}

    Provide a comprehensive, practical answer that includes:
    - Specific recommendations based on the context
    - Any relevant location-specific information (state/district)
    - Seasonal considerations if mentioned
    - Practical steps or solutions
    - Mention which dataset the information comes from

    Answer:
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    print(f"\n✅ RAG API is ready and running!")

# --- API Endpoints (No change) ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html not found.</h1>"

@app.post("/ask", response_model=Answer)
async def ask_question(query: Query):
    if not retrieval_chain:
        return {"error": "RAG chain not initialized. Check server logs."}
    print(f"🤔 Received question: {query.text}")
    response = await retrieval_chain.ainvoke({"input": query.text})
    source_documents = []
    if "context" in response and response["context"]:
        for doc in response["context"]:
            source_documents.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
    return {"answer": response["answer"], "source_documents": source_documents}

# --- To run the server directly (No change) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)