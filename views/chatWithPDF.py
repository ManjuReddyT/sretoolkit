import streamlit as st
import requests
import json
import sqlite3
import asyncio
import aiohttp
import threading
import time
from datetime import datetime
from pathlib import Path
import hashlib
import pickle
import base64
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
import io
import zipfile

# PDF Processing
from PyPDF2 import PdfReader
import fitz  # PyMuPDF for better text extraction and page viewing

# Enhanced imports
import pandas as pd
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced PDF Chat with Ollama",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark/light theme and better styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem 0;
    border-bottom: 2px solid #f0f2f6;
    margin-bottom: 2rem;
}

.chat-area {
    padding: 1.5rem;
    border: 1px solid #e6e6e6;
    border-radius: 10px;
    background-color: var(--background-color);
    height: 70vh;
    display: flex;
    flex-direction: column;
}

.chat-history {
    flex-grow: 1;
    overflow-y: auto;
    padding-right: 1rem; /* For scrollbar */
}

.chat-controls {
    padding-bottom: 1rem;
    border-bottom: 1px solid #e6e6e6;
    margin-bottom: 1rem;
}

.source-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 4px solid #007bff;
}

.document-card {
    background: #fff;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
}

.status-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-online { background-color: #28a745; }
.status-offline { background-color: #dc3545; }
.status-processing { background-color: #ffc107; }

/* Dark theme support */
@media (prefers-color-scheme: dark) {
    .chat-area {
        border: 1px solid #4a5568;
    }
    .document-card {
        background: #2d3748;
        color: #e2e8f0;
    }
    .source-card {
        background: #4a5568;
        color: #e2e8f0;
    }
}
</style>
""", unsafe_allow_html=True)

@dataclass
class Document:
    id: str
    name: str
    content: str
    chunks: List[str]
    embeddings: Optional[np.ndarray]
    summary: str
    upload_date: datetime
    page_count: int
    file_size: int

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: datetime
    sources: List[str] = field(default_factory=list)
    document_ids: List[str] = field(default_factory=list)

@dataclass
class ConfigProfile:
    name: str
    model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    context_chunks: int
    temperature: float

# Initialize session state with enhanced structure
def init_session_state():
    """Initializes session state with default values."""
    defaults = {
        'messages': [],
        'documents': {},
        'current_conversation_id': None,
        'conversations': {},
        'ollama_status': 'unknown',
        'processing_queue': queue.Queue(),
        'config_profiles': {},
        'current_profile': 'default',
        'dark_mode': False,
        'streaming_enabled': True,
        'auto_summarize': True,
        'last_user_prompt': None,
        'regenerate_response': False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

class DatabaseManager:
    def __init__(self, db_path: str = "pdf_chat.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                content TEXT,
                summary TEXT,
                upload_date TIMESTAMP,
                page_count INTEGER,
                file_size INTEGER,
                embeddings BLOB
            )
        """)
        
        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                name TEXT,
                created_date TIMESTAMP
            )
        """)
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP,
                sources TEXT,
                document_ids TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        """)
        
        # Config profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config_profiles (
                name TEXT PRIMARY KEY,
                model TEXT,
                embedding_model TEXT,
                chunk_size INTEGER,
                chunk_overlap INTEGER,
                context_chunks INTEGER,
                temperature REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_document(self, document: Document):
        """Save document to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embeddings_blob = pickle.dumps(document.embeddings) if document.embeddings is not None else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO documents 
            (id, name, content, summary, upload_date, page_count, file_size, embeddings)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            document.id, document.name, document.content, document.summary,
            document.upload_date, document.page_count, document.file_size, embeddings_blob
        ))
        
        conn.commit()
        conn.close()
    
    def load_documents(self) -> Dict[str, Document]:
        """Load all documents from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM documents")
        rows = cursor.fetchall()
        
        documents = {}
        for row in rows:
            try:
                embeddings = pickle.loads(row[7]) if row[7] else None
                doc = Document(
                    id=row[0], name=row[1], content=row[2], chunks=[],
                    embeddings=embeddings, summary=row[3],
                    upload_date=datetime.fromisoformat(row[4]) if row[4] else datetime.now(),
                    page_count=row[5], file_size=row[6]
                )
                documents[doc.id] = doc
            except Exception as e:
                logger.error(f"Failed to load document {row[0]}: {e}")

        conn.close()
        return documents
    
    def save_conversation(self, conversation_id: str, name: str, messages: List[ChatMessage]):
        """Save conversation to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Save conversation
        cursor.execute("""
            INSERT OR REPLACE INTO conversations (id, name, created_date)
            VALUES (?, ?, ?)
        """, (conversation_id, name, datetime.now()))
        
        # Clear existing messages
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        
        # Save messages
        for msg in messages:
            cursor.execute("""
                INSERT INTO messages 
                (conversation_id, role, content, timestamp, sources, document_ids)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                conversation_id, msg.role, msg.content, msg.timestamp,
                json.dumps(msg.sources) if msg.sources else None,
                json.dumps(msg.document_ids) if msg.document_ids else None
            ))
        
        conn.commit()
        conn.close()

class EnhancedOllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()
        
    async def check_status(self) -> bool:
        """Check if Ollama server is running"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags", timeout=5) as response:
                    return response.status == 200
        except Exception:
            return False
    
    def generate_stream(self, model: str, prompt: str, context: str = None) -> str:
        """Generate streaming response from Ollama model"""
        url = f"{self.base_url}/api/generate"
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "context": context,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        try:
            with requests.post(url, json=data, stream=True, timeout=120) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line)
                            if 'response' in json_response:
                                yield json_response['response']
                        except json.JSONDecodeError:
                            continue
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to Ollama: {e}")
            yield ""
    
    def generate_embeddings(self, model: str, text: str) -> Optional[List[float]]:
        """Generate embeddings using Ollama"""
        url = f"{self.base_url}/api/embeddings"
        
        data = {
            "model": model,
            "prompt": text
        }
        
        try:
            response = self.session.post(url, json=data, timeout=30)
            response.raise_for_status()
            return response.json().get('embedding', None)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error generating embeddings: {e}")
            return None
    
    def list_models(self) -> List[str]:
        """List available Ollama models"""
        url = f"{self.base_url}/api/tags"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry"""
        url = f"{self.base_url}/api/pull"
        data = {"name": model_name}
        
        try:
            response = self.session.post(url, json=data, timeout=300)
            return response.status_code == 200
        except Exception:
            return False

class AdvancedPDFProcessor:
    @staticmethod
    def extract_text_with_metadata(pdf_file) -> Tuple[str, int, Dict]:
        """Extract text with metadata using PyMuPDF"""
        try:
            pdf_bytes = pdf_file.read()
            pdf_file.seek(0)
            
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            text = ""
            metadata = {'pages': [], 'images': 0, 'links': 0}
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text("text")
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                metadata['pages'].append({'page_num': page_num + 1, 'char_count': len(page_text)})
                metadata['images'] += len(page.get_images(full=True))
                metadata['links'] += len(page.get_links())
            
            page_count = doc.page_count
            doc.close()
            return text, page_count, metadata
            
        except Exception as e:
            logger.error(f"Error reading PDF with PyMuPDF: {e}")
            try:
                pdf_reader = PdfReader(pdf_file)
                text = "".join(page.extract_text() for page in pdf_reader.pages)
                return text, len(pdf_reader.pages), {}
            except Exception as e2:
                logger.error(f"Error reading PDF with PyPDF2: {e2}")
                return "", 0, {}
    
    @staticmethod
    def recursive_character_split(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Advanced recursive character splitting"""
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

class EnhancedVectorStore:
    def __init__(self, ollama_client: EnhancedOllamaClient):
        self.ollama_client = ollama_client
        self.embeddings_cache = {}
    
    def create_embeddings(self, chunks: List[str], model: str) -> Optional[np.ndarray]:
        """Create embeddings for chunks using Ollama"""
        if not chunks:
            return None
        
        embeddings = []
        progress_bar = st.progress(0, text="Generating embeddings...")
        
        for i, chunk in enumerate(chunks):
            cache_key = hashlib.md5(f"{model}:{chunk}".encode()).hexdigest()
            
            if cache_key in self.embeddings_cache:
                embedding = self.embeddings_cache[cache_key]
            else:
                embedding = self.ollama_client.generate_embeddings(model, chunk)
                if embedding:
                    self.embeddings_cache[cache_key] = embedding
                else:
                    embedding = [0.0] * 384  # Fallback
            
            embeddings.append(embedding)
            progress_bar.progress((i + 1) / len(chunks))
            time.sleep(0.05)
        
        progress_bar.empty()
        return np.array(embeddings)
    
    def find_similar_chunks(self, query: str, embeddings: np.ndarray, 
                          chunks: List[str], model: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar chunks with scores"""
        if embeddings is None or not chunks:
            return []
        
        query_embedding = self.ollama_client.generate_embeddings(model, query)
        if not query_embedding:
            return []
        
        similarities = cosine_similarity([query_embedding], embeddings).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(chunks[idx], float(similarities[idx])) for idx in top_indices if similarities[idx] > 0.1]

class RAGPipeline:
    def __init__(self, ollama_client: EnhancedOllamaClient, vector_store: EnhancedVectorStore):
        self.ollama_client = ollama_client
        self.vector_store = vector_store
    
    def generate_response(self, query: str, documents: Dict[str, Document], 
                         config: ConfigProfile, streaming: bool = True):
        """Generate RAG response with memory"""
        all_relevant_chunks = []
        source_docs = set()
        
        for doc_id, document in documents.items():
            if document.embeddings is not None and document.chunks:
                similar_chunks = self.vector_store.find_similar_chunks(
                    query, document.embeddings, document.chunks, 
                    config.embedding_model, config.context_chunks
                )
                for chunk, score in similar_chunks:
                    all_relevant_chunks.append((chunk, score, document.name))
                    source_docs.add(document.name)
        
        all_relevant_chunks.sort(key=lambda x: x[1], reverse=True)
        top_chunks = all_relevant_chunks[:config.context_chunks * 2]
        
        if not top_chunks:
            yield "I couldn't find relevant information in the selected documents to answer your question."
            return

        context = "\n---\n".join([f"[Source: {doc_name}]\n{chunk}\n" for chunk, _, doc_name in top_chunks])
        
        enhanced_prompt = f"""You are an AI assistant. Use the provided context from documents to answer the question.
Context from documents:
{context}
Question: {query}
Answer:"""
        
        if streaming:
            yield from self.ollama_client.generate_stream(config.model, enhanced_prompt)
        else:
            response = ""
            for chunk in self.ollama_client.generate_stream(config.model, enhanced_prompt):
                response += chunk
            yield response

def create_sidebar():
    """Create enhanced sidebar with all controls"""
    with st.sidebar:
        st.header("🔧 Configuration")
        st.subheader("🤖 Ollama Status")
        # Status will be checked and displayed in main()
        
        st.subheader("🛠️ Ollama Settings")
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
        ollama_client = EnhancedOllamaClient(ollama_url)

        st.subheader("📦 Model Management")
        models = ollama_client.list_models()
        
        if models:
            chat_model = st.selectbox("Chat Model", models, index=0, key="chat_model")
            embedding_models = [m for m in models if 'embed' in m.lower() or 'nomic' in m.lower()] or models
            embedding_model = st.selectbox("Embedding Model", embedding_models, index=0, key="embedding_model")
        else:
            st.warning("No models found!")
            chat_model = st.text_input("Chat Model", value="llama2")
            embedding_model = st.text_input("Embedding Model", value="nomic-embed-text")
        
        st.subheader("📝 Processing Settings")
        chunk_size = st.slider("Chunk Size", 200, 2000, 1000, key="chunk_size")
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, key="chunk_overlap")
        context_chunks = st.slider("Context Chunks", 1, 10, 3, key="context_chunks")
        
        st.subheader("📤 Export Options")
        if st.button("💾 Export Chat History"):
            export_chat_history()
        
        return {
            'ollama_client': ollama_client,
            'chat_model': chat_model,
            'embedding_model': embedding_model,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'context_chunks': context_chunks,
            'streaming_enabled': st.session_state.streaming_enabled,
            'auto_summarize': st.session_state.auto_summarize,
        }

def create_document_manager():
    """Create document management interface"""
    st.subheader("📚 Document Manager")
    
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    col1, col2 = st.columns(2)
    process_button = col1.button("🔄 Process Documents", type="primary", use_container_width=True)
    if col2.button("🗑️ Clear All Documents", use_container_width=True):
        st.session_state.documents = {}
        st.success("All documents cleared!")
        st.rerun()
    
    if st.session_state.documents:
        st.write("### Current Documents")
        for doc_id, doc in st.session_state.documents.items():
            with st.expander(f"📄 {doc.name}"):
                st.metric("Pages", doc.page_count)
                if st.button(f"🗑️ Remove", key=f"remove_{doc_id}"):
                    del st.session_state.documents[doc_id]
                    st.rerun()
    
    return uploaded_files, process_button

def process_documents(uploaded_files, config):
    """Process uploaded documents with enhanced features"""
    if not uploaded_files: return

    ollama_client = config['ollama_client']
    pdf_processor = AdvancedPDFProcessor()
    vector_store = EnhancedVectorStore(ollama_client)
    db_manager = DatabaseManager()

    for uploaded_file in uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            try:
                doc_id = hashlib.md5(f"{uploaded_file.name}_{uploaded_file.size}".encode()).hexdigest()
                content, page_count, _ = pdf_processor.extract_text_with_metadata(uploaded_file)
                if not content.strip():
                    st.error(f"Could not extract text from {uploaded_file.name}")
                    continue

                chunks = pdf_processor.recursive_character_split(content, config['chunk_size'], config['chunk_overlap'])
                embeddings = vector_store.create_embeddings(chunks, config['embedding_model'])
                
                summary = ""
                if config['auto_summarize']:
                    summary_prompt = f"Provide a concise summary of this document:\n\n{content[:2000]}..."
                    summary_gen = ollama_client.generate_stream(config['chat_model'], summary_prompt)
                    summary = "".join(list(summary_gen))

                document = Document(
                    id=doc_id, name=uploaded_file.name, content=content, chunks=chunks,
                    embeddings=embeddings, summary=summary, upload_date=datetime.now(),
                    page_count=page_count, file_size=uploaded_file.size
                )
                
                st.session_state.documents[doc_id] = document
                db_manager.save_document(document)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
    st.success(f"Successfully processed {len(uploaded_files)} documents!")

def create_chat_interface(config):
    """Create the main chat interface with enhanced controls."""
    st.subheader("💬 Chat with Your Documents")

    if not st.session_state.documents:
        st.info("👆 Please upload and process some PDF documents to start chatting!")
        return

    # --- Chat Area Wrapper ---
    with st.container():
        # --- Document Selection and Controls ---
        with st.container(border=True):
            st.markdown('<div class="chat-controls">', unsafe_allow_html=True)
            doc_options = {doc.name: doc_id for doc_id, doc in st.session_state.documents.items()}
            selected_doc_names = st.multiselect(
                "Select documents to chat with:",
                options=list(doc_options.keys()),
                default=list(doc_options.keys())
            )
            selected_doc_ids = [doc_options[name] for name in selected_doc_names]
            
            control_cols = st.columns([1, 1, 5])
            if control_cols[0].button("🔄 Regenerate", help="Regenerate the last response"):
                if st.session_state.last_user_prompt:
                    st.session_state.regenerate_response = True
                    if st.session_state.messages and st.session_state.messages[-1].role == 'assistant':
                        st.session_state.messages.pop()

            if control_cols[1].button("🗑️ Clear Chat", help="Clear the current conversation"):
                st.session_state.messages = []
                st.session_state.last_user_prompt = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # --- Chat History ---
        st.markdown('<div class="chat-history">', unsafe_allow_html=True)
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message.role):
                st.markdown(message.content)
                if st.button(f"📋 Copy", key=f"copy_{i}_{message.role}"):
                    st.code(message.content)
                if message.role == 'assistant' and message.sources:
                    with st.expander("📚 View Sources", expanded=False):
                        for j, source in enumerate(message.sources):
                            st.info(f"Source {j+1}:\n\n{source[:500]}...")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Generation Logic ---
    prompt = st.chat_input("Ask a question about your documents...")
    if st.session_state.regenerate_response:
        prompt = st.session_state.last_user_prompt
        st.session_state.regenerate_response = False

    if prompt:
        if not selected_doc_ids:
            st.warning("Please select at least one document to chat with.", icon="⚠️")
            st.stop()
            
        st.session_state.last_user_prompt = prompt
        if st.session_state.messages and st.session_state.messages[-1].content != prompt:
             st.session_state.messages.append(ChatMessage(role="user", content=prompt, timestamp=datetime.now()))
        elif not st.session_state.messages:
             st.session_state.messages.append(ChatMessage(role="user", content=prompt, timestamp=datetime.now()))

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_text = ""
            
            rag_pipeline = RAGPipeline(config['ollama_client'], EnhancedVectorStore(config['ollama_client']))
            config_profile = ConfigProfile("current", config['chat_model'], config['embedding_model'], 
                                           config['chunk_size'], config['chunk_overlap'], 
                                           config['context_chunks'], 0.7)
            
            selected_docs = {doc_id: st.session_state.documents[doc_id] for doc_id in selected_doc_ids}
            
            # Find relevant chunks to store as sources
            vector_store = EnhancedVectorStore(config['ollama_client'])
            all_relevant_chunks = []
            for doc_id, document in selected_docs.items():
                if document.embeddings is not None and document.chunks:
                    similar_chunks = vector_store.find_similar_chunks(
                        prompt, document.embeddings, document.chunks,
                        config['embedding_model'], config['context_chunks']
                    )
                    all_relevant_chunks.extend([chunk for chunk, score in similar_chunks])

            # Generate response
            response_generator = rag_pipeline.generate_response(prompt, selected_docs, config_profile, streaming=True)
            for chunk in response_generator:
                response_text += chunk
                response_placeholder.markdown(response_text + "▌")
            response_placeholder.markdown(response_text)

        assistant_message = ChatMessage("assistant", response_text, datetime.now(), sources=all_relevant_chunks[:3])
        st.session_state.messages.append(assistant_message)
        st.rerun()

def create_conversation_manager():
    """Create conversation management interface"""
    st.subheader("💾 Conversation Manager")
    if st.session_state.messages:
        st.metric("Messages in current conversation", len(st.session_state.messages))
    else:
        st.info("No active conversation")

def export_chat_history():
    """Export chat history to JSON"""
    if not st.session_state.messages:
        st.warning("No messages to export")
        return
    
    export_data = [{"role": m.role, "content": m.content, "timestamp": m.timestamp.isoformat()} for m in st.session_state.messages]
    json_str = json.dumps(export_data, indent=2)
    st.download_button(
        label="Download Chat History",
        data=json_str,
        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )

def check_ollama_status(ollama_client):
    """Check and display Ollama status"""
    with st.sidebar:
        if asyncio.run(ollama_client.check_status()):
            st.success("Ollama is running")
            return True
        else:
            st.error("Ollama is not running")
            return False

def main():
    """Main application function"""
    init_session_state()
    
    if not st.session_state.documents:
        db_manager = DatabaseManager()
        st.session_state.documents = db_manager.load_documents()
    
    st.markdown("<div class='main-header'><h1>📚 Advanced PDF Chat with Ollama</h1></div>", unsafe_allow_html=True)
    
    config = create_sidebar()
    ollama_status = check_ollama_status(config['ollama_client'])
    
    tab1, tab2 = st.tabs(["📄 Documents", "💬 Chat"])
    
    with tab1:
        uploaded_files, process_button = create_document_manager()
        if process_button and uploaded_files:
            if ollama_status:
                process_documents(uploaded_files, config)
            else:
                st.error("Ollama is not running. Please start it before processing documents.")
    
    with tab2:
        if ollama_status:
            create_chat_interface(config)
        else:
            st.error("Ollama is not running. Please start it to use the chat feature.")

if __name__ == "__main__":
    main()
