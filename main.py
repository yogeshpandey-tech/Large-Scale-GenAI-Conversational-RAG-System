import streamlit as st
import tempfile
import os
import zipfile
from io import BytesIO
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from pathlib import Path
import shutil
from datetime import datetime

# ---------------- ENV SETUP ----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Persistent directory for vector store
PERSIST_DIRECTORY = "./chroma_db"
PROCESSED_FILES_LOG = "./processed_files.txt"
PDF_FOLDER_PATH = "./pdf_storage"  # For direct folder processing

# Create necessary directories
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
os.makedirs(PDF_FOLDER_PATH, exist_ok=True)

# Initialize embeddings once
@st.cache_resource
def get_embeddings():
    """Cached embeddings model to avoid reloading"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
    )

embeddings = get_embeddings()

# ---------------- STREAMLIT UI ----------------
st.set_page_config(
    page_title="Large-Scale RAG System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📚 Large-Scale Conversational RAG System")
st.write("**Supports 6,000+ PDFs with persistent storage and batch processing**")

# ---------------- LLM SETUP ----------------
if not GROQ_API_KEY:
    st.error("🚨 Please add your GROQ_API_KEY to the .env file")
    st.stop()

@st.cache_resource
def get_llm():
    """Cached LLM instance"""
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

llm = get_llm()

# ---------------- SESSION MANAGEMENT ----------------
st.sidebar.header("⚙️ Session Settings")
session_id = st.sidebar.text_input("Session ID", value="default_session")

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session: str) -> BaseChatMessageHistory:
    """Manage chat history with memory limit"""
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    history = st.session_state.store[session]
    # Keep only last 10 messages to manage memory
    if len(history.messages) > 10:
        history.messages = history.messages[-10:]
    return history

# ---------------- HELPER FUNCTIONS ----------------
def get_file_hash(file_path: str) -> str:
    """Generate unique hash for file to track processing"""
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        st.error(f"Error hashing file {file_path}: {str(e)}")
        return None

def load_processed_files() -> dict:
    """Load list of already processed file hashes with filenames"""
    processed = {}
    if os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, 'r') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) == 2:
                    processed[parts[0]] = parts[1]
    return processed

def save_processed_file(file_hash: str, file_name: str):
    """Save processed file hash and name to log"""
    with open(PROCESSED_FILES_LOG, 'a') as f:
        f.write(f"{file_hash}|{file_name}\n")

def process_single_file(file_path: str, file_name: str) -> tuple:
    """Process a single file and return documents"""
    try:
        if file_name.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_name.lower().endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_name.lower().endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            return [], None, None
        
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = file_name
            d.metadata["processed_date"] = datetime.now().isoformat()
        
        file_hash = get_file_hash(file_path)
        return docs, file_hash, file_name
    except Exception as e:
        return [], None, f"Error: {str(e)}"

def batch_process_files(file_list: list, batch_size: int = 50, max_workers: int = 4) -> tuple:
    """Process files in batches using multiprocessing"""
    all_documents = []
    processed_files_dict = load_processed_files()
    new_files = []
    skipped_files = []
    failed_files = []
    
    # Filter out already processed files
    for file_path, file_name in file_list:
        file_hash = get_file_hash(file_path)
        if file_hash and file_hash not in processed_files_dict:
            new_files.append((file_path, file_name))
        else:
            skipped_files.append(file_name)
    
    if not new_files:
        return [], skipped_files, failed_files
    
    st.info(f"📋 Processing {len(new_files)} new files (Skipped {len(skipped_files)} already processed)...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    processed_count = 0
    total_files = len(new_files)
    
    # Process in batches
    for i in range(0, len(new_files), batch_size):
        batch = new_files[i:i + batch_size]
        status_text.text(f"Processing batch {i//batch_size + 1}/{(total_files//batch_size) + 1}...")
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_file, fp, fn): (fp, fn) 
                      for fp, fn in batch}
            
            for future in as_completed(futures):
                file_path, file_name = futures[future]
                docs, file_hash, error = future.result()
                
                if docs and file_hash:
                    all_documents.extend(docs)
                    save_processed_file(file_hash, file_name)
                    processed_count += 1
                elif error:
                    failed_files.append(f"{file_name}: {error}")
        
        # Update progress
        progress = min((i + batch_size) / total_files, 1.0)
        progress_bar.progress(progress)
    
    status_text.empty()
    return all_documents, skipped_files, failed_files

@st.cache_resource
def get_or_create_vectorstore():
    """Get existing vector store or create new one"""
    try:
        if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
            return Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings
            )
        else:
            return Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings
            )
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def add_documents_to_vectorstore(documents: list):
    """Add documents to vector store in batches"""
    if not documents:
        return False
    
    try:
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        
        st.info(f"📄 Generated {len(splits)} text chunks from {len(documents)} documents")
        
        # Add to persistent vector store in batches
        vectorstore = get_or_create_vectorstore()
        if vectorstore is None:
            return False
        
        batch_size = 100
        progress_bar = st.progress(0)
        
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            vectorstore.add_documents(batch)
            progress = min((i + batch_size) / len(splits), 1.0)
            progress_bar.progress(progress)
        
        st.success(f"✅ Successfully added {len(splits)} chunks to vector database!")
        return True
    except Exception as e:
        st.error(f"Error adding documents to vector store: {str(e)}")
        return False

# ---------------- SIDEBAR: DATABASE MANAGEMENT ----------------
st.sidebar.header("🗄️ Database Management")

# Show database stats
if st.sidebar.button("📊 Show Database Stats"):
    vectorstore = get_or_create_vectorstore()
    if vectorstore:
        try:
            collection = vectorstore._collection
            count = collection.count()
            st.sidebar.success(f"**Total chunks:** {count}")
            
            # Count processed files
            processed_files = load_processed_files()
            st.sidebar.success(f"**Total files:** {len(processed_files)}")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")
    else:
        st.sidebar.warning("No database found")

# Clear database
if st.sidebar.button("🗑️ Clear Entire Database"):
    confirm = st.sidebar.checkbox("⚠️ I confirm deletion")
    if confirm:
        try:
            if os.path.exists(PERSIST_DIRECTORY):
                shutil.rmtree(PERSIST_DIRECTORY)
            if os.path.exists(PROCESSED_FILES_LOG):
                os.remove(PROCESSED_FILES_LOG)
            st.cache_resource.clear()
            st.sidebar.success("✅ Database cleared!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

# Export processed files list
if st.sidebar.button("📥 Export Processed Files List"):
    processed_files = load_processed_files()
    if processed_files:
        file_list = "\n".join(processed_files.values())
        st.sidebar.download_button(
            label="Download List",
            data=file_list,
            file_name="processed_files_list.txt",
            mime="text/plain"
        )

# ---------------- MAIN AREA: FILE UPLOAD OPTIONS ----------------
st.header("📤 Upload Documents")

tab1, tab2, tab3 = st.tabs(["📦 ZIP Upload", "📄 Individual Files", "📁 Folder Path"])

# TAB 1: ZIP Upload
with tab1:
    st.write("Upload one or multiple ZIP files containing PDFs, TXT, or DOCX files")
    uploaded_zips = st.file_uploader(
        "Select ZIP files",
        type="zip",
        accept_multiple_files=True,
        key="zip_uploader"
    )
    
    if uploaded_zips:
        st.info(f"📦 {len(uploaded_zips)} ZIP file(s) selected")
        
        if st.button("⚡ Process ZIP Files", key="process_zip"):
            file_list = []
            
            with st.spinner("Extracting ZIP files..."):
                for uploaded_zip in uploaded_zips:
                    try:
                        with zipfile.ZipFile(BytesIO(uploaded_zip.read())) as z:
                            for file_name in z.namelist():
                                if file_name.lower().endswith((".pdf", ".txt", ".docx")):
                                    with z.open(file_name) as f:
                                        # Create unique temp path
                                        temp_file_path = os.path.join(
                                            tempfile.gettempdir(),
                                            f"{hashlib.md5(file_name.encode()).hexdigest()}_{os.path.basename(file_name)}"
                                        )
                                        with open(temp_file_path, "wb") as temp_file:
                                            temp_file.write(f.read())
                                        file_list.append((temp_file_path, file_name))
                    except Exception as e:
                        st.error(f"Error extracting {uploaded_zip.name}: {str(e)}")
            
            if file_list:
                st.success(f"✅ Extracted {len(file_list)} files from ZIP(s)")
                
                # Process files
                documents, skipped, failed = batch_process_files(file_list, batch_size=50, max_workers=4)
                
                # Show results
                if documents:
                    if add_documents_to_vectorstore(documents):
                        st.balloons()
                
                if skipped:
                    with st.expander(f"⏭️ Skipped {len(skipped)} already processed files"):
                        for f in skipped[:20]:  # Show first 20
                            st.text(f)
                        if len(skipped) > 20:
                            st.text(f"... and {len(skipped) - 20} more")
                
                if failed:
                    with st.expander(f"❌ Failed to process {len(failed)} files"):
                        for f in failed:
                            st.text(f)

# TAB 2: Individual Files
with tab2:
    st.write("Upload individual PDF, TXT, or DOCX files")
    uploaded_files = st.file_uploader(
        "Select files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        st.info(f"📄 {len(uploaded_files)} file(s) selected")
        
        if st.button("⚡ Process Individual Files", key="process_files"):
            file_list = []
            
            for uploaded_file in uploaded_files:
                temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                file_list.append((temp_file_path, uploaded_file.name))
            
            # Process files
            documents, skipped, failed = batch_process_files(file_list, batch_size=50, max_workers=4)
            
            # Show results
            if documents:
                if add_documents_to_vectorstore(documents):
                    st.balloons()
            
            if skipped:
                st.warning(f"⏭️ Skipped {len(skipped)} already processed files")
            
            if failed:
                with st.expander(f"❌ Failed files ({len(failed)})"):
                    for f in failed:
                        st.text(f)

# TAB 3: Folder Path
with tab3:
    st.write("Process files directly from a folder on your server (best for 6000+ PDFs)")
    
    folder_path = st.text_input(
        "📁 Enter full folder path:",
        value=PDF_FOLDER_PATH,
        help="Enter the full path to the folder containing your PDF/TXT/DOCX files"
    )
    
    include_subfolders = st.checkbox("Include subfolders", value=True)
    
    if st.button("🔍 Scan Folder", key="scan_folder"):
        if folder_path and os.path.exists(folder_path):
            file_list = []
            
            if include_subfolders:
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        if file.lower().endswith((".pdf", ".txt", ".docx")):
                            full_path = os.path.join(root, file)
                            file_list.append((full_path, file))
            else:
                for file in os.listdir(folder_path):
                    if file.lower().endswith((".pdf", ".txt", ".docx")):
                        full_path = os.path.join(folder_path, file)
                        file_list.append((full_path, file))
            
            if file_list:
                st.success(f"✅ Found {len(file_list)} compatible files")
                st.session_state['folder_files'] = file_list
            else:
                st.warning("No PDF/TXT/DOCX files found in folder")
        else:
            st.error("❌ Folder path does not exist")
    
    if 'folder_files' in st.session_state and st.session_state['folder_files']:
        st.info(f"📋 Ready to process {len(st.session_state['folder_files'])} files")
        
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.number_input("Batch size", min_value=10, max_value=200, value=50)
        with col2:
            max_workers = st.number_input("Parallel workers", min_value=1, max_value=8, value=4)
        
        if st.button("⚡ Process Folder Files", key="process_folder"):
            file_list = st.session_state['folder_files']
            
            # Process files
            documents, skipped, failed = batch_process_files(
                file_list,
                batch_size=batch_size,
                max_workers=max_workers
            )
            
            # Show results
            if documents:
                if add_documents_to_vectorstore(documents):
                    st.balloons()
                    del st.session_state['folder_files']
            
            if skipped:
                with st.expander(f"⏭️ Skipped {len(skipped)} already processed files"):
                    for f in skipped[:50]:
                        st.text(f)
                    if len(skipped) > 50:
                        st.text(f"... and {len(skipped) - 50} more")
            
            if failed:
                with st.expander(f"❌ Failed files ({len(failed)})"):
                    for f in failed:
                        st.text(f)

# ---------------- QUERY INTERFACE ----------------
st.divider()
st.header("💬 Query Your Documents")

# Check if vector store exists
if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
    vectorstore = get_or_create_vectorstore()
    
    if vectorstore:
        # Advanced search settings
        with st.expander("🔧 Advanced Search Settings"):
            col1, col2 = st.columns(2)
            with col1:
                search_type = st.selectbox(
                    "Search type",
                    ["similarity", "mmr"],
                    help="MMR provides more diverse results"
                )
                k_results = st.slider("Number of results", 3, 10, 5)
            with col2:
                if search_type == "mmr":
                    fetch_k = st.slider("Fetch K", 10, 50, 20)
                    lambda_mult = st.slider("Lambda (diversity)", 0.0, 1.0, 0.5)
        
        # Create retriever
        if search_type == "similarity":
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k_results}
            )
        else:
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k_results,
                    "fetch_k": fetch_k,
                    "lambda_mult": lambda_mult
                }
            )
        
        # Contextualize prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question understandable "
            "without the chat history. Do NOT answer the question; "
            "just reformulate it if needed."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        # QA prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following retrieved context to answer thoroughly and accurately. "
            "Each chunk has metadata 'source' (filename) and 'page' (page number if applicable). "
            "When answering:\n"
            "1. Provide detailed, comprehensive answers\n"
            "2. Cite the source filename and page number for each piece of information\n"
            "3. If the question is in Hindi, answer in Hindi. If in English, answer in English.\n"
            "4. If you cannot find the answer in the context, say so clearly.\n\n"
            "Context:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        # User input
        user_input = st.text_area("Your question:", height=100, placeholder="Type your question here...")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                if session_id in st.session_state.store:
                    st.session_state.store[session_id] = ChatMessageHistory()
                    st.success("Chat history cleared!")
        
        if ask_button and user_input:
            with st.spinner("🔍 Searching and generating answer..."):
                try:
                    session_history = get_session_history(session_id)
                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": session_id}}
                    )
                    
                    # Display answer
                    st.markdown("### 🤖 Assistant:")
                    st.markdown(response["answer"])
                    
                    # Display source documents
                    with st.expander("📄 Source Documents"):
                        for idx, doc in enumerate(response.get("context", []), 1):
                            st.markdown(f"**Source {idx}:** {doc.metadata.get('source', 'Unknown')}")
                            st.markdown(f"**Page:** {doc.metadata.get('page', 'N/A')}")
                            st.markdown(f"``````")
                            st.divider()
                    
                    # Display chat history
                    with st.expander("💭 Chat History"):
                        for idx, msg in enumerate(session_history.messages):
                            msg_type = "🧑 User" if idx % 2 == 0 else "🤖 Assistant"
                            st.markdown(f"**{msg_type}:** {msg.content[:200]}...")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Display current chat history
        if session_id in st.session_state.store:
            session_history = get_session_history(session_id)
            if len(session_history.messages) > 0:
                st.divider()
                st.subheader("📝 Current Conversation")
                for idx, msg in enumerate(session_history.messages[-6:]):  # Show last 6 messages
                    if idx % 2 == 0:
                        st.markdown(f"**🧑 You:** {msg.content}")
                    else:
                        st.markdown(f"**🤖 Assistant:** {msg.content[:300]}...")

else:
    st.warning("⚠️ No documents in database. Please upload and process files first using one of the tabs above.")
    st.info("💡 **Tip:** For 6000+ PDFs, use the 'Folder Path' option for best performance.")

# ---------------- FOOTER ----------------
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>🚀 Large-Scale RAG System | Powered by LangChain & Groq | Supports 6000+ Documents</p>
</div>
""", unsafe_allow_html=True)
