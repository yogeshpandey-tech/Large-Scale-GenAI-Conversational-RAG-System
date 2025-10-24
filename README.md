# 📚 Large-Scale Conversational RAG System

An advanced **Retrieval-Augmented Generation (RAG)** system built using **Streamlit**, **LangChain**, **Groq LLMs**, and **ChromaDB** — capable of handling **6,000+ PDFs** with persistent memory, batch processing, and dynamic querying.

---

## 🚀 Key Features

- 🧠 **Conversational RAG Pipeline**
  - Context-aware retrieval with persistent chat memory  
  - Powered by **Groq LLMs** for ultra-fast inference  
  - Answers with **source citations** and **multi-language support (Hindi + English)**  

- 📂 **Massive Document Handling**
  - Process **ZIPs, individual files, or entire folders** (PDF, TXT, DOCX)  
  - Supports **6,000+ PDFs** with multiprocessing for fast ingestion  
  - Tracks processed files to skip duplicates  

- 💾 **Persistent Vector Database**
  - Uses **ChromaDB** for long-term storage of embeddings  
  - Incremental updates without reprocessing all files  

- ⚙️ **Configurable Search**
  - Choose **similarity** or **MMR (Max Marginal Relevance)** search  
  - Tune top-k results, diversity, and fetch count  

- 💬 **Interactive UI (Streamlit)**
  - Clean, responsive interface  
  - Upload, process, and query directly from browser  
  - Real-time progress, chat history, and citations  

---

## 🏗️ Tech Stack

| Layer | Technologies |
|:------|:--------------|
| **Frontend / UI** | Streamlit |
| **LLM Engine** | Groq (Llama 3.1 8B Instant) |
| **Framework** | LangChain |
| **Embeddings** | HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Vector Store** | ChromaDB |
| **Concurrency** | Python `concurrent.futures` (multiprocessing) |
| **Storage** | Persistent local directories |

---

## 📁 Project Structure

<img width="666" height="319" alt="image" src="https://github.com/user-attachments/assets/af17aa9d-513c-4568-8446-65ddd9da5b3f" />

