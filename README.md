# First Aid RAG Knowledge Base Ingestion Script

This script downloads authoritative first-aid PDF documents, extracts text,
and ingests them into the RAG knowledge base used by the Streamlit
First-Aid RAG application.

It works together with:

streamlit_first_aid_rag_app.py

which defines the `RAGEngine` class used for embeddings, chunking,
and vector search.

---

## Features

- Download PDFs from URL
- Extract text using PyPDF2
- Chunk + embed using SentenceTransformers (inside RAGEngine)
- Store embeddings in FAISS / sklearn index
- Add multiple sources automatically
- Designed for first-aid / medical guideline documents

---

## File Structure
