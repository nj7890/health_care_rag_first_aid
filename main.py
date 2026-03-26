import streamlit as st
import re
import requests
from io import BytesIO
import numpy as np
from typing import List, Dict, Any
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# ---------------- TEXT UTILS ----------------
def clean_text(text):
    text = text.replace('\r', '\n')
    text = re.sub('\n{3,}', '\n\n', text)
    return text.strip()

def chunk_text(text, size=400, stride=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - stride
    return chunks

def extract_pdf(pdf_bytes):
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    text = []
    for p in reader.pages:
        try:
            text.append(p.extract_text() or "")
        except:
            text.append("")
    return "\n".join(text)

# ---------------- EMERGENCY ----------------
def emergency_check(query, chunks):
    patterns = ['not breathing', 'no pulse', 'unconscious', 'bleeding', 'choking']
    found = [p for p in patterns if p in query.lower()]
    return len(found) > 0, found

# ---------------- RAG ENGINE ----------------
class RAGEngine:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.text_chunks = []
        self.metadata = []
        self.index = None

    def add_source(self, text, source):
        chunks = chunk_text(clean_text(text))
        for i, c in enumerate(chunks):
            self.text_chunks.append(c)
            self.metadata.append({"source": source})
        self.build_index()

    def build_index(self):
        if not self.text_chunks:
            return
        emb = self.embedder.encode(self.text_chunks)
        self.index = NearestNeighbors().fit(emb)

    def retrieve(self, query, k=5):
        emb = self.embedder.encode([query])
        dists, idxs = self.index.kneighbors(emb, n_neighbors=k)
        results = []
        for d, i in zip(dists[0], idxs[0]):
            results.append((d, self.text_chunks[i], self.metadata[i]))
        return results

    def answer(self, query):
        results = self.retrieve(query)
        steps, warnings = [], []

        for _, text, meta in results:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for s in sentences:
                if re.match(r'^(Call|Apply|Check|Keep|Stop)', s, re.I):
                    steps.append(s)
                if "do not" in s.lower():
                    warnings.append(s)

        return steps[:5], warnings[:3], results

# ---------------- INIT ----------------
st.set_page_config(layout="wide")
st.title("🚑 First-Aid RAG Assistant")

if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

rag = st.session_state.rag

# ---------------- AUTO LOAD KB ----------------
def load_default_kb():
    sources = {
        "IFRC": "https://www.ifrc.org/sites/default/files/2022-02/EN_GFARC_GUIDELINES_2020.pdf",
        "ICRC": "https://www.icrc.org/sites/default/files/external/doc/en/assets/files/publications/icrc-002-0526.pdf"
    }
    for name, url in sources.items():
        try:
            pdf = requests.get(url).content
            text = extract_pdf(pdf)
            rag.add_source(text, name)
        except:
            pass

# AUTO INIT (FIX FOR EMPTY KB)
if "kb_loaded" not in st.session_state:
    with st.spinner("Loading medical knowledge base..."):
        load_default_kb()
    st.session_state.kb_loaded = True

# ---------------- SIDEBAR ----------------
st.sidebar.header("Knowledge Base")

uploaded = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
if uploaded:
    text = extract_pdf(uploaded.read())
    rag.add_source(text, uploaded.name)
    st.sidebar.success("Added to KB")

st.sidebar.write("Chunks:", len(rag.text_chunks))

# ---------------- MAIN ----------------
query = st.text_area("Describe situation")

if st.button("Get Guidance"):
    if not rag.text_chunks:
        st.warning("Knowledge base is empty.")
    else:
        steps, warnings, sources = rag.answer(query)

        tabs = st.tabs(["Guidance", "Warnings", "Sources"])

        with tabs[0]:
            for i, s in enumerate(steps, 1):
                st.write(f"{i}. {s}")

        with tabs[1]:
            for w in warnings:
                st.warning(w)

        with tabs[2]:
            for _, t, m in sources:
                st.write(m["source"])
                st.caption(t[:300])
