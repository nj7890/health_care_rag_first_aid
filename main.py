import streamlit as st
import re
import requests
from io import BytesIO
import numpy as np
from typing import List, Dict, Any

# PDF support
import PyPDF2

# Embeddings
from sentence_transformers import SentenceTransformer

# Vector Search
try:
    import faiss
    USE_FAISS = True
except:
    from sklearn.neighbors import NearestNeighbors
    USE_FAISS = False


# ---------------- TEXT PROCESSING ----------------
def clean_text(text: str) -> str:
    text = text.replace('\r', '\n')
    text = re.sub('\n{3,}', '\n\n', text)
    return text.strip()


def chunk_text(text: str, chunk_size=400, stride=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - stride
    return chunks


def extract_pdf(pdf_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except:
            pages.append("")
    return "\n".join(pages)


# ---------------- EMERGENCY DETECTION ----------------
def emergency_check(query: str, retrieved_chunks: List[str]) -> Dict[str, Any]:
    patterns = [
        r'not breathing|no breathing',
        r'no pulse|cardiac arrest',
        r'unconscious|unresponsive',
        r'severe bleeding|heavy bleeding',
        r'choking',
        r'seizure',
        r'collapse'
    ]

    score = 0
    triggers = []

    combined = query.lower() + " " + " ".join(retrieved_chunks).lower()

    for p in patterns:
        if re.search(p, combined):
            score += 1
            triggers.append(p)

    severity = "HIGH" if score >= 2 else "MEDIUM" if score == 1 else "LOW"

    return {
        "is_emergency": score > 0,
        "severity": severity,
        "triggers": triggers
    }


# ---------------- RAG ENGINE ----------------
class RAGEngine:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.text_chunks = []
        self.metadata = []
        self.embeddings = None
        self.index = None

    def add_source(self, text: str, source_id: str):
        text = clean_text(text)
        chunks = chunk_text(text)

        for i, c in enumerate(chunks):
            self.text_chunks.append(c)
            self.metadata.append({
                "source": source_id,
                "chunk_id": f"{source_id}_{i}"
            })

        self._build_index()

    def _build_index(self):
        if not self.text_chunks:
            return

        self.embeddings = self.embedder.encode(self.text_chunks)

        if USE_FAISS:
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(np.array(self.embeddings))
        else:
            self.index = NearestNeighbors().fit(self.embeddings)

    def retrieve(self, query: str, top_k=5):
        q_emb = self.embedder.encode([query])[0]

        if USE_FAISS:
            D, I = self.index.search(np.array([q_emb]), top_k)
        else:
            D, I = self.index.kneighbors([q_emb], n_neighbors=top_k)

        results = []
        query_terms = set(query.lower().split())

        for dist, idx in zip(D[0], I[0]):
            text = self.text_chunks[idx]
            meta = self.metadata[idx]

            overlap = len(query_terms.intersection(set(text.lower().split())))
            score = float(dist) - 0.05 * overlap

            results.append((score, text, meta))

        results.sort(key=lambda x: x[0])
        return results

    def assemble_answer(self, query: str, top_k=5):
        retrieved = self.retrieve(query, top_k)

        snippets = [r[1] for r in retrieved]
        emergency = emergency_check(query, snippets)

        steps, warnings, explain = [], [], []

        for score, text, meta in retrieved:
            sentences = re.split(r'(?<=[.!?])\s+', text)

            for sent in sentences:
                sent = sent.strip()

                if len(sent) < 15:
                    continue

                if re.match(r'^(Call|Apply|Check|Place|Keep|Stop|Perform|Ensure)', sent, re.I):
                    steps.append(sent)
                    explain.append({
                        "step": sent,
                        "source": meta["source"],
                        "confidence": round(1 / (1 + score), 3)
                    })

                if "do not" in sent.lower() or "avoid" in sent.lower():
                    warnings.append(sent)

            if len(steps) >= 6:
                break

        return {
            "steps": steps[:6],
            "warnings": warnings[:3],
            "explain": explain,
            "sources": retrieved,
            "emergency": emergency
        }


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="First-Aid RAG", layout="wide")
st.title("🚑 First-Aid RAG Assistant (Explainable & Safe)")

# Initialize session
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

rag = st.session_state.rag


# ---------------- SIDEBAR ----------------
st.sidebar.header("Knowledge Base")

if st.sidebar.button("Load Standard Medical Sources"):
    sources = {
        "IFRC_2020": "https://www.ifrc.org/sites/default/files/2022-02/EN_GFARC_GUIDELINES_2020.pdf",
        "ICRC": "https://www.icrc.org/sites/default/files/external/doc/en/assets/files/publications/icrc-002-0526.pdf"
    }

    for name, url in sources.items():
        try:
            pdf = requests.get(url).content
            text = extract_pdf(pdf)
            rag.add_source(text, name)
        except:
            st.sidebar.warning(f"Failed to load {name}")

    st.sidebar.success("Medical sources loaded")


uploaded = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded:
    text = extract_pdf(uploaded.read())
    rag.add_source(text, uploaded.name)
    st.sidebar.success("File added to KB")


# ---------------- MAIN ----------------
query = st.text_area("Describe the situation or ask a first-aid question")

top_k = st.slider("Retrieval depth (k)", 1, 10, 5)

if st.button("Get Guidance"):
    if not query.strip():
        st.warning("Please enter a query.")
    elif not rag.text_chunks:
        st.warning("Knowledge base is empty.")
    else:
        with st.spinner("Analyzing..."):
            ans = rag.assemble_answer(query, top_k)

        # 🚨 Emergency Banner
        if ans["emergency"]["is_emergency"]:
            st.error(
                f"🚨 EMERGENCY DETECTED (Severity: {ans['emergency']['severity']})\n"
                f"Triggers: {', '.join(ans['emergency']['triggers'])}\n"
                f"👉 Call emergency services immediately."
            )

        tabs = st.tabs(["🩺 Guidance", "⚠️ Warnings", "🔍 Explainability", "📚 Sources"])

        with tabs[0]:
            st.subheader("Step-by-step Guidance")
            for i, step in enumerate(ans["steps"], 1):
                st.markdown(f"**{i}.** {step}")

        with tabs[1]:
            if ans["warnings"]:
                st.subheader("What NOT to do")
                for w in ans["warnings"]:
                    st.warning(w)

        with tabs[2]:
            st.subheader("Why these steps?")
            for e in ans["explain"]:
                st.write(f"• {e['step']}")
                st.caption(f"Source: {e['source']} | Confidence: {e['confidence']}")

        with tabs[3]:
            for _, text, meta in ans["sources"]:
                st.write("Source:", meta["source"])
                st.caption(text[:400])


st.markdown("---")
st.caption("⚠️ This tool provides educational first-aid guidance from authoritative sources. Always seek professional medical help in emergencies.")