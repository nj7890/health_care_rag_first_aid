# ingest.py
"""
Offline ingestion script for First-Aid RAG system.

✔ Downloads authoritative PDFs
✔ Extracts text
✔ Chunks + embeds
✔ Builds FAISS index
✔ Saves index + metadata to disk

Run:
    python ingest.py

Output:
    /data/faiss.index
    /data/meta.pkl
    /data/chunks.pkl
"""

import os
import pickle
import requests
from io import BytesIO
import numpy as np
import PyPDF2

<<<<<<< HEAD
from sentence_transformers import SentenceTransformer
=======
#  RAGEngine from your Streamlit app file
from streamlit_first_aid_rag_app import RAGEngine
>>>>>>> e4f94a24f423c2fd8417cd0ae19dc75de53a5d3e

# Try FAISS
try:
    import faiss
    USE_FAISS = True
except:
    from sklearn.neighbors import NearestNeighbors
    USE_FAISS = False


# ---------------- CONFIG ----------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------- UTILS ----------------
def extract_pdf(pdf_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except:
            pages.append("")
    return "\n".join(pages)


def clean_text(text):
    return text.replace('\r', '\n')


def chunk_text(text, size=400, stride=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - stride
    return chunks


# ---------------- INGEST ENGINE ----------------
class IngestEngine:
    def __init__(self):
        self.embedder = SentenceTransformer(MODEL_NAME)
        self.chunks = []
        self.metadata = []

    def add_document(self, text: str, source_id: str):
        text = clean_text(text)
        chunks = chunk_text(text)

        for i, c in enumerate(chunks):
            self.chunks.append(c)
            self.metadata.append({
                "source": source_id,
                "chunk_id": f"{source_id}_{i}"
            })

    def build_index(self):
        print("🔄 Generating embeddings...")
        embeddings = self.embedder.encode(self.chunks)

        if USE_FAISS:
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(np.array(embeddings))
        else:
            index = NearestNeighbors().fit(embeddings)

        return index, embeddings

    def save(self, index, embeddings):
        print("💾 Saving index + metadata...")

        if USE_FAISS:
            faiss.write_index(index, os.path.join(DATA_DIR, "faiss.index"))
        else:
            pickle.dump(index, open(os.path.join(DATA_DIR, "sklearn.index"), "wb"))

        pickle.dump(self.metadata, open(os.path.join(DATA_DIR, "meta.pkl"), "wb"))
        pickle.dump(self.chunks, open(os.path.join(DATA_DIR, "chunks.pkl"), "wb"))

        print("✅ Saved successfully!")


# ---------------- DOWNLOAD + INGEST ----------------
def download_pdf(url):
    print(f"⬇️ Downloading: {url}")
    r = requests.get(url)
    r.raise_for_status()
    return r.content


def ingest_sources(engine, sources):
    for src in sources:
        try:
            pdf = download_pdf(src["url"])
            text = extract_pdf(pdf)

            print(f"📄 {src['source_id']} | chars: {len(text)}")

            engine.add_document(text, src["source_id"])

        except Exception as e:
            print(f"❌ Failed: {src['source_id']} | {e}")


# ---------------- MAIN ----------------
if __name__ == "__main__":

    engine = IngestEngine()

    sources = [
        {
            "source_id": "IFRC_2020",
            "url": "https://www.ifrc.org/sites/default/files/2022-02/EN_GFARC_GUIDELINES_2020.pdf"
        },
        {
            "source_id": "ICRC_FirstAid",
            "url": "https://www.icrc.org/sites/default/files/external/doc/en/assets/files/publications/icrc-002-0526.pdf"
        },
        {
            "source_id": "Canadian_RedCross",
            "url": "https://cdn.redcross.ca/prodmedia/crc/documents/Comprehensive_Guide_for_FirstAidCPR_en.pdf"
        }
    ]

    print("🚀 Starting ingestion pipeline...\n")

    ingest_sources(engine, sources)

    print(f"\n📦 Total chunks: {len(engine.chunks)}")

    index, embeddings = engine.build_index()

    engine.save(index, embeddings)

    print("\n🎉 Ingestion complete! Data stored in /data/")