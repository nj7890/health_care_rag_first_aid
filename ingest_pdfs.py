import requests
import pickle
from io import BytesIO
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# ---------------- UTILS ----------------
def extract_pdf(pdf_bytes):
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    text = []
    for p in reader.pages:
        text.append(p.extract_text() or "")
    return "\n".join(text)

def chunk_text(text, size=400, stride=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - stride
    return chunks

# ---------------- INGEST ----------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

sources = [
    ("IFRC", "https://www.ifrc.org/sites/default/files/2022-02/EN_GFARC_GUIDELINES_2020.pdf"),
    ("ICRC", "https://www.icrc.org/sites/default/files/external/doc/en/assets/files/publications/icrc-002-0526.pdf"),
]

chunks = []
meta = []

for name, url in sources:
    print("Downloading:", name)
    pdf = requests.get(url).content
    text = extract_pdf(pdf)
    ch = chunk_text(text)

    for i, c in enumerate(ch):
        chunks.append(c)
        meta.append({"source": name})

print("Embedding...")
emb = embedder.encode(chunks)

print("Building index...")
index = NearestNeighbors().fit(emb)

# Save
pickle.dump(index, open("index.pkl", "wb"))
pickle.dump(chunks, open("chunks.pkl", "wb"))
pickle.dump(meta, open("meta.pkl", "wb"))

print("✅ Ingestion complete")
