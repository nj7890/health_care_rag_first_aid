import requests
from io import BytesIO
import PyPDF2
from typing import List, Dict

# Import RAGEngine from your Streamlit app file
from streamlit_first_aid_rag_app import RAGEngine

def download_pdf(url: str) -> bytes:
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.content

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)

def ingest_document(rag_engine: RAGEngine, url: str, source_id: str):
    print(f"[{source_id}] Downloading {url} ...")
    pdf_bytes = download_pdf(url)
    print(f"[{source_id}] Extracting text …")
    txt = extract_text_from_pdf_bytes(pdf_bytes)
    print(f"[{source_id}] Text length: {len(txt)} characters")
    rag_engine.add_source(txt, source_id=source_id)
    print(f"[{source_id}] Ingested into knowledge base.")

def ingest_multiple(rag_engine: RAGEngine, sources: List[Dict[str, str]]):
    for src in sources:
        url = src["url"]
        sid = src["source_id"]
        try:
            ingest_document(rag_engine, url, sid)
        except Exception as e:
            print(f"Failed to ingest {sid} ({url}): {e}")

if __name__ == "__main__":
    rag = RAGEngine(embed_model_name="sentence-transformers/all-MiniLM-L6-v2")

    sources = [
        {
            "url": "https://www.ifrc.org/sites/default/files/2022-02/EN_GFARC_GUIDELINES_2020.pdf",
            "source_id": "IFRC_FirstAid_Guidelines_2020",
        },
        {
            "url": "https://www.icrc.org/sites/default/files/external/doc/en/assets/files/publications/icrc-002-0526.pdf",
            "source_id": "ICRC_FirstAid_Booklet",
        },
        {
            "url": "https://cdn.redcross.ca/prodmedia/crc/documents/Comprehensive_Guide_for_FirstAidCPR_en.pdf",
            "source_id": "CanadianRedCross_FirstAidCPR",
        },
        # Add more here if you want
    ]

    ingest_multiple(rag, sources)
    print("✅ All downloads + ingests finished (in this process).")
