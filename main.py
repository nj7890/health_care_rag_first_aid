# streamlit_first_aid_rag_app.py
# Minimal Streamlit RAG-style first-aid education app using open-source components.
# Features:
# - Upload authoritative documents (PDF / plain text / HTML) or paste text
# - Chunking + embeddings (sentence-transformers)
# - In-memory vector search (FAISS if available, fallback to sklearn)
# - Deterministic answer assembly: returns short summary + numbered steps using retrieved excerpts
# - Emergency detection (keywords) and clear "Call emergency services" banner
# - Shows sources and exact excerpts used (traceability)

import streamlit as st
from io import BytesIO
import tempfile
import os
import re
from typing import List, Dict, Any, Tuple

# Text extraction
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# Embeddings
from sentence_transformers import SentenceTransformer

# Vector search
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False
    from sklearn.neighbors import NearestNeighbors

import numpy as np

# ---------------------- Utility functions ----------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if PyPDF2 is None:
        raise RuntimeError("PyPDF2 not installed. Please install PyPDF2 to support PDF uploads.")
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    texts = []
    for p in reader.pages:
        try:
            texts.append(p.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n\n".join(texts)


def clean_text(t: str) -> str:
    # basic cleaning
    t = t.replace('\r', '\n')
    t = re.sub('\n{3,}', '\n\n', t)
    return t.strip()


def chunk_text(text: str, chunk_size: int = 400, stride: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - stride
    return chunks


def emergency_check(query: str, chunks_texts: List[str]) -> Tuple[bool, List[str]]:
    # Keywords that indicate life-threatening situations
    keywords = [
        'not breathing', 'no pulse', 'unconscious', 'severe bleeding', 'heavy bleeding',
        'choking', 'cardiac arrest', 'no breathing', 'unresponsive', 'loss of consciousness'
    ]
    found = []
    qlow = query.lower()
    for k in keywords:
        if k in qlow:
            found.append(k)
    # Also check retrieved chunks for presence of keywords
    for t in chunks_texts:
        tl = t.lower()
        for k in keywords:
            if k in tl and k not in found:
                found.append(k)
    return (len(found) > 0, found)


# ---------------------- RAG Engine ----------------------
class RAGEngine:
    def __init__(self, embed_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer(embed_model_name)
        self.metadata: List[Dict[str, Any]] = []
        self.text_chunks: List[str] = []
        self.embeddings = None
        self.index = None
        self._use_faiss = _HAS_FAISS

    def add_source(self, text: str, source_id: str, section: str = ''):
        text = clean_text(text)
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            meta = {'source_id': source_id, 'section': section, 'chunk_id': f"{source_id}__{i}"}
            self.metadata.append(meta)
            self.text_chunks.append(c)
        # rebuild embeddings lazily
        self._rebuild_index()

    def _rebuild_index(self):
        if len(self.text_chunks) == 0:
            self.embeddings = None
            self.index = None
            return
        self.embeddings = self.embedder.encode(self.text_chunks, convert_to_numpy=True, show_progress_bar=False)
        if self._use_faiss:
            d = self.embeddings.shape[1]
            idx = faiss.IndexFlatL2(d)
            idx.add(self.embeddings)
            self.index = idx
        else:
            # sklearn NearestNeighbors
            self.index = NearestNeighbors(n_neighbors=min(10, len(self.embeddings)), algorithm='auto').fit(self.embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[float, str, Dict[str, Any]]]:
        if self.index is None:
            return []
        q_emb = self.embedder.encode([query], convert_to_numpy=True)[0]
        if self._use_faiss:
            D, I = self.index.search(np.expand_dims(q_emb, axis=0), top_k)
            results = []
            for dist, idx in zip(D[0], I[0]):
                results.append((float(dist), self.text_chunks[int(idx)], self.metadata[int(idx)]))
            return results
        else:
            dists, idxs = self.index.kneighbors([q_emb], n_neighbors=min(top_k, len(self.text_chunks)))
            results = []
            for dist, idx in zip(dists[0], idxs[0]):
                results.append((float(dist), self.text_chunks[int(idx)], self.metadata[int(idx)]))
            return results

    def assemble_answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        retrieved = self.retrieve(query, top_k=top_k)
        snippets = [r[1] for r in retrieved]
        metas = [r[2] for r in retrieved]

        is_emergency, found_k = emergency_check(query, snippets)

        # Deterministic assembly: create short summary + numbered steps by extracting imperative sentences
        summary = 'Based on authoritative sources, here are the main recommended steps.'

        # Extract candidate sentences that look like steps (start with verbs or contain 'call', 'apply', etc.)
        candidate_sentences = []
        for s, m in zip(snippets, metas):
            sentences = re.split(r'(?<=[.!?])\s+', s)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 10: continue
                # naive heuristic: imperative sentences often start with a verb or 'call'
                if re.match(r'^(Call|Apply|Stop|Place|Hold|Lay|Perform|Check|Ensure|Give|Tilt|Compress)\b', sent, flags=re.I) or 'call' in sent.lower() or 'apply' in sent.lower() or 'compress' in sent.lower():
                    candidate_sentences.append({'text': sent, 'meta': m})
        # If not many candidates, fall back to top sentences from snippets
        if len(candidate_sentences) < 3:
            for s, m in zip(snippets, metas):
                sentences = re.split(r'(?<=[.!?])\s+', s)
                for sent in sentences[:3]:
                    sent = sent.strip()
                    if len(sent) >= 10:
                        candidate_sentences.append({'text': sent, 'meta': m})
        # Deduplicate while preserving order
        seen = set()
        ordered = []
        for c in candidate_sentences:
            key = c['text'][:120]
            if key in seen: continue
            seen.add(key)
            ordered.append(c)
            if len(ordered) >= 6:
                break

        steps = [c['text'] for c in ordered]

        sources = []
        for r in retrieved:
            _, text, meta = r
            sources.append({'source_id': meta.get('source_id', ''), 'excerpt': text[:500]})

        return {
            'summary': summary,
            'steps': steps,
            'sources': sources,
            'is_emergency': is_emergency,
            'emergency_keys': found_k
        }


# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title='First-Aid RAG (Open Source) — Streamlit', layout='wide')
st.title('First-Aid Education — RAG (open-source)')
st.markdown(
    """
    This is a minimal Retrieval-Augmented-Generation (RAG) style Streamlit app that **assembles first-aid guidance** from uploaded authoritative documents.

    **Design choices:**
    - Uses sentence-transformers embeddings + in-memory vector search.
    - Answers are constructed from retrieved authoritative excerpts (no external LLM required).
    - Emergency detection flags life-threatening queries and shows a prominent "Call emergency services" banner.

    Upload PDFs or paste text to build the knowledge base, then ask first-aid questions.
    """
)

# Sidebar: model info and ingestion
with st.sidebar:
    st.header('Knowledge Base')
    st.info('Upload authoritative documents (PDF / plain text) or paste text. Each upload becomes a source.')

    uploaded_file = st.file_uploader('Upload PDF / TXT', type=['pdf', 'txt'], accept_multiple_files=False)
    text_paste = st.text_area('Or paste text (e.g., IFRC excerpts)')
    src_id = st.text_input('Source ID (name)', value='source_' + str(np.random.randint(1000,9999)))

    if st.button('Add source'):
        if uploaded_file is None and text_paste.strip() == '':
            st.warning('Provide a PDF / TXT file or paste text to add a source.')
        else:
            if 'rag_engine' not in st.session_state:
                st.session_state.rag_engine = RAGEngine()
            rag: RAGEngine = st.session_state.rag_engine
            try:
                if uploaded_file is not None:
                    if uploaded_file.type == 'application/pdf':
                        data = uploaded_file.read()
                        txt = extract_text_from_pdf(data)
                    else:
                        data = uploaded_file.read().decode('utf-8')
                        txt = data
                    rag.add_source(txt, source_id=src_id)
                    st.success(f'Added source {src_id} with chunks: {len(chunk_text(txt))}')
                else:
                    rag.add_source(text_paste, source_id=src_id)
                    st.success(f'Added pasted text as {src_id} with chunks: {len(chunk_text(text_paste))}')
            except Exception as e:
                st.error(f'Failed to add source: {e}')

    if 'rag_engine' in st.session_state:
        rag: RAGEngine = st.session_state.rag_engine
        st.write(f"Sources: {len(rag.metadata)} chunks indexed")
        if st.button('Clear KB'):
            del st.session_state.rag_engine
            st.experimental_rerun()

st.markdown('---')

# Main area: query and results
col1, col2 = st.columns([2,3])

with col1:
    st.subheader('Ask a first-aid question')
    user_query = st.text_area('Describe the situation or ask a question', height=150)
    top_k = st.slider('Number of retrieved chunks (k)', min_value=1, max_value=10, value=5)
    if st.button('Get guidance'):
        if 'rag_engine' not in st.session_state:
            st.warning('Knowledge base is empty. Upload at least one authoritative source in the sidebar.')
        elif user_query.strip() == '':
            st.warning('Please enter a question or description of the situation.')
        else:
            rag: RAGEngine = st.session_state.rag_engine
            with st.spinner('Retrieving and assembling answer...'):
                ans = rag.assemble_answer(user_query, top_k=top_k)
            # Emergency banner
            if ans['is_emergency']:
                st.error('POTENTIAL EMERGENCY DETECTED — CALL EMERGENCY SERVICES IMMEDIATELY. Keywords: ' + ', '.join(ans['emergency_keys']))

            st.markdown('**Summary**')
            st.write(ans['summary'])
            st.markdown('**Step-by-step guidance (assembled from retrieved sources)**')
            if len(ans['steps']) == 0:
                st.info('No clear procedural steps found in retrieved excerpts. Below are the retrieved excerpts to consult.')
            else:
                for i, s in enumerate(ans['steps'], start=1):
                    st.markdown(f"**{i}.** {s}")

            st.markdown('**Sources & excerpts**')
            for s in ans['sources']:
                st.write('Source:', s['source_id'])
                st.caption(s['excerpt'][:1000])

with col2:
    st.subheader('KB Explorer')
    if 'rag_engine' not in st.session_state:
        st.info('No sources indexed yet.')
    else:
        rag: RAGEngine = st.session_state.rag_engine
        st.write(f'Indexed chunks: {len(rag.text_chunks)}')
        q_explore = st.text_input('Search KB (semantic) — enter a phrase to find related chunks')
        if st.button('Search KB'):
            if q_explore.strip() == '':
                st.warning('Enter a search phrase.')
            else:
                res = rag.retrieve(q_explore, top_k=10)
                for dist, text, meta in res:
                    st.write('Source:', meta.get('source_id', ''))
                    st.caption(text[:800])
                    st.write('Distance:', round(dist, 4))

st.markdown('---')
st.caption('This app assembles answers by selecting and presenting authoritative excerpts. It does not perform open-ended medical generation; for safety, steps are taken from the indexed sources. If you want an LLM-based final phrasing, integrate a local open-source model and replace assemble_answer with a model call.');
