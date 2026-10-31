import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(page_title="College Knowledge Assistant")

st.title("College Knowledge Assistant")

uploaded_files = st.file_uploader(
    "Upload at least 5 PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

question = st.text_input("Ask a Question")


def extract_text(pdf_file):
    text = ""

    reader = PdfReader(pdf_file)

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


if st.button("Process Documents"):

    if len(uploaded_files) < 5:
        st.error("Please upload at least 5 documents.")
        st.stop()

    all_chunks = []
    all_sources = []

    for file in uploaded_files:

        text = extract_text(file)

        chunks = [
            text[i:i+500]
            for i in range(0, len(text), 500)
        ]

        for chunk in chunks:
            all_chunks.append(chunk)
            all_sources.append(file.name)

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = model.encode(
        all_chunks
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        np.array(embeddings).astype("float32")
    )

    st.session_state.index = index
    st.session_state.model = model
    st.session_state.chunks = all_chunks
    st.session_state.sources = all_sources

    st.success("Knowledge Base Created")


if st.button("Answer"):

    if "index" not in st.session_state:
        st.error("Please process documents first.")
        st.stop()

    query_embedding = (
        st.session_state.model.encode([question])
    )

    distances, indices = (
        st.session_state.index.search(
            np.array(query_embedding).astype("float32"),
            k=3
        )
    )

    best_distance = distances[0][0]

    # threshold for unknown answers
    if best_distance > 1.5:

        st.warning(
            "Answer not found in knowledge base."
        )

    else:

        st.subheader("Answer")

        for idx in indices[0]:

            st.write(
                st.session_state.chunks[idx]
            )

            st.caption(
                f"Source: {st.session_state.sources[idx]}"
            )



    