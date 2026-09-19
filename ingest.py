"""
ingest.py
---------
Step 1 of the RAG pipeline.

What this does:
1. Reads every PDF inside the /data folder (your syllabus, exam rules,
   placement brochure, etc.)
2. Splits each document into small overlapping chunks (so the model
   only ever has to read relevant snippets, not entire PDFs)
3. Converts each chunk into a vector (embedding) using a free,
   local sentence-transformer model
4. Stores all vectors in a FAISS index on disk, so chatbot.py / app.py
   can search them instantly without re-processing the PDFs every time

Run this ONCE whenever you add or change files inside /data:
    python ingest.py
"""

import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # free, runs locally, no API key


def load_documents():
    """Load every PDF found in the data/ folder."""
    if not os.path.isdir(DATA_DIR) or not os.listdir(DATA_DIR):
        raise FileNotFoundError(
            f"No files found in '{DATA_DIR}/'. Put your college PDFs "
            f"(syllabus, exam rules, placement info, etc.) there first."
        )

    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} page(s) from PDFs in '{DATA_DIR}/'.")
    return documents


def split_documents(documents):
    """Break long documents into small, overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,        # characters per chunk
        chunk_overlap=150,     # overlap so context isn't lost at chunk edges
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def build_vectorstore(chunks):
    """Embed chunks and save them into a local FAISS vector store."""
    print(f"Loading embedding model '{EMBEDDING_MODEL}' (first run downloads it)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Embedding chunks and building FAISS index... this may take a minute.")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"Vector store saved to '{VECTORSTORE_DIR}/'.")


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    build_vectorstore(chunks)
    print("\nIngestion complete. You can now run chatbot.py or app.py.")
