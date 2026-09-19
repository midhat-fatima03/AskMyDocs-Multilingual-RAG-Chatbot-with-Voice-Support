"""
rag_chain.py
------------
The core "brain" of the chatbot. Shared by both chatbot.py (CLI) and
app.py (Streamlit UI) so there's one place that defines how retrieval
and generation work.

Pipeline:
  question -> embed question -> search FAISS for closest chunks
           -> stuff chunks + question into a prompt -> LLM generates answer
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"  # fast + free tier on Groq 

PROMPT_TEMPLATE = """You are a helpful assistant answering questions about a
university using only the context below, which comes from the college's own
documents (syllabus, exam rules, placement information, etc.).

- If the answer is in the context, answer it clearly and concisely.
- If the answer is NOT in the context, say "I couldn't find that in the
  provided documents" instead of guessing.
- Quote specific rules, dates, or numbers exactly as they appear in the context.

Context:
{context}

Question: {question}

Answer:"""


def load_qa_chain():
    """Load the vector store and wire it up to the LLM as a QA chain."""
    if not os.path.isdir(VECTORSTORE_DIR):
        raise FileNotFoundError(
            "No vector store found. Run 'python ingest.py' first to process "
            "your PDFs in the data/ folder."
        )

    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your "
            "free key from https://console.groq.com/keys"
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # top 4 chunks

    llm = ChatGroq(model=LLM_MODEL, temperature=0.2)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa_chain


def ask(qa_chain, question: str):
    """Run one question through the chain and return answer + sources."""
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    sources = result.get("source_documents", [])
    return answer, sources
