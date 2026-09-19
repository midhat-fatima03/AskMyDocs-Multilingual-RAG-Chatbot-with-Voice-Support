# College Documents Q&A Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about
your university — syllabus, exam rules, placement info — using your college's
own PDF documents as the source of truth, instead of relying on the LLM's
general knowledge (which would hallucinate specifics like dates and rules).

## How it works

```
Your PDFs (data/)
      │
      ▼
 ingest.py   → splits PDFs into chunks → embeds chunks → saves to FAISS
      │
      ▼
vectorstore/  (local vector database, created once)
      │
      ▼
rag_chain.py  → on each question:
                  1. embed the question
                  2. search FAISS for the most relevant chunks
                  3. pass those chunks + question to the LLM (Llama 3 via Groq)
                  4. LLM answers using ONLY that retrieved context
      │
      ▼
chatbot.py (CLI)  or  app.py (Streamlit web UI)
```

This is the same core pattern used in production RAG systems: chunking,
embeddings, vector search, and grounded generation — just scoped down to
run for free on your laptop.

## Setup

1. **Install dependencies** (Python 3.10+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

2. **Get a free Groq API key**: https://console.groq.com/keys (no credit
   card required). Groq hosts Llama 3 models and has a generous free tier —
   good for both development and demos.

3. **Set up your API key**:
   ```bash
   cp .env.example .env
   # then open .env and paste your key in place of your_groq_api_key_here
   ```

4. **Add your college documents**: drop PDF files (syllabus, exam rules,
   placement brochure, etc.) into the `data/` folder.

5. **Build the vector store** (run this once, and again any time you add
   new PDFs):
   ```bash
   python ingest.py
   ```

6. **Chat with it**:
   - Command line: `python chatbot.py`
   - Web UI: `streamlit run app.py`

## Project structure

```
college-rag-chatbot/
├── data/                  # your college PDFs go here
├── vectorstore/           # auto-generated FAISS index (created by ingest.py)
├── ingest.py              # step 1: process PDFs into the vector store
├── rag_chain.py           # core retrieval + generation logic
├── chatbot.py             # command-line chat interface
├── app.py                 # Streamlit web UI
├── requirements.txt
├── .env.example
└── README.md
```

## Extending this project (optional, good for interviews)

Small, honest upgrades that add real depth if you want to go further:

- **Multiple document types**: extend `ingest.py` to also load `.docx` or
  `.txt` files using other LangChain loaders.
- **Better chunking**: try chunking by section headers instead of raw
  character count, so answers don't cut off mid-rule.
- **Evaluation**: write a small set of test questions with known correct
  answers, and measure how often the chatbot gets them right — this shows
  interviewers you think about correctness, not just demos.
- **Deploy it**: push to GitHub, deploy the Streamlit app for free on
  Streamlit Community Cloud, and put the live link on your resume next to
  the repo link.

## Talking about this project in interviews / on your resume

Sample resume bullet (edit once it's actually built and deployed):

> Built a Retrieval-Augmented Generation (RAG) chatbot that answers
> questions about university documents (syllabus, exam rules, placement
> info) by combining FAISS vector search with an LLM, ensuring answers are
> grounded in real institutional documents rather than model hallucination.

Be ready to explain, in your own words:
- Why RAG instead of just asking an LLM directly (grounding, avoiding
  hallucination, no need to retrain the model on new data)
- What an embedding is, at a high level (a vector representation of text
  where similar meanings end up close together in vector space)
- Why you chunk documents instead of embedding the whole PDF at once
  (LLMs have limited context, and smaller chunks give more precise retrieval)
- One limitation of your system (e.g., it can't yet handle scanned/image-only
  PDFs without OCR, or it may retrieve irrelevant chunks for vague questions)
