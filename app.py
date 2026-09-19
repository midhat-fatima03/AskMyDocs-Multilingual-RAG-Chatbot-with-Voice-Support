"""
app.py (v3 — quiz generation + multi-language + voice)
-------------------------------------------------------
Builds on v2 (in-browser PDF upload, no manual folder steps) and adds:

1. Auto-generated study quiz from the uploaded document (LLM generates
   multiple-choice questions directly from the document content).
2. Multi-language Q&A — answer in English, Hindi, or Kannada regardless
   of what language the question was typed/spoken in.
3. Voice input (speak your question, transcribed via Groq's Whisper) and
   voice output (answer read aloud using gTTS).

Run:
    streamlit run app.py

Extra one-time setup for this version: nothing beyond your existing
GROQ_API_KEY — Whisper transcription uses the same Groq account.
"""

import os
import io
import json
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from groq import Groq as GroqClient  # direct client, used for Whisper transcription
from gtts import gTTS

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"
WHISPER_MODEL = "whisper-large-v3"

LANGUAGE_OPTIONS = {
    "Auto (same as question)": None,
    "English": "English",
    "Hindi": "Hindi",
    "Kannada": "Kannada",
}
GTTS_LANG_CODES = {"English": "en", "Hindi": "hi", "Kannada": "kn"}
PROMPT_TEMPLATE = """You are a helpful assistant answering questions using
only the context below, which comes from documents the user has uploaded.

- If the answer is in the context, answer it clearly and concisely.
- If the answer is NOT in the context, say "I couldn't find that in the
  uploaded documents" instead of guessing.
- Quote specific rules, dates, or numbers exactly as they appear in the context.
- {language_instruction}

Context:
{{context}}

Question: {{input}}

Answer:"""

st.set_page_config(page_title="Document Q&A Chatbot", page_icon="📄")
st.title("📄 Ask Questions About Your Documents")
st.caption(
    "Upload one or more PDFs, then ask questions by typing or speaking, "
    "in English, Hindi, or Kannada. You can also generate a study quiz "
    "straight from the document."
)


# ---------------- Cached resources ----------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_llm():
    if not os.getenv("GROQ_API_KEY"):
        st.error(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your "
            "free key from https://console.groq.com/keys, then restart the app."
        )
        st.stop()
    return ChatGroq(model=LLM_MODEL, temperature=0.2)


def get_groq_client():
    return GroqClient(api_key=os.getenv("GROQ_API_KEY"))


# ---------------- Document processing ----------------
def process_uploaded_pdfs(uploaded_files):
    """Chunk + embed uploaded PDFs into an in-memory FAISS vector store."""
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded_file in uploaded_files:
            tmp_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = uploaded_file.name

            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    return vectorstore, all_chunks


def build_qa_chain(vectorstore, language_choice):
    if LANGUAGE_OPTIONS[language_choice] is None:
        language_instruction = "Answer in the same language the question was asked in."
    else:
        language_instruction = (
            f"Answer in {LANGUAGE_OPTIONS[language_choice]}, regardless of what "
            f"language the question was asked in."
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = get_llm()
    filled_template = PROMPT_TEMPLATE.format(language_instruction=language_instruction)
    prompt = ChatPromptTemplate.from_template(filled_template)

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


# ---------------- Feature: quiz generation ----------------
def generate_quiz(chunks, num_questions=5):
    """
    Ask the LLM to generate multiple-choice study questions directly from
    the uploaded document's content. Returns a list of dicts:
    {question, options: [...], correct_index, explanation}
    """
    llm = get_llm()
    combined_text = "\n\n".join(c.page_content for c in chunks)[:8000]  # keep prompt small

    prompt = f"""Based on the following document content, generate {num_questions}
multiple-choice study questions to help a student revise. Vary the difficulty.

Return ONLY valid JSON (no markdown code fences, no extra commentary), in
exactly this format:
[
  {{"question": "...", "options": ["...", "...", "...", "..."], "correct_index": 0, "explanation": "..."}}
]

Document content:
{combined_text}
"""
    response = llm.invoke(prompt)
    text = response.content.strip()

    # Defensive cleanup in case the model wraps output in code fences anyway
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    return json.loads(text)


# ---------------- Feature: voice ----------------
def transcribe_audio(audio_bytes: bytes) -> str:
    """Send recorded audio to Groq's Whisper model and return the transcript text."""
    client = get_groq_client()
    transcript = client.audio.transcriptions.create(
        file=("question.wav", audio_bytes),
        model=WHISPER_MODEL,
    )
    return transcript.text


def synthesize_speech(text: str, language_choice: str) -> bytes:
    """Convert answer text to speech (mp3 bytes) using gTTS."""
    lang_name = LANGUAGE_OPTIONS[language_choice] or "English"
    lang_code = GTTS_LANG_CODES.get(lang_name, "en")
    tts = gTTS(text=text, lang=lang_code)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ---------------- Session state ----------------
for key, default in {
    "vectorstore": None,
    "chunks": None,
    "qa_chain": None,
    "messages": [],
    "processed_files": [],
    "language_choice": "Auto (same as question)",
    "quiz": None,
    "quiz_answers": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------- Sidebar: upload + settings ----------------
with st.sidebar:
    st.header("1. Upload your documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        current_names = sorted(f.name for f in uploaded_files)
        if current_names != st.session_state.processed_files:
            with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
                vectorstore, chunks = process_uploaded_pdfs(uploaded_files)
                st.session_state.vectorstore = vectorstore
                st.session_state.chunks = chunks
                st.session_state.qa_chain = build_qa_chain(
                    vectorstore, st.session_state.language_choice
                )
                st.session_state.processed_files = current_names
                st.session_state.messages = []
                st.session_state.quiz = None
            st.success(f"Processed {len(uploaded_files)} file(s) into "
                       f"{len(st.session_state.chunks)} searchable chunks.")

    if st.session_state.processed_files:
        st.write("**Currently loaded:**")
        for name in st.session_state.processed_files:
            st.write(f"- {name}")

    st.divider()
    st.header("2. Answer language")
    new_language = st.selectbox(
        "Answer in:", list(LANGUAGE_OPTIONS.keys()),
        index=list(LANGUAGE_OPTIONS.keys()).index(st.session_state.language_choice),
    )
    if new_language != st.session_state.language_choice:
        st.session_state.language_choice = new_language
        if st.session_state.vectorstore is not None:
            st.session_state.qa_chain = build_qa_chain(
                st.session_state.vectorstore, new_language
            )

    voice_output_enabled = st.checkbox("Read answers aloud", value=False)

    st.divider()
    st.header("3. Study quiz")
    num_questions = st.slider("Number of questions", 3, 10, 5)
    if st.button("Generate quiz from document", disabled=not st.session_state.chunks):
        with st.spinner("Generating quiz questions..."):
            try:
                st.session_state.quiz = generate_quiz(st.session_state.chunks, num_questions)
                st.session_state.quiz_answers = {}
            except json.JSONDecodeError:
                st.error("The model didn't return valid quiz JSON. Try again.")

    st.divider()
    st.caption(
        "Nothing you upload is saved permanently — documents are processed "
        "in memory for this session only."
    )

# ---------------- Main area: tabs for Chat and Quiz ----------------
if not st.session_state.qa_chain:
    st.info("Upload at least one PDF in the sidebar to get started.")
    st.stop()

tab_chat, tab_quiz = st.tabs(["💬 Chat", "📝 Study Quiz"])

# ---- Chat tab ----
with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "audio" in msg:
                st.audio(msg["audio"], format="audio/mp3")

    st.write("**Ask by voice:**")
    audio_value = st.audio_input("Record your question")

    text_question = st.chat_input("...or type your question here")

    question = None
    if audio_value is not None:
        with st.spinner("Transcribing your voice..."):
            question = transcribe_audio(audio_value.getvalue())
    elif text_question:
        question = text_question

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                result = st.session_state.qa_chain.invoke({"input": question})
                answer = result["answer"]
                sources = result.get("context", [])

                audio_bytes = None
                if voice_output_enabled:
                    with st.spinner("Generating audio..."):
                        audio_bytes = synthesize_speech(answer, st.session_state.language_choice)
                        st.audio(audio_bytes, format="audio/mp3")

                if sources:
                    with st.expander("View sources"):
                        seen = set()
                        for doc in sources:
                            page = doc.metadata.get("page", "?")
                            page_display = int(page) + 1 if isinstance(page, int) else page
                            src = doc.metadata.get("source", "unknown file")
                            key = (src, page_display)
                            if key in seen:
                                continue
                            seen.add(key)
                            st.markdown(f"**{src}** — page {page_display}")
                            st.caption(doc.page_content[:300] + "...")

        assistant_msg = {"role": "assistant", "content": answer}
        if audio_bytes:
            assistant_msg["audio"] = audio_bytes
        st.session_state.messages.append(assistant_msg)

# ---- Quiz tab ----
with tab_quiz:
    if not st.session_state.quiz:
        st.info("Click **Generate quiz from document** in the sidebar to create "
                 "study questions from your uploaded PDF.")
    else:
        st.write(f"**{len(st.session_state.quiz)} question(s)** — pick an answer for each, "
                 "then check your score.")

        for i, q in enumerate(st.session_state.quiz):
            st.markdown(f"**Q{i + 1}. {q['question']}**")
            choice = st.radio(
                f"quiz_q_{i}", q["options"], key=f"quiz_radio_{i}",
                label_visibility="collapsed",
            )
            st.session_state.quiz_answers[i] = q["options"].index(choice)
            st.divider()

        if st.button("Check my answers"):
            score = 0
            for i, q in enumerate(st.session_state.quiz):
                user_answer = st.session_state.quiz_answers.get(i)
                correct = q["correct_index"]
                if user_answer == correct:
                    score += 1
                    st.success(f"Q{i + 1}: Correct! {q.get('explanation', '')}")
                else:
                    st.error(
                        f"Q{i + 1}: Incorrect. Correct answer: "
                        f"{q['options'][correct]}. {q.get('explanation', '')}"
                    )
            st.info(f"Score: {score} / {len(st.session_state.quiz)}")
