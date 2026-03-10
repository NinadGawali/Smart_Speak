# ==========================================================
# Imports
# ==========================================================

import os
from pathlib import Path
from dotenv import load_dotenv
import json
import io
import time
import datetime

import streamlit as st
import numpy as np
import torch
from transformers import pipeline
import soundfile as sf

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document



# ==========================================================
# Page Config
# ==========================================================

st.set_page_config(
    page_title="Smart Speech Evaluator",
    page_icon="🎙️",
    layout="wide"
)


# ==========================================================
# Load .env from project root
# ==========================================================

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found in .env file")
    st.stop()


# ==========================================================
# Recordings Directory & Documents Directory
# ==========================================================

RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"
DOCUMENTS_DIR.mkdir(exist_ok=True)

VECTORSTORE_DIR = Path(__file__).resolve().parent / "vectorstore"
VECTORSTORE_DIR.mkdir(exist_ok=True)


# ==========================================================
# Load Whisper STT Model (lazy — only when needed)
# ==========================================================

@st.cache_resource
def load_stt_model():
    return pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-base",
        device=0 if torch.cuda.is_available() else -1,
        chunk_length_s=30,
        stride_length_s=5
    )


# ==========================================================
# Load Embeddings Model
# ==========================================================

# from langchain.embeddings import HuggingFaceEmbeddings
# import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ==========================================================
# Document Processing Functions
# ==========================================================

def process_uploaded_document(uploaded_file):
    """Process uploaded PDF or TXT file and return text content."""
    try:
        # Save uploaded file temporarily
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = uploaded_file.name.split('.')[-1].lower()
        file_path = DOCUMENTS_DIR / f"document_{timestamp}.{file_extension}"
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load document based on type
        if file_extension == 'pdf':
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
        elif file_extension == 'txt':
            loader = TextLoader(str(file_path), encoding='utf-8')
            documents = loader.load()
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None, None
        
        return documents, file_path
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return None, None


def create_vectorstore(documents):
    """Create FAISS vector store from documents."""
    try:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = load_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Save vector store
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        vectorstore_path = VECTORSTORE_DIR / f"vectorstore_{timestamp}"
        vectorstore.save_local(str(vectorstore_path))
        
        return vectorstore, len(chunks), vectorstore_path
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None, 0, None


# ==========================================================
# Audio Processing Functions
# ==========================================================

def save_recorded_audio(audio_data):
    """Save recorded audio from st.audio_input to recordings folder."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = RECORDINGS_DIR / f"recording_{timestamp}.wav"
        
        with open(file_path, "wb") as f:
            f.write(audio_data.getbuffer())
        
        return file_path
    except Exception as e:
        st.error(f"Error saving recorded audio: {e}")
        return None


def load_audio_bytes(file_path):
    """Read a saved WAV file as bytes for st.audio."""
    with open(file_path, "rb") as f:
        return f.read()


def speech_to_text(file_path):
    """Transcribe a saved WAV file using Whisper."""
    stt = load_stt_model()
    audio_data, sr = sf.read(str(file_path), dtype="float32")
    # Ensure mono
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    result = stt({"array": audio_data, "sampling_rate": sr})
    return result["text"].strip()


def split_transcript_into_chunks(transcript, chunk_size=3):
    """Split transcript into sentence-based chunks."""
    # Split by common sentence endings
    import re
    sentences = re.split(r'(?<=[.!?])\s+', transcript)
    
    # Group sentences into chunks
    chunks = []
    temp_chunk = []
    
    for sentence in sentences:
        temp_chunk.append(sentence)
        if len(temp_chunk) >= chunk_size:
            chunks.append(' '.join(temp_chunk))
            temp_chunk = []
    
    # Add remaining sentences
    if temp_chunk:
        chunks.append(' '.join(temp_chunk))
    
    return chunks


def evaluate_speech_with_rag(transcript, vectorstore):
    """Evaluate speech transcript against reference content using RAG."""
    try:
        # Split transcript into chunks
        transcript_chunks = split_transcript_into_chunks(transcript, chunk_size=3)
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=GEMINI_API_KEY
        )
        
        chunk_evaluations = []
        
        # Evaluate each chunk against the vector store
        for i, chunk in enumerate(transcript_chunks):
            # Retrieve relevant context from vector store
            relevant_docs = vectorstore.similarity_search(chunk, k=3)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            prompt = ChatPromptTemplate.from_template("""
You are an expert speech evaluator analyzing a segment of a speech against reference material.

Reference Material:
{context}

Speech Segment {chunk_num}:
"{chunk}"

Evaluate this specific segment on:
1. Accuracy: Does it match the reference content?
2. Completeness: Are key points covered?
3. Clarity: Is it well articulated?

Provide evaluation in JSON format:
{{"score": <0-100>, "accuracy": <0-100>, "completeness": <0-100>, "clarity": <0-100>, "feedback": "<specific feedback>", "issues": ["<issue1>", "<issue2>"]}}
""")
            
            chain = prompt | llm | StrOutputParser()
            response = chain.invoke({
                "context": context,
                "chunk": chunk,
                "chunk_num": i + 1
            }).strip()
            
            # Clean JSON
            if response.startswith("```json"):
                response = response[7:-3]
            elif response.startswith("```"):
                response = response[3:-3]
            
            evaluation = json.loads(response)
            evaluation['chunk_text'] = chunk
            evaluation['chunk_number'] = i + 1
            chunk_evaluations.append(evaluation)
        
        # Generate overall evaluation
        overall_prompt = ChatPromptTemplate.from_template("""
You are an expert speech coach providing comprehensive feedback.

Transcript:
{transcript}

Chunk-by-Chunk Evaluations:
{chunk_evals}

Provide a comprehensive evaluation report in JSON format:
{{
    "overall_score": <0-100>,
    "strengths": ["<strength1>", "<strength2>"],
    "weaknesses": ["<weakness1>", "<weakness2>"],
    "specific_improvements": [
        {{"line": "<specific line from transcript>", "suggestion": "<how to improve it>"}},
        {{"line": "<specific line from transcript>", "suggestion": "<how to improve it>"}}
    ],
    "common_mistakes": ["<mistake1>", "<mistake2>"],
    "summary": "<overall summary of performance>"
}}
""")
        
        chain = overall_prompt | llm | StrOutputParser()
        overall_response = chain.invoke({
            "transcript": transcript,
            "chunk_evals": json.dumps(chunk_evaluations, indent=2)
        }).strip()
        
        # Clean JSON
        if overall_response.startswith("```json"):
            overall_response = overall_response[7:-3]
        elif overall_response.startswith("```"):
            overall_response = overall_response[3:-3]
        
        overall_evaluation = json.loads(overall_response)
        
        return {
            "chunk_evaluations": chunk_evaluations,
            "overall_evaluation": overall_evaluation,
            "transcript_chunks": transcript_chunks
        }
        
    except Exception as e:
        st.error(f"Error in evaluation: {e}")
        return None


# ==========================================================
# Initialize Session State
# ==========================================================

defaults = {
    "uploaded_doc": None,       # Uploaded document file
    "vectorstore": None,        # FAISS vector store
    "num_chunks": 0,            # Number of chunks in vectorstore
    "saved_path": None,         # Path to saved WAV file
    "transcript": None,
    "evaluation_result": None,
    "pipeline_running": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ==========================================================
# Main UI
# ==========================================================

st.title("🎙️ Smart Speech Evaluator with RAG")
st.markdown("*Evaluate your speech against reference documents using AI-powered analysis*")
st.markdown("---")

# ── STEP 1: Upload Reference Document ────────────────────
st.subheader("📄 Step 1 — Upload Reference Document")
st.caption("Upload a PDF or TXT file containing the topic/speech content you want to evaluate against.")

if st.session_state.vectorstore is None:
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=['pdf', 'txt'],
        help="Upload the reference document for your speech topic"
    )
    
    if uploaded_file is not None:
        with st.spinner("📚 Processing document and creating knowledge base..."):
            documents, doc_path = process_uploaded_document(uploaded_file)
            
            if documents:
                st.success(f"✅ Loaded {len(documents)} document page(s)")
                
                with st.status("🔍 Creating vector embeddings...", expanded=True) as status:
                    st.write("Splitting document into chunks...")
                    time.sleep(0.3)
                    st.write("Generating embeddings...")
                    
                    vectorstore, num_chunks, vs_path = create_vectorstore(documents)
                    
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.num_chunks = num_chunks
                        st.session_state.uploaded_doc = uploaded_file.name
                        status.update(
                            label=f"✅ Knowledge base created ({num_chunks} chunks)",
                            state="complete"
                        )
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        status.update(label="❌ Failed to create knowledge base", state="error")
else:
    st.success(f"✅ **Document Loaded:** {st.session_state.uploaded_doc}")
    st.info(f"📊 **Knowledge Base:** {st.session_state.num_chunks} text chunks indexed")


# ── STEP 2: Record Audio ─────────────────────────────────
if st.session_state.vectorstore is not None:
    st.markdown("---")
    st.subheader("🎤 Step 2 — Record Your Speech")
    st.caption("Click the microphone button below to start recording, then click stop when finished.")
    
    if st.session_state.saved_path is None:
        audio_input = st.audio_input("Record your audio")
        
        if audio_input is not None:
            with st.spinner("💾 Saving recorded audio..."):
                saved_path = save_recorded_audio(audio_input)
                if saved_path:
                    st.session_state.saved_path = saved_path
                    st.success(f"✅ Audio recorded and saved → `recordings/{saved_path.name}`")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Failed to save the recording. Please try again.")


# ── STEP 3: Playback + Transcribe & Analyse ──────────────
if st.session_state.saved_path is not None:
    st.markdown("---")
    st.subheader("🎧 Step 3 — Review Audio")
    
    audio_bytes = load_audio_bytes(st.session_state.saved_path)
    st.audio(audio_bytes, format="audio/wav")
    st.caption(f"📁 File: `{st.session_state.saved_path.name}`")
    
    st.markdown("---")
    st.subheader("🚀 Step 4 — Analyze Speech")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_pipeline = st.button(
            "🔍 Transcribe & Analyze with RAG",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.pipeline_running
        )
    
    if run_pipeline:
        st.session_state.pipeline_running = True
        
        # ── Sub-step A: Transcription ──────────────────
        with st.status("🔊 Step 4a — Speech-to-Text (Whisper)", expanded=True) as status_stt:
            st.write("Loading Whisper model…")
            time.sleep(0.3)
            st.write("Transcribing audio…")
            try:
                transcript = speech_to_text(st.session_state.saved_path)
                st.session_state.transcript = transcript
                status_stt.update(label="✅ Step 4a — Transcription complete", state="complete")
            except Exception as e:
                status_stt.update(label="❌ Transcription failed", state="error")
                st.error(str(e))
                st.session_state.pipeline_running = False
                st.stop()
        
        st.info(f"**Transcript:** {st.session_state.transcript}")
        
        # ── Sub-step B: RAG-based Evaluation ───────────
        with st.status("🤖 Step 4b — RAG-based Evaluation (Gemini)", expanded=True) as status_rag:
            st.write("Splitting transcript into chunks...")
            st.write("Retrieving relevant context from knowledge base...")
            st.write("Evaluating each segment against reference material...")
            st.write("Generating comprehensive feedback...")
            
            try:
                result = evaluate_speech_with_rag(
                    st.session_state.transcript,
                    st.session_state.vectorstore
                )
                
                if result:
                    st.session_state.evaluation_result = result
                    status_rag.update(label="✅ Step 4b — Evaluation complete", state="complete")
                else:
                    status_rag.update(label="❌ Evaluation failed", state="error")
                    
            except Exception as e:
                status_rag.update(label="❌ Evaluation failed", state="error")
                st.error(str(e))
                st.session_state.pipeline_running = False
                st.stop()
        
        st.session_state.pipeline_running = False
        st.rerun()


# ── STEP 5: Detailed Results ──────────────────────────────
if st.session_state.evaluation_result is not None:
    st.markdown("---")
    st.subheader("📊 Step 5 — Comprehensive Evaluation Report")
    
    result = st.session_state.evaluation_result
    overall = result.get("overall_evaluation", {})
    chunk_evals = result.get("chunk_evaluations", [])
    
    # Overall Score Card
    overall_score = overall.get("overall_score", 0)
    
    col1, col2, col3 = st.columns(3)
    with col2:
        if overall_score >= 80:
            st.success(f"### 🎯 Overall Score: {overall_score}/100")
        elif overall_score >= 60:
            st.warning(f"### ⚠️ Overall Score: {overall_score}/100")
        else:
            st.error(f"### ❌ Overall Score: {overall_score}/100")
    
    st.progress(overall_score / 100)
    
    # Summary
    st.markdown("#### 📝 Summary")
    st.info(overall.get("summary", "No summary available"))
    
    # Strengths and Weaknesses
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Strengths")
        strengths = overall.get("strengths", [])
        if strengths:
            for strength in strengths:
                st.success(f"• {strength}")
        else:
            st.write("No specific strengths identified")
    
    with col2:
        st.markdown("#### ⚠️ Areas for Improvement")
        weaknesses = overall.get("weaknesses", [])
        if weaknesses:
            for weakness in weaknesses:
                st.warning(f"• {weakness}")
        else:
            st.write("No specific weaknesses identified")
    
    # Specific Line-by-Line Improvements
    st.markdown("---")
    st.markdown("#### 🔧 Specific Improvements")
    st.caption("Lines that need improvement and how to fix them:")
    
    improvements = overall.get("specific_improvements", [])
    if improvements:
        for i, improvement in enumerate(improvements, 1):
            with st.expander(f"💡 Improvement {i}: \"{improvement.get('line', '')[:60]}...\""):
                st.markdown(f"**Original Line:**")
                st.code(improvement.get('line', ''), language=None)
                st.markdown(f"**Suggestion:**")
                st.info(improvement.get('suggestion', ''))
    else:
        st.write("No specific line improvements needed")
    
    # Common Mistakes
    st.markdown("---")
    st.markdown("#### 🚫 Common Mistakes Identified")
    mistakes = overall.get("common_mistakes", [])
    if mistakes:
        for mistake in mistakes:
            st.error(f"• {mistake}")
    else:
        st.success("✅ No common mistakes identified!")
    
    # Chunk-by-Chunk Analysis (Expandable)
    st.markdown("---")
    st.markdown("#### 📋 Detailed Chunk-by-Chunk Analysis")
    
    for chunk_eval in chunk_evals:
        chunk_num = chunk_eval.get('chunk_number', 0)
        chunk_score = chunk_eval.get('score', 0)
        chunk_text = chunk_eval.get('chunk_text', '')
        
        # Color code based on score
        if chunk_score >= 80:
            emoji = "🟢"
        elif chunk_score >= 60:
            emoji = "🟡"
        else:
            emoji = "🔴"
        
        with st.expander(f"{emoji} Chunk {chunk_num} — Score: {chunk_score}/100"):
            st.markdown(f"**Text:**")
            st.code(chunk_text, language=None)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                accuracy = chunk_eval.get('accuracy', 0)
                st.metric("Accuracy", f"{accuracy}/100")
            with col2:
                completeness = chunk_eval.get('completeness', 0)
                st.metric("Completeness", f"{completeness}/100")
            with col3:
                clarity = chunk_eval.get('clarity', 0)
                st.metric("Clarity", f"{clarity}/100")
            
            st.markdown(f"**Feedback:**")
            st.info(chunk_eval.get('feedback', 'No feedback'))
            
            issues = chunk_eval.get('issues', [])
            if issues:
                st.markdown(f"**Issues:**")
                for issue in issues:
                    st.warning(f"• {issue}")
    
    # Transcript Display
    st.markdown("---")
    st.markdown("#### 📄 Full Transcript")
    with st.expander("View Full Transcript"):
        st.text_area("Transcript", st.session_state.transcript, height=200, disabled=True)


# ── Reset ─────────────────────────────────────────────────
if st.session_state.vectorstore is not None:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔄 New Recording", type="secondary", use_container_width=True):
            st.session_state.saved_path = None
            st.session_state.transcript = None
            st.session_state.evaluation_result = None
            st.session_state.pipeline_running = False
            st.rerun()
    
    with col3:
        if st.button("📄 New Document", type="secondary", use_container_width=True):
            for key in defaults.keys():
                st.session_state[key] = defaults[key]
            st.rerun()


# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🎙️ Smart Speech Evaluator with RAG | Powered by Whisper + Gemini 2.0 Flash + FAISS"
    "</div>",
    unsafe_allow_html=True
)