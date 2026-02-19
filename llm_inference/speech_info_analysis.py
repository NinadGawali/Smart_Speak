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

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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
# Recordings Directory
# ==========================================================

RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)


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


def judge_relevance(expected_content, spoken_text):
    """Evaluate relevance using Gemini."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=GEMINI_API_KEY
    )
    prompt = ChatPromptTemplate.from_template("""
You are a strict speech relevance evaluator. Analyze how well the spoken content matches the expected content.

Expected Content:
{expected}

Spoken Content:
{spoken}

Evaluation Criteria:
- Topic relevance
- Key points coverage
- Content accuracy
- Overall alignment

Provide your evaluation in this exact JSON format:
{{"score": <integer 0-100>, "reason": "<concise explanation of score>"}}
""")
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "expected": expected_content.strip(),
        "spoken": spoken_text.strip()
    }).strip()

    if response.startswith("```json"):
        response = response[7:-3]
    elif response.startswith("```"):
        response = response[3:-3]

    return json.loads(response)


# ==========================================================
# Initialize Session State
# ==========================================================

defaults = {
    "saved_path": None,      # Path to saved WAV file
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

st.title("🎙️ Smart Speech Relevance Evaluator")
st.markdown("---")

# ── Expected content ──────────────────────────────────────
st.subheader("📘 Expected Content")
expected_content = st.text_area(
    "Enter the content you expect to hear:",
    height=100,
    placeholder="Type the expected speech content here..."
)

st.markdown("---")

# ── STEP 1 : Record Audio ─────────────────────────────────────
st.subheader("🎤 Step 1 — Record Your Speech")
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


# ── STEP 2 : Playback + Transcribe & Analyse button ───────
if st.session_state.saved_path is not None:
    st.markdown("---")
    st.subheader("🎧 Step 2 — Review Audio")

    audio_bytes = load_audio_bytes(st.session_state.saved_path)
    st.audio(audio_bytes, format="audio/wav")
    st.caption(f"📁 File: `{st.session_state.saved_path.name}`")

    st.markdown("---")
    st.subheader("🚀 Step 3 — Transcribe & Analyse")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_pipeline = st.button(
            "📝 Transcribe & Analyse",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.pipeline_running
        )

    if run_pipeline:
        if not expected_content.strip():
            st.warning("⚠️ Please fill in the Expected Content before analysing.")
        else:
            st.session_state.pipeline_running = True

            # ── Sub-step A: Transcription ──────────────────
            with st.status("🔊 Step 3a — Speech-to-Text (Whisper)", expanded=True) as status_stt:
                st.write("Loading Whisper model…")
                time.sleep(0.3)
                st.write("Transcribing audio…")
                try:
                    transcript = speech_to_text(st.session_state.saved_path)
                    st.session_state.transcript = transcript
                    status_stt.update(label="✅ Step 3a — Transcription complete", state="complete")
                except Exception as e:
                    status_stt.update(label="❌ Transcription failed", state="error")
                    st.error(str(e))
                    st.session_state.pipeline_running = False
                    st.stop()

            st.info(f"**Transcript:** {st.session_state.transcript}")

            # ── Sub-step B: LLM Evaluation ─────────────────
            with st.status("🤖 Step 3b — LLM Relevance Evaluation (Gemini)", expanded=True) as status_llm:
                st.write("Sending transcript to Gemini 2.5 Flash…")
                try:
                    result = judge_relevance(expected_content, st.session_state.transcript)
                    st.session_state.evaluation_result = result
                    status_llm.update(label="✅ Step 3b — Evaluation complete", state="complete")
                except Exception as e:
                    status_llm.update(label="❌ Evaluation failed", state="error")
                    st.error(str(e))
                    st.session_state.pipeline_running = False
                    st.stop()

            st.session_state.pipeline_running = False
            st.rerun()


# ── STEP 4 : Results ──────────────────────────────────────
if st.session_state.transcript is not None and st.session_state.evaluation_result is None:
    # Transcript shown but evaluation still pending (shouldn't normally show,
    # but keeps UI consistent if page reruns mid-flow)
    st.markdown("---")
    st.subheader("📝 Transcript")
    st.info(f"**Transcript:** {st.session_state.transcript}")

if st.session_state.evaluation_result is not None:
    st.markdown("---")
    st.subheader("📊 Step 4 — Evaluation Results")

    result = st.session_state.evaluation_result
    score = result.get("score", 0)
    reason = result.get("reason", "No reason provided")

    # Transcript recap
    st.info(f"**Transcript:** {st.session_state.transcript}")

    # Score with colour coding
    if score >= 80:
        st.success(f"🎯 **Relevance Score: {score} / 100**")
    elif score >= 60:
        st.warning(f"⚠️ **Relevance Score: {score} / 100**")
    else:
        st.error(f"❌ **Relevance Score: {score} / 100**")

    st.info(f"**Explanation:** {reason}")

    # Progress bar for visual flair
    st.progress(score / 100)


# ── Reset ─────────────────────────────────────────────────
if st.session_state.saved_path is not None:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Record New Audio", type="secondary", use_container_width=True):
            for key in ["saved_path", "transcript", "evaluation_result", "pipeline_running"]:
                st.session_state[key] = None if key != "pipeline_running" else False
            st.rerun()


# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🎙️ Smart Speech Evaluator | Powered by Whisper + Gemini 2.5 Flash"
    "</div>",
    unsafe_allow_html=True
)