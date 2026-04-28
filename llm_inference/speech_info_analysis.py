# ==========================================================
# Imports
# ==========================================================

import datetime
import json
import os
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import requests
import soundfile as sf
import streamlit as st
import torch
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# ==========================================================
# Page Config
# ==========================================================

st.set_page_config(page_title="Smart Speech Evaluator + RAG", page_icon="🎙️", layout="wide")


# ==========================================================
# Load .env from project root
# ==========================================================

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMOTION_SERVICE_URL = os.getenv("EMOTION_SERVICE_URL", "http://127.0.0.1:5001")


# ==========================================================
# Constants
# ==========================================================

RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

STEPS = {
    1: "Reference Setup",
    2: "Audio Input",
    3: "Transcription",
    4: "RAG Retrieval",
    5: "Detailed Scoring",
}


# ==========================================================
# Cached Model Loaders
# ==========================================================

@st.cache_resource
def load_stt_model():
    """Load Whisper once for repeated transcriptions."""
    return pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-base",
        device=0 if torch.cuda.is_available() else -1,
        chunk_length_s=30,
        stride_length_s=5,
    )


@st.cache_resource
def load_embedding_model():
    """Open-source sentence-transformer model for embeddings."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ==========================================================
# Helpers
# ==========================================================

def save_audio_bytes(raw_bytes: bytes, suffix: str = ".wav") -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = RECORDINGS_DIR / f"recording_{timestamp}{suffix}"
    with open(file_path, "wb") as file_handle:
        file_handle.write(raw_bytes)
    return file_path


def load_audio_bytes(file_path: Path) -> bytes:
    with open(file_path, "rb") as file_handle:
        return file_handle.read()


def read_text_file(uploaded_file) -> str:
    raw = uploaded_file.getvalue()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="ignore")


def speech_to_text(file_path: Path) -> str:
    stt = load_stt_model()
    audio_data, sample_rate = sf.read(str(file_path), dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    result = stt({"array": audio_data, "sampling_rate": sample_rate})
    return str(result.get("text", "")).strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def build_faiss_index(chunks: list[str]) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    model = load_embedding_model()
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings


def retrieve_context(query: str, index: faiss.IndexFlatIP, chunks: list[str], k: int = 4) -> list[dict[str, Any]]:
    if not query.strip() or not chunks:
        return []

    model = load_embedding_model()
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)

    k = min(k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, chunk_idx in zip(scores[0], indices[0]):
        if chunk_idx == -1:
            continue
        results.append(
            {
                "chunk_id": int(chunk_idx),
                "similarity": float(score),
                "text": chunks[int(chunk_idx)],
            }
        )
    return results


def extract_json_from_response(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```json"):
        candidate = candidate[7:]
    if candidate.startswith("```"):
        candidate = candidate[3:]
    if candidate.endswith("```"):
        candidate = candidate[:-3]

    candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def tokenize_for_scoring(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z]{3,}", text.lower())}


def local_scoring(expected: str, transcript: str, retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    expected_tokens = tokenize_for_scoring(expected)
    transcript_tokens = tokenize_for_scoring(transcript)

    if expected_tokens:
        overlap_ratio = len(expected_tokens & transcript_tokens) / len(expected_tokens)
    else:
        overlap_ratio = 0.0

    retrieved_text = " ".join(item["text"] for item in retrieved)
    retrieved_tokens = tokenize_for_scoring(retrieved_text)
    rag_alignment = 0.0
    if retrieved_tokens:
        rag_alignment = len(retrieved_tokens & transcript_tokens) / len(retrieved_tokens)

    relevance = int(min(100, max(0, round(overlap_ratio * 100))))
    coverage = int(min(100, max(0, round((0.6 * overlap_ratio + 0.4 * rag_alignment) * 100))))
    accuracy = int(min(100, max(0, round((0.5 * overlap_ratio + 0.5 * rag_alignment) * 100))))
    clarity = int(min(100, max(30, min(95, len(transcript.split()) * 2))))
    overall = int(round((relevance * 0.35) + (coverage * 0.25) + (accuracy * 0.25) + (clarity * 0.15)))

    return {
        "overall_score": overall,
        "scores": {
            "topic_relevance": relevance,
            "key_point_coverage": coverage,
            "content_accuracy": accuracy,
            "clarity": clarity,
        },
        "step_by_step_analysis": [
            {
                "step": "Topic Relevance",
                "score": relevance,
                "analysis": "Token overlap between expected content and transcript.",
            },
            {
                "step": "Key Point Coverage",
                "score": coverage,
                "analysis": "Checks how many expected ideas are reflected with RAG context support.",
            },
            {
                "step": "Content Accuracy",
                "score": accuracy,
                "analysis": "Measures alignment between transcript and retrieved knowledge chunks.",
            },
            {
                "step": "Clarity",
                "score": clarity,
                "analysis": "Estimated from transcript length and completeness.",
            },
        ],
        "strengths": ["Contains relevant terms from expected content."] if relevance >= 70 else ["Transcript captured basic topic terms."],
        "improvements": ["Add more specific key points from expected content."] if coverage < 70 else ["Improve precision for edge details."],
        "final_summary": "Generated via local fallback scorer (Gemini unavailable).",
    }


def get_wav_for_emotion(audio_path: Path) -> Path:
    """Ensure emotion backend receives a wav path usable by scipy wav reader."""
    if audio_path.suffix.lower() == ".wav":
        return audio_path

    converted_path = audio_path.with_name(f"{audio_path.stem}_emotion.wav")
    audio_data, sample_rate = sf.read(str(audio_path), dtype="float32")
    sf.write(str(converted_path), audio_data, sample_rate, subtype="PCM_16")
    return converted_path


def call_emotion_service(
    audio_path: Path,
    model_name: str = "cnn",
    timeout_sec: int = 60,
) -> dict[str, Any]:
    payload = {"audio_path": str(audio_path)}
    params = {"model": model_name, "mfcc_len": "39"}
    response = requests.post(
        f"{EMOTION_SERVICE_URL.rstrip('/')}/predict",
        json=payload,
        params=params,
        timeout=timeout_sec,
    )
    if response.status_code != 200:
        try:
            error_payload = response.json()
            message = error_payload.get("error", response.text)
        except Exception:
            message = response.text
        raise RuntimeError(f"Emotion backend error ({response.status_code}): {message}")
    return response.json()


def tone_score_from_result(predicted_tone: str, confidence: float, expected_tone: str) -> int:
    expected = (expected_tone or "Any").strip().title()
    predicted = (predicted_tone or "").strip().title()
    confidence = float(max(0.0, min(1.0, confidence)))

    if expected == "Any":
        return int(round(50 + (confidence * 50)))
    if predicted == expected:
        return int(round(70 + (confidence * 30)))
    return int(round((1.0 - confidence) * 30))


def merge_context_and_tone(context_score: int, tone_score: int, tone_weight: float = 0.25) -> int:
    context_weight = 1.0 - tone_weight
    merged = (context_score * context_weight) + (tone_score * tone_weight)
    return int(round(max(0.0, min(100.0, merged))))


def llm_scoring(expected: str, transcript: str, retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    if not GEMINI_API_KEY:
        return local_scoring(expected, transcript, retrieved)

    rag_context = "\n\n".join(
        [f"[Chunk {item['chunk_id']} | sim={item['similarity']:.3f}] {item['text']}" for item in retrieved]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=GEMINI_API_KEY,
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a strict speech evaluator using RAG evidence.

Expected Content:
{expected}

Spoken Transcript:
{spoken}

Retrieved RAG Evidence:
{rag_context}

Evaluate in a step-by-step manner and return JSON only.

Required JSON schema:
{{
  "overall_score": <int 0-100>,
  "scores": {{
    "topic_relevance": <int 0-100>,
    "key_point_coverage": <int 0-100>,
    "content_accuracy": <int 0-100>,
    "clarity": <int 0-100>
  }},
  "step_by_step_analysis": [
    {{"step": "Topic Relevance", "score": <int>, "analysis": "..."}},
    {{"step": "Key Point Coverage", "score": <int>, "analysis": "..."}},
    {{"step": "Content Accuracy", "score": <int>, "analysis": "..."}},
    {{"step": "Clarity", "score": <int>, "analysis": "..."}}
  ],
  "strengths": ["...", "..."],
  "improvements": ["...", "..."],
  "final_summary": "..."
}}

Rules:
- Scores must be integers between 0 and 100.
- Be concise and evidence-driven.
- Do not add any text outside JSON.
"""
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke(
        {
            "expected": expected.strip(),
            "spoken": transcript.strip(),
            "rag_context": rag_context.strip(),
        }
    )
    return extract_json_from_response(response)


def initialize_state() -> None:
    defaults = {
        "current_step": 1,
        "expected_content": "",
        "expected_tone": "Any",
        "knowledge_source_mode": "Paste text",
        "knowledge_text": "",
        "saved_path": None,
        "transcript": "",
        "chunks": [],
        "faiss_index": None,
        "retrieved_passages": [],
        "emotion_result": None,
        "analysis_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go_to_step(step_number: int) -> None:
    st.session_state.current_step = max(1, min(5, step_number))


def reset_after_reference_change() -> None:
    st.session_state.chunks = []
    st.session_state.faiss_index = None
    st.session_state.retrieved_passages = []
    st.session_state.emotion_result = None
    st.session_state.analysis_result = None


def reset_all() -> None:
    for key in [
        "current_step",
        "expected_content",
        "expected_tone",
        "knowledge_source_mode",
        "knowledge_text",
        "saved_path",
        "transcript",
        "chunks",
        "faiss_index",
        "retrieved_passages",
        "emotion_result",
        "analysis_result",
    ]:
        if key == "current_step":
            st.session_state[key] = 1
        elif key == "expected_tone":
            st.session_state[key] = "Any"
        elif key == "knowledge_source_mode":
            st.session_state[key] = "Paste text"
        elif key == "chunks":
            st.session_state[key] = []
        elif key == "retrieved_passages":
            st.session_state[key] = []
        else:
            st.session_state[key] = "" if key in ["expected_content", "knowledge_text", "transcript"] else None


# ==========================================================
# Main App
# ==========================================================

initialize_state()

st.title("🎙️ Smart Speech Evaluator with RAG (FAISS)")
st.caption("Embeddings: sentence-transformers/all-MiniLM-L6-v2 | STT: openai/whisper-base")

if not GEMINI_API_KEY:
    st.warning("Gemini API key not found. App will use local fallback scoring instead of LLM scoring.")

progress_value = (st.session_state.current_step - 1) / (len(STEPS) - 1)
st.progress(progress_value)
st.write(f"Current Step: {st.session_state.current_step} - {STEPS[st.session_state.current_step]}")
st.markdown("---")


if st.session_state.current_step == 1:
    st.subheader("Step 1 - Provide Expected Content + Knowledge Base (File or Text)")
    expected_input = st.text_area(
        "Expected speech content",
        value=st.session_state.expected_content,
        height=130,
        placeholder="Describe what the speaker is expected to say.",
    )

    tone_options = ["Any", "Neutral", "Happy", "Sad", "Angry"]
    tone_index = tone_options.index(st.session_state.expected_tone) if st.session_state.expected_tone in tone_options else 0
    expected_tone_input = st.selectbox(
        "Expected tone (for tone scoring)",
        options=tone_options,
        index=tone_index,
    )

    source_mode = st.radio(
        "Knowledge base source for RAG",
        options=["Paste text", "Upload file (.txt, .md, .csv)"],
        index=0 if st.session_state.knowledge_source_mode == "Paste text" else 1,
        horizontal=True,
    )

    kb_text = ""
    if source_mode == "Paste text":
        kb_text = st.text_area(
            "Paste knowledge text",
            value=st.session_state.knowledge_text if st.session_state.knowledge_source_mode == "Paste text" else "",
            height=220,
            placeholder="Paste source content used for retrieval and evidence-based scoring.",
        )
    else:
        uploaded_doc = st.file_uploader("Upload text-based file", type=["txt", "md", "csv"])
        if uploaded_doc is not None:
            kb_text = read_text_file(uploaded_doc)
            st.text_area("Extracted file preview", value=kb_text[:3000], height=220, disabled=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Save Step 1", type="primary", use_container_width=True):
            if not expected_input.strip():
                st.error("Expected speech content is required.")
            elif not kb_text.strip():
                st.error("Knowledge base text is required for RAG.")
            else:
                st.session_state.expected_content = expected_input.strip()
                st.session_state.expected_tone = expected_tone_input
                st.session_state.knowledge_text = kb_text.strip()
                st.session_state.knowledge_source_mode = "Paste text" if source_mode == "Paste text" else "Upload file"
                reset_after_reference_change()
                st.success("Step 1 saved successfully.")

    with col_b:
        if st.button("Next Step -> Audio Input", use_container_width=True):
            if not st.session_state.expected_content or not st.session_state.knowledge_text:
                st.warning("Please save Step 1 first.")
            else:
                go_to_step(2)
                st.rerun()


elif st.session_state.current_step == 2:
    st.subheader("Step 2 - Record or Upload Audio")

    audio_input = st.audio_input("Record your speech")
    if audio_input is not None:
        try:
            st.session_state.saved_path = save_audio_bytes(audio_input.getvalue(), suffix=".wav")
            st.success(f"Recorded audio saved: {st.session_state.saved_path.name}")
        except Exception as exc:
            st.error(f"Unable to save recorded audio: {exc}")

    uploaded_audio = st.file_uploader("Or upload audio file", type=["wav", "mp3", "m4a", "flac"])
    if uploaded_audio is not None:
        suffix = Path(uploaded_audio.name).suffix or ".wav"
        try:
            st.session_state.saved_path = save_audio_bytes(uploaded_audio.getvalue(), suffix=suffix)
            st.success(f"Uploaded audio saved: {st.session_state.saved_path.name}")
        except Exception as exc:
            st.error(f"Unable to save uploaded audio: {exc}")

    if st.session_state.saved_path:
        st.audio(load_audio_bytes(st.session_state.saved_path))
        st.caption(f"Current audio file: {st.session_state.saved_path.name}")

    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("<- Back to Step 1", use_container_width=True):
            go_to_step(1)
            st.rerun()
    with col_next:
        if st.button("Next Step -> Transcription", type="primary", use_container_width=True):
            if not st.session_state.saved_path:
                st.warning("Please record or upload audio first.")
            else:
                go_to_step(3)
                st.rerun()


elif st.session_state.current_step == 3:
    st.subheader("Step 3 - Transcribe Speech")

    if not st.session_state.saved_path:
        st.warning("No audio found. Go back to Step 2.")
    else:
        st.audio(load_audio_bytes(st.session_state.saved_path))

        if st.button("Run Transcription", type="primary"):
            with st.status("Transcribing with Whisper...", expanded=True) as status:
                try:
                    transcript = speech_to_text(st.session_state.saved_path)
                    st.session_state.transcript = transcript
                    st.session_state.retrieved_passages = []
                    st.session_state.emotion_result = None
                    st.session_state.analysis_result = None
                    status.update(label="Transcription completed.", state="complete")
                except Exception as exc:
                    status.update(label="Transcription failed.", state="error")
                    st.error(str(exc))

        if st.session_state.transcript:
            st.text_area("Transcript", value=st.session_state.transcript, height=200)

    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("<- Back to Step 2", use_container_width=True):
            go_to_step(2)
            st.rerun()
    with col_next:
        if st.button("Next Step -> Build RAG", type="primary", use_container_width=True):
            if not st.session_state.transcript.strip():
                st.warning("Please transcribe audio first.")
            else:
                go_to_step(4)
                st.rerun()


elif st.session_state.current_step == 4:
    st.subheader("Step 4 - Build FAISS Index and Retrieve Context")
    st.caption("RAG retrieval uses sentence-transformers embeddings and FAISS similarity search.")

    st.write(f"Knowledge text length: {len(st.session_state.knowledge_text)} characters")
    st.write(f"Transcript length: {len(st.session_state.transcript)} characters")

    if st.button("Build FAISS Index + Retrieve Top Chunks", type="primary"):
        with st.status("Building FAISS index and running retrieval...", expanded=True) as status:
            try:
                st.session_state.chunks = chunk_text(st.session_state.knowledge_text)
                if not st.session_state.chunks:
                    raise ValueError("No chunks were generated from knowledge text.")

                index, _ = build_faiss_index(st.session_state.chunks)
                st.session_state.faiss_index = index

                query = f"Expected: {st.session_state.expected_content}\nSpoken: {st.session_state.transcript}"
                st.session_state.retrieved_passages = retrieve_context(
                    query=query,
                    index=st.session_state.faiss_index,
                    chunks=st.session_state.chunks,
                    k=4,
                )
                status.update(label="RAG retrieval completed.", state="complete")
            except Exception as exc:
                status.update(label="RAG retrieval failed.", state="error")
                st.error(str(exc))

    if st.session_state.retrieved_passages:
        st.success("Retrieved evidence passages:")
        for i, passage in enumerate(st.session_state.retrieved_passages, start=1):
            st.markdown(
                f"**Top {i} | Chunk {passage['chunk_id']} | Similarity {passage['similarity']:.3f}**\n\n"
                f"{passage['text']}"
            )

    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("<- Back to Step 3", use_container_width=True):
            go_to_step(3)
            st.rerun()
    with col_next:
        if st.button("Next Step -> Detailed Scoring", type="primary", use_container_width=True):
            if not st.session_state.retrieved_passages:
                st.warning("Run RAG retrieval first.")
            else:
                go_to_step(5)
                st.rerun()


elif st.session_state.current_step == 5:
    st.subheader("Step 5 - Step-by-Step Analysis and Scoring")

    if st.button("Run Detailed Analysis", type="primary"):
        with st.status("Scoring transcript with RAG evidence...", expanded=True) as status:
            try:
                st.session_state.analysis_result = llm_scoring(
                    expected=st.session_state.expected_content,
                    transcript=st.session_state.transcript,
                    retrieved=st.session_state.retrieved_passages,
                )
                wav_path = get_wav_for_emotion(Path(st.session_state.saved_path))
                st.session_state.emotion_result = call_emotion_service(wav_path, model_name="cnn")
                status.update(label="Detailed scoring completed.", state="complete")
            except Exception as exc:
                status.update(label="One or more scoring components failed. Running robust fallback.", state="error")
                st.warning(str(exc))
                if st.session_state.analysis_result is None:
                    st.session_state.analysis_result = local_scoring(
                        st.session_state.expected_content,
                        st.session_state.transcript,
                        st.session_state.retrieved_passages,
                    )
                if st.session_state.emotion_result is None:
                    st.session_state.emotion_result = {
                        "emotion_label": "Unknown",
                        "confidence": 0.0,
                        "probabilities": {},
                    }

    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        overall = int(result.get("overall_score", 0))
        scores = result.get("scores", {})
        emotion = st.session_state.emotion_result or {}
        predicted_tone = str(emotion.get("emotion_label", "Unknown"))
        tone_confidence = float(emotion.get("confidence", 0.0))
        tone_score = tone_score_from_result(
            predicted_tone=predicted_tone,
            confidence=tone_confidence,
            expected_tone=st.session_state.expected_tone,
        )
        combined_score = merge_context_and_tone(overall, tone_score, tone_weight=0.25)

        st.markdown("### Overall Score")
        st.metric("Overall", f"{overall}/100")
        st.progress(max(0.0, min(1.0, overall / 100.0)))

        st.markdown("### Tone Recognition (Flask Backend)")
        st.metric("Predicted Tone", predicted_tone)
        st.metric("Tone Confidence", f"{tone_confidence:.2f}")
        st.metric("Tone Score", f"{tone_score}/100")
        st.caption(f"Expected tone: {st.session_state.expected_tone}")

        st.markdown("### Combined Score")
        st.metric("Context + Tone", f"{combined_score}/100")
        st.progress(max(0.0, min(1.0, combined_score / 100.0)))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Topic Relevance", f"{int(scores.get('topic_relevance', 0))}/100")
        c2.metric("Key Point Coverage", f"{int(scores.get('key_point_coverage', 0))}/100")
        c3.metric("Content Accuracy", f"{int(scores.get('content_accuracy', 0))}/100")
        c4.metric("Clarity", f"{int(scores.get('clarity', 0))}/100")

        st.markdown("### Step-by-Step Analysis")
        for item in result.get("step_by_step_analysis", []):
            st.markdown(
                f"**{item.get('step', 'Step')} - {int(item.get('score', 0))}/100**\n\n"
                f"{item.get('analysis', '')}"
            )

        st.markdown("### Strengths")
        for strength in result.get("strengths", []):
            st.write(f"- {strength}")

        st.markdown("### Improvements")
        for improvement in result.get("improvements", []):
            st.write(f"- {improvement}")

        st.markdown("### Final Summary")
        st.info(result.get("final_summary", ""))

        st.markdown("### Transcript")
        st.text_area("Transcribed speech", value=st.session_state.transcript, height=150, disabled=True)

        st.markdown("### Retrieved Evidence Used for Scoring")
        for i, passage in enumerate(st.session_state.retrieved_passages, start=1):
            st.markdown(f"**Evidence {i} | similarity {passage['similarity']:.3f}**\n\n{passage['text']}")

        if emotion.get("probabilities"):
            st.markdown("### Tone Class Probabilities")
            prob_map = emotion.get("probabilities", {})
            for label in ["Neutral", "Angry", "Happy", "Sad"]:
                if label in prob_map:
                    st.write(f"- {label}: {float(prob_map[label]):.4f}")

        st.download_button(
            label="Download Analysis JSON",
            data=json.dumps(result, indent=2),
            file_name="speech_analysis_result.json",
            mime="application/json",
        )

    col_prev, col_reset = st.columns(2)
    with col_prev:
        if st.button("<- Back to Step 4", use_container_width=True):
            go_to_step(4)
            st.rerun()
    with col_reset:
        if st.button("Start New Evaluation", type="primary", use_container_width=True):
            reset_all()
            st.rerun()


st.markdown("---")
st.caption("Smart Speech Evaluator | Whisper + SentenceTransformers + FAISS + Gemini (optional)")