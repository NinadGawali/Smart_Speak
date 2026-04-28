"""Flask backend for speech emotion (tone) recognition.

Runs inference using the same CNN/LSTM pipeline used by infer.py and exposes
an HTTP API for Streamlit integration.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from infer import LABELS, load_model, predict_emotion, resolve_weights_path
from speechemotionrecognition.utilities import get_feature_vector_from_mfcc


app = Flask(__name__)


@lru_cache(maxsize=16)
def _load_cached_model(
    model_name: str,
    input_shape: tuple[int, ...],
    weights_path: str,
):
    """Load and cache Keras model instances to avoid reload per request."""
    return load_model(model_name, input_shape, Path(weights_path))


def _predict_from_wav(
    audio_file: Path,
    model_name: str,
    mfcc_len: int,
    weights_override: str | None,
) -> dict[str, Any]:
    if audio_file.suffix.lower() != ".wav":
        raise ValueError("Emotion model supports .wav input only.")

    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    sample = get_feature_vector_from_mfcc(str(audio_file), flatten=False, mfcc_len=mfcc_len)
    weights_path = resolve_weights_path(model_name, weights_override)

    model = _load_cached_model(model_name, tuple(sample.shape), str(weights_path))
    predicted_index = predict_emotion(model_name, model, sample)

    if model_name == "cnn":
        batch = sample[None, ..., None]
    else:
        batch = sample[None, ...]
    probabilities = model.model.predict(batch, verbose=0)[0].tolist()

    prob_map = {label: float(probabilities[idx]) for idx, label in enumerate(LABELS)}

    return {
        "emotion_index": int(predicted_index),
        "emotion_label": LABELS[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": prob_map,
        "model": model_name,
        "mfcc_len": mfcc_len,
        "weights": str(weights_path),
        "audio_file": str(audio_file),
    }


@app.get("/health")
def health() -> tuple[Any, int]:
    return jsonify({"status": "ok"}), 200


@app.post("/predict")
def predict() -> tuple[Any, int]:
    """Run emotion prediction from multipart upload or local file path.

    Supported inputs:
    1) multipart/form-data with "audio" file
    2) application/json with {"audio_path": "..."}
    """
    try:
        model_name = request.form.get("model") or request.args.get("model") or "cnn"
        if model_name not in ("cnn", "lstm"):
            return jsonify({"error": "model must be one of: cnn, lstm"}), 400

        mfcc_raw = request.form.get("mfcc_len") or request.args.get("mfcc_len") or "39"
        try:
            mfcc_len = int(mfcc_raw)
        except ValueError:
            return jsonify({"error": "mfcc_len must be an integer"}), 400

        weights_override = request.form.get("weights") or request.args.get("weights")

        if "audio" in request.files:
            uploaded = request.files["audio"]
            if not uploaded.filename:
                return jsonify({"error": "audio file name is missing"}), 400

            uploads_dir = Path(__file__).resolve().parent / "llm_inference" / "recordings"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            upload_path = uploads_dir / f"emotion_upload_{uploaded.filename}"
            uploaded.save(upload_path)

            result = _predict_from_wav(upload_path, model_name, mfcc_len, weights_override)
            return jsonify(result), 200

        payload = request.get_json(silent=True) or {}
        audio_path = payload.get("audio_path")
        if not audio_path:
            return jsonify({"error": "Provide multipart field 'audio' or JSON field 'audio_path'."}), 400

        result = _predict_from_wav(Path(audio_path).expanduser().resolve(), model_name, mfcc_len, weights_override)
        return jsonify(result), 200

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
