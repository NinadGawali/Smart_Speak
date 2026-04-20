from pathlib import Path
import argparse

import numpy as np

try:
    from .load_model import load_cnn_model
    from .preprocess import preprocess_audio
except ImportError:
    # Allow running this file directly from the src directory.
    from load_model import load_cnn_model
    from preprocess import preprocess_audio

_SRC_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _SRC_DIR.parent / "models"
_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_cnn_model(
            str(_MODELS_DIR / "CNN_model.json"),
            str(_MODELS_DIR / "CNN_model_weights.h5"),
        )
    return _MODEL

# Emotion labels (update if different in original notebook)
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

def predict(file_path):

    features = preprocess_audio(file_path)

    # Add batch dimension
    features = np.expand_dims(features, axis=0)

    predictions = _get_model().predict(features)

    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    emotion = emotion_labels[predicted_class]

    return emotion, float(confidence)


def main():
    parser = argparse.ArgumentParser(
        description="Predict emotion from an audio file."
    )
    parser.add_argument("audio_path", help="Path to input audio file")
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    emotion, confidence = predict(str(audio_path))
    print(f"Predicted emotion: {emotion}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
