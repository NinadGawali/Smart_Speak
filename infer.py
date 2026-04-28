"""Run speech-emotion inference for a single audio file.

This script loads one of the saved deep-learning models shipped with the repo
and prints the predicted emotion label for a new .wav file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from speechemotionrecognition.dnn import CNN, LSTM
from speechemotionrecognition.utilities import get_feature_vector_from_mfcc


LABELS: Tuple[str, ...] = ("Neutral", "Angry", "Happy", "Sad")
DEFAULT_MODEL_PATHS: Dict[str, str] = {
    "cnn": "models/best_model_CNN.h5",
    "lstm": "models/best_model_LSTM.h5",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict the emotion label for a .wav file."
    )
    parser.add_argument("audio_file", help="Path to the input .wav file")
    parser.add_argument(
        "--model",
        choices=("cnn", "lstm"),
        default="cnn",
        help="Model architecture to use for inference (default: cnn)",
    )
    parser.add_argument(
        "--weights",
        help="Path to saved model weights. Defaults to the repo's bundled weights for the selected model.",
    )
    parser.add_argument(
        "--mfcc-len",
        type=int,
        default=39,
        help="MFCC feature length used to build the input vector (default: 39)",
    )
    return parser


def resolve_weights_path(model_name: str, weights_override: str | None) -> Path:
    repo_root = Path(__file__).resolve().parent
    if weights_override:
        return Path(weights_override).expanduser().resolve()
    return (repo_root / DEFAULT_MODEL_PATHS[model_name]).resolve()


def load_model(model_name: str, feature_shape: Tuple[int, ...], weights_path: Path):
    if model_name == "cnn":
        model = CNN(input_shape=feature_shape, num_classes=len(LABELS))
    else:
        model = LSTM(input_shape=feature_shape, num_classes=len(LABELS))

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model.restore_model(str(weights_path))
    return model


def predict_emotion(model_name: str, model, sample: np.ndarray) -> int:
    if model_name == "cnn":
        batch = np.expand_dims(sample, axis=(0, -1))
    elif model_name == "lstm":
        batch = np.expand_dims(sample, axis=0)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    prediction = model.model.predict(batch, verbose=0)
    return int(np.argmax(prediction, axis=1)[0])


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    audio_path = Path(args.audio_file).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.suffix.lower() != ".wav":
        raise ValueError("The input file must be a .wav file")

    sample = get_feature_vector_from_mfcc(
        str(audio_path), flatten=False, mfcc_len=args.mfcc_len
    )
    weights_path = resolve_weights_path(args.model, args.weights)
    model = load_model(args.model, sample.shape, weights_path)
    emotion_index = predict_emotion(args.model, model, sample)

    print(f"audio: {audio_path}")
    print(f"model: {args.model}")
    print(f"weights: {weights_path}")
    print(f"prediction: {emotion_index} ({LABELS[emotion_index]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())