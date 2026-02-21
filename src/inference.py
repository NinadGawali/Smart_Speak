import numpy as np
from load_model import load_cnn_model
from preprocess import preprocess_audio

# Load model once
model = load_cnn_model(
    "../models/CNN_model.json",
    "../models/CNN_model_weights.h5"
)

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

    predictions = model.predict(features)

    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    emotion = emotion_labels[predicted_class]

    return emotion, confidence
