import tensorflow as tf
import json

def load_cnn_model(json_path, weights_path):

    # Enable legacy serialization compatibility
    tf.keras.config.enable_legacy_serialization()

    with open(json_path, "r") as json_file:
        model_json = json_file.read()

    model = tf.keras.models.model_from_json(model_json)

    model.load_weights(weights_path)

    print("Model loaded successfully!")
    return model
