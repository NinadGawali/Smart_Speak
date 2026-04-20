import json


def load_cnn_model(json_path, weights_path):
    import tensorflow as tf

    # Keep compatibility across TensorFlow/Keras versions where this API may differ.
    try:
        tf.keras.config.enable_legacy_serialization()
    except AttributeError:
        pass

    with open(json_path, "r") as json_file:
        model_json = json_file.read()

    try:
        model = tf.keras.models.model_from_json(model_json)
    except TypeError:
        # Fallback for older JSON model configs under newer Keras runtimes.
        model_config = json.loads(model_json)
        if model_config.get("class_name") == "Sequential":
            model = tf.keras.Sequential.from_config(model_config["config"])
        else:
            raise

    model.load_weights(weights_path)

    print("Model loaded successfully!")
    return model
