def load_cnn_model(json_path, weights_path):
    import tensorflow as tf

    # Keep compatibility across TensorFlow/Keras versions where this API may differ.
    try:
        tf.keras.config.enable_legacy_serialization()
    except AttributeError:
        pass

    with open(json_path, "r") as json_file:
        model_json = json_file.read()

    model = tf.keras.models.model_from_json(model_json)

    model.load_weights(weights_path)

    print("Model loaded successfully!")
    return model
