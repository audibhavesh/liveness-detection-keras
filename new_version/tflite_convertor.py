import tensorflow as tf

if __name__ == '__main__':
    # Load the Keras model
    # model = tf.keras.models.load_model('liveness_export/saved_model.pb')

    # Create a TFLite converter object from the Keras model
    converter = tf.lite.TFLiteConverter.from_saved_model('liveness_export')

    # Optional: Apply optimizations (e.g., quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Convert the model to TFLite format
    tflite_model = converter.convert()

    # Save the TFLite model
    with open('liveness_model.tflite', 'wb') as f:
        f.write(tflite_model)
