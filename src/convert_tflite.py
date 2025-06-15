import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model('model/signal_classifier.keras')

# Create a TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# (Optional) Enable quantization for smaller, faster model
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open('model/signal_classifier.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite model saved!")