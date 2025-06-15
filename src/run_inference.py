import numpy as np
import tensorflow as tf

# Functions

def generate_sine_wave(length=128, freq=5, sampling_rate=50):
    t = np.linspace(0, length/sampling_rate, length, endpoint=False)
    sine_wave = np.sin(2 * np.pi * freq * t)
    return sine_wave

def generate_square_wave(length=128, freq=5, sampling_rate=50):
    t = np.linspace(0, length / sampling_rate, length, endpoint=False)
    sine_wave = np.sin(2 * np.pi * freq * t)
    square_wave = np.where(sine_wave >= 0, 1, -1)
    return square_wave

# Define your classes (same order as training)
classes = ['sine', 'square', 'noise']

# Prepare input signal

# Sine
#signal = generate_sine_wave()
#input_data = signal.reshape(1, 128, 1).astype(np.float32)

# Square
#signal = generate_square_wave()
#input_data = signal.reshape(1, 128, 1).astype(np.float32)

# Noise
signal = np.random.normal(0, 1, 128)
input_data = signal.reshape(1, 128, 1).astype(np.float32)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model/signal_classifier.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output tensor (probabilities)
output_data = interpreter.get_tensor(output_details[0]['index'])

# Get predicted class index and name
predicted_class = np.argmax(output_data)
print(f"Predicted class: {classes[predicted_class]}")
print(f"Output probabilities: {output_data}")